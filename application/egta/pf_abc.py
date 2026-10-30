from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .exp_ext import Exp_Long_Term_Forecast_Ext
from .objective import count_parameters, set_global_seed
from .search_space import SearchSpace


PF_METHOD = "pf_tahc_abc"
PF_DISPLAY_NAME = "PF-ABC"
PF_METHOD_FULL_NAME = "Progressive-Fidelity Artificial Bee Colony"
_RESUME_SUPPORTED = False


@dataclass
class CandidateRecord:
    uid: str
    config: Dict[str, object]
    encoded: Optional[Any]
    stage: int
    fidelity: int
    score: float
    val_mse: float
    val_mae: Optional[float]
    key_val_mse: Optional[float]
    status: str
    parent_uid: Optional[str]
    ckpt_path: Optional[str]
    train_time: float
    extra: Dict[str, object] = field(default_factory=dict)


def make_config_key(config: dict) -> tuple:
    return (
        int(config["seq_len"]),
        int(config["d_model"]),
        int(config["n_heads"]),
        int(config["e_layers"]),
        int(config["d_ff"]),
        round(float(config["dropout"]), 4),
        round(math.log10(float(config["learning_rate"])), 4),
        int(config["batch_size"]),
        int(config.get("csg_reduction", 4)),
    )


def compute_pf_score(eval_result, args) -> float:
    if hasattr(eval_result, "val_mse"):
        return float(eval_result.val_mse)
    return float(eval_result["val_mse"])


def _model_name_from_args(args) -> str:
    return str(
        getattr(
            args,
            "model_name",
            getattr(args, "model_variant", getattr(args, "model", "iTransformer")),
        )
    )


def _candidate_uid(config: Dict[str, object], stage: int, fidelity: int, seed: int) -> str:
    payload = json.dumps(
        {
            "seed": int(seed),
            "stage": int(stage),
            "fidelity": int(fidelity),
            "config_key": list(make_config_key(config)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _pending_candidate(
    config: Dict[str, object],
    encoded: Optional[Any],
    stage: int,
    seed: int,
    parent_uid: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> CandidateRecord:
    return CandidateRecord(
        uid=_candidate_uid(config, stage=stage, fidelity=0, seed=seed),
        config=dict(config),
        encoded=encoded,
        stage=int(stage),
        fidelity=0,
        score=float("inf"),
        val_mse=float("inf"),
        val_mae=None,
        key_val_mse=None,
        status="pending",
        parent_uid=parent_uid,
        ckpt_path=None,
        train_time=0.0,
        extra=dict(extra or {}),
    )


def _record_to_dict(record: CandidateRecord) -> Dict[str, object]:
    return asdict(record)


def _project_config(config: Dict[str, object], search_space: SearchSpace, model_name: str) -> Dict[str, object]:
    return search_space.project_config(dict(config), model_name=model_name)


def _adjacent_choice(values: Sequence[object], current, rng: random.Random):
    values = list(values)
    if len(values) <= 1:
        return current
    idx = values.index(current)
    if idx <= 0:
        return values[1]
    if idx >= len(values) - 1:
        return values[-2]
    return values[idx + rng.choice([-1, 1])]


def _toggle_choice(values: Sequence[object], current, rng: random.Random):
    values = list(values)
    if len(values) <= 1:
        return current
    if current not in values:
        return values[0]
    others = [value for value in values if value != current]
    if not others:
        return current
    return rng.choice(others)


def _mutate_learning_rate(
    config: Dict[str, object],
    search_space: SearchSpace,
    rng: random.Random,
    sigma: float = 0.25,
) -> None:
    spec = search_space.name_to_spec["learning_rate"]
    log_lr = math.log(float(config["learning_rate"]))
    log_lr_new = log_lr + rng.gauss(0.0, max(0.0, float(sigma)))
    config["learning_rate"] = min(float(spec.high), max(float(spec.low), math.exp(log_lr_new)))


def _mutate_dropout(
    config: Dict[str, object],
    search_space: SearchSpace,
    rng: random.Random,
    sigma: float = 0.03,
) -> None:
    spec = search_space.name_to_spec["dropout"]
    dropout_new = float(config["dropout"]) + rng.gauss(0.0, max(0.0, float(sigma)))
    config["dropout"] = min(float(spec.high), max(float(spec.low), dropout_new))


def _mutate_batch_size(config: Dict[str, object], search_space: SearchSpace, rng: random.Random) -> None:
    spec = search_space.name_to_spec["batch_size"]
    config["batch_size"] = _adjacent_choice(spec.values or [], config["batch_size"], rng)


def _mutate_structure_variable(
    config: Dict[str, object],
    var_name: str,
    search_space: SearchSpace,
    model_name: str,
    rng: random.Random,
) -> None:
    if var_name == "batch_size":
        _mutate_batch_size(config, search_space, rng)
        return
    if var_name == "csg_reduction" and model_name != "iTransformer_CSG":
        config["csg_reduction"] = 4
        return
    spec = search_space.name_to_spec[var_name]
    config[var_name] = _toggle_choice(spec.values or [], config[var_name], rng)


def _sample_unique_random_config(
    args,
    search_space: SearchSpace,
    archive_keys: set[tuple],
    local_keys: set[tuple],
) -> tuple[Dict[str, object], bool]:
    rng: random.Random = args.rng
    model_name = _model_name_from_args(args)
    center = search_space.encode_config(search_space.get_default_config(model_name))
    active_names = search_space.get_active_names(model_name, stage=None)
    max_tries = max(1, int(args.max_duplicate_tries))
    fallback_config = None

    for _ in range(max_tries):
        vector = search_space.sample_vector(
            rng,
            model_name=model_name,
            center=center,
            active_names=active_names,
            radius=None,
        )
        config = search_space.decode_vector(vector, model_name=model_name)
        key = make_config_key(config)
        fallback_config = config
        if key not in archive_keys and key not in local_keys:
            return config, False

    return dict(fallback_config or search_space.get_default_config(model_name)), True


def evaluate_pf_candidate(
    config,
    fidelity,
    seed,
    args,
    resume_ckpt=None,
    save_ckpt=True,
    stage=0,
    parent_uid=None,
):
    """
    Train and validate a candidate at the requested fidelity.

    Resume checkpoints are intentionally ignored in the first PF-ABC version,
    but the argument is kept to preserve the interface for later extensions.
    """

    evaluator = getattr(args, "pf_evaluator", None)
    if evaluator is None:
        raise ValueError("PF evaluator is not attached to args.")
    return evaluator.evaluate(
        config=config,
        fidelity=fidelity,
        seed=seed,
        args=args,
        resume_ckpt=resume_ckpt,
        save_ckpt=save_ckpt,
        stage=stage,
        parent_uid=parent_uid,
    )


def initialize_pf_sources(args, search_space, n):
    """
    Generate the initial PF-ABC candidate pool.
    """

    n = max(0, int(n))
    if n <= 0:
        return []

    rng: random.Random = args.rng
    model_name = _model_name_from_args(args)
    center_config = search_space.get_default_config(model_name)
    center_encoded = search_space.encode_config(center_config)
    seen_keys: set[tuple] = set()
    candidates: List[CandidateRecord] = []
    max_tries = max(1, int(args.max_duplicate_tries))

    center_key = make_config_key(center_config)
    seen_keys.add(center_key)
    candidates.append(
        _pending_candidate(
            config=center_config,
            encoded=center_encoded,
            stage=0,
            seed=int(args.current_search_seed),
            extra={"source": "default_center"},
        )
    )

    while len(candidates) < n:
        config = None
        duplicate = False
        for _ in range(max_tries):
            vector = search_space.sample_vector(
                rng,
                model_name=model_name,
                center=center_encoded,
                active_names=search_space.get_active_names(model_name, stage=None),
                radius=None,
            )
            sampled = search_space.decode_vector(vector, model_name=model_name)
            key = make_config_key(sampled)
            config = sampled
            if key not in seen_keys:
                seen_keys.add(key)
                break
            duplicate = True
        else:
            duplicate = True
            if config is not None:
                print(
                    f"[PF-ABC] warning: duplicate-heavy initialization; keeping repeated config after {max_tries} tries.",
                    flush=True,
                )

        if config is None:
            config = search_space.get_default_config(model_name)
        encoded = search_space.encode_config(config)
        candidates.append(
            _pending_candidate(
                config=config,
                encoded=encoded,
                stage=0,
                seed=int(args.current_search_seed),
                extra={"source": "random_init", "duplicate_warning": int(duplicate)},
            )
        )

    return candidates


def generate_challenger_bees(
    survivors,
    n_challengers,
    args,
    search_space,
    archive_keys,
    stage,
):
    """
    Generate challenger bees around high-ranking survivors.
    """

    n_challengers = max(0, int(n_challengers))
    if n_challengers <= 0 or not survivors:
        return []

    rng: random.Random = args.rng
    model_name = _model_name_from_args(args)
    ranked = sorted(survivors, key=lambda record: (float(record.score), record.uid))
    weights = [1.0 / float(rank + 1) for rank in range(len(ranked))]
    local_keys: set[tuple] = set()
    challengers: List[CandidateRecord] = []
    max_tries = max(1, int(args.max_duplicate_tries))
    local_prob = max(0.0, float(args.challenge_local_prob))
    struct_prob = max(0.0, float(args.challenge_struct_prob))

    for _ in range(n_challengers):
        parent = rng.choices(ranked, weights=weights, k=1)[0]
        config = None
        duplicate_warning = False

        for _ in range(max_tries):
            mutated = copy.deepcopy(parent.config)
            draw = rng.random()
            if draw < local_prob:
                target = rng.choice(["learning_rate", "dropout", "batch_size"])
                if target == "learning_rate":
                    _mutate_learning_rate(mutated, search_space, rng)
                elif target == "dropout":
                    _mutate_dropout(mutated, search_space, rng)
                else:
                    _mutate_batch_size(mutated, search_space, rng)
                mode = "local"
            elif draw < local_prob + struct_prob:
                struct_choices = ["seq_len", "n_heads", "e_layers", "d_ff", "batch_size", "csg_reduction"]
                struct_name = rng.choice(struct_choices)
                opt_name = rng.choice(["learning_rate", "dropout"])
                _mutate_structure_variable(mutated, struct_name, search_space, model_name, rng)
                if opt_name == "learning_rate":
                    _mutate_learning_rate(mutated, search_space, rng)
                else:
                    _mutate_dropout(mutated, search_space, rng)
                mode = "struct"
            else:
                _mutate_learning_rate(mutated, search_space, rng)
                mode = "fallback_local"

            projected = _project_config(mutated, search_space, model_name)
            key = make_config_key(projected)
            config = projected
            if key not in archive_keys and key not in local_keys:
                local_keys.add(key)
                break
        else:
            config, duplicate_warning = _sample_unique_random_config(
                args=args,
                search_space=search_space,
                archive_keys=set(archive_keys),
                local_keys=local_keys,
            )
            local_keys.add(make_config_key(config))
            mode = "random_fallback"
            if duplicate_warning:
                print(
                    f"[PF-ABC] warning: duplicate-heavy challenger generation at stage={stage}; using repeated config fallback.",
                    flush=True,
                )

        encoded = search_space.encode_config(config)
        challengers.append(
            _pending_candidate(
                config=config,
                encoded=encoded,
                stage=stage,
                seed=int(args.current_search_seed),
                parent_uid=parent.uid,
                extra={
                    "source": "challenger",
                    "mode": mode,
                    "duplicate_warning": int(duplicate_warning),
                },
            )
        )

    return challengers


def generate_hf_abc_candidates(
    survivors,
    n_candidates,
    args,
    search_space,
    archive_keys,
    stage,
):
    """
    Generate HF-ABC refinement candidates around elite survivors.
    """

    n_candidates = max(0, int(n_candidates))
    if n_candidates <= 0 or not survivors:
        return []

    rng: random.Random = args.rng
    model_name = _model_name_from_args(args)
    ranked = sorted(survivors, key=lambda record: (float(record.score), record.uid))
    elite_top_m = max(1, min(int(getattr(args, "elite_top_m", 4)), len(ranked)))
    elites = ranked[:elite_top_m]
    weights = [1.0 / float(rank + 1) for rank in range(len(elites))]
    local_keys: set[tuple] = set()
    candidates: List[CandidateRecord] = []
    max_tries = max(1, int(getattr(args, "elite_max_tries", getattr(args, "max_duplicate_tries", 50))))

    refine_ratio = max(0.0, float(getattr(args, "elite_refine_ratio", 0.70)))
    struct_ratio = max(0.0, float(getattr(args, "elite_struct_ratio", 0.20)))
    random_ratio = max(0.0, float(getattr(args, "elite_random_ratio", 0.10)))
    ratio_total = refine_ratio + struct_ratio + random_ratio
    if ratio_total <= 0.0:
        refine_ratio, struct_ratio, random_ratio = 0.70, 0.20, 0.10
        ratio_total = 1.0
    refine_cutoff = refine_ratio / ratio_total
    struct_cutoff = (refine_ratio + struct_ratio) / ratio_total

    lr_sigma = float(getattr(args, "elite_lr_sigma", 0.20))
    dropout_sigma = float(getattr(args, "elite_dropout_sigma", 0.02))
    struct_choices = [
        name
        for name in ["seq_len", "d_model", "n_heads", "e_layers", "d_ff", "csg_reduction"]
        if search_space.is_searchable(name, model_name)
    ]

    for _ in range(n_candidates):
        config = None
        parent = None
        mode = "random_restart"
        duplicate_warning = False

        for _ in range(max_tries):
            draw = rng.random()
            parent = None

            if draw < refine_cutoff:
                mode = "elite_local"
                parent = rng.choices(elites, weights=weights, k=1)[0]
                mutated = copy.deepcopy(parent.config)
                target = rng.choice(["learning_rate", "dropout", "batch_size"])
                if target == "learning_rate":
                    _mutate_learning_rate(mutated, search_space, rng, sigma=lr_sigma)
                elif target == "dropout":
                    _mutate_dropout(mutated, search_space, rng, sigma=dropout_sigma)
                else:
                    _mutate_batch_size(mutated, search_space, rng)
                projected = _project_config(mutated, search_space, model_name)
            elif draw < struct_cutoff and struct_choices:
                mode = "elite_struct"
                parent = rng.choices(elites, weights=weights, k=1)[0]
                mutated = copy.deepcopy(parent.config)
                struct_name = rng.choice(struct_choices)
                opt_name = rng.choice(["learning_rate", "dropout"])
                _mutate_structure_variable(mutated, struct_name, search_space, model_name, rng)
                if opt_name == "learning_rate":
                    _mutate_learning_rate(mutated, search_space, rng, sigma=lr_sigma)
                else:
                    _mutate_dropout(mutated, search_space, rng, sigma=dropout_sigma)
                projected = _project_config(mutated, search_space, model_name)
            else:
                mode = "random_restart"
                projected, duplicate_warning = _sample_unique_random_config(
                    args=args,
                    search_space=search_space,
                    archive_keys=set(archive_keys),
                    local_keys=local_keys,
                )

            key = make_config_key(projected)
            config = projected
            if key not in archive_keys and key not in local_keys:
                local_keys.add(key)
                break
            duplicate_warning = True
        else:
            config, duplicate_warning = _sample_unique_random_config(
                args=args,
                search_space=search_space,
                archive_keys=set(archive_keys),
                local_keys=local_keys,
            )
            local_keys.add(make_config_key(config))
            parent = None
            mode = "random_restart_fallback"
            if duplicate_warning:
                print(
                    f"[HF-ABC] warning: duplicate-heavy refinement generation at stage={stage}; using repeated config fallback.",
                    flush=True,
                )

        encoded = search_space.encode_config(config)
        candidates.append(
            _pending_candidate(
                config=config,
                encoded=encoded,
                stage=stage,
                seed=int(args.current_search_seed),
                parent_uid=parent.uid if parent is not None else None,
                extra={
                    "source": "hf_abc_refinement",
                    "mode": mode,
                    "parent_uid": parent.uid if parent is not None else None,
                    "parent_score": float(parent.score) if parent is not None else None,
                    "stage": int(stage),
                    "duplicate_warning": int(duplicate_warning),
                },
            )
        )

    return candidates


class ProgressiveFidelityEvaluator:
    def __init__(self, base_args, output_dir: str | Path) -> None:
        self.base_args = copy.deepcopy(base_args)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_supported = _RESUME_SUPPORTED

    def evaluate(
        self,
        config: Dict[str, object],
        fidelity: int,
        seed: int,
        args,
        resume_ckpt: Optional[str] = None,
        save_ckpt: bool = True,
        stage: int = 0,
        parent_uid: Optional[str] = None,
    ) -> CandidateRecord:
        model_name = _model_name_from_args(args)
        search_space: Optional[SearchSpace] = getattr(args, "search_space", None)
        projected_config = (
            _project_config(config, search_space, model_name) if search_space is not None else dict(config)
        )
        uid = _candidate_uid(projected_config, stage=stage, fidelity=fidelity, seed=seed)
        local_args = copy.deepcopy(self.base_args)
        for key, value in projected_config.items():
            setattr(local_args, key, value)
        local_args.train_epochs = int(fidelity)
        local_args.patience = min(int(args.search_patience), max(1, int(fidelity)))
        local_args.is_training = 1

        candidate_tag = f"pf_s{seed}_st{stage}_ep{fidelity}_{uuid.uuid4().hex[:6]}"
        setting = None
        ckpt_path = None
        status = "ok"
        val_mse = float("inf")
        score = float("inf")
        train_time = 0.0
        param_count = 0

        extra = {
            "seed": int(seed),
            "resume_requested": bool(resume_ckpt),
            "resume_supported": bool(self.resume_supported),
            "resume_ckpt": resume_ckpt,
            "save_ckpt_requested": bool(save_ckpt),
            "candidate_tag": candidate_tag,
            "model_name": model_name,
        }

        try:
            set_global_seed(int(seed))
            exp = Exp_Long_Term_Forecast_Ext(local_args)
            setting = exp.make_setting(extra_tag=candidate_tag)
            param_count = count_parameters(exp.model)

            start = time.time()
            exp.train(setting)
            train_time = time.time() - start

            vali_data, vali_loader = exp._get_data(flag="val")
            criterion = exp._select_criterion()
            val_mse = float(exp.vali(vali_data, vali_loader, criterion))
            score = compute_pf_score({"val_mse": val_mse}, args)

            if self.resume_supported and save_ckpt and not getattr(args, "pf_no_resume", False):
                ckpt_candidate = Path(local_args.checkpoints) / setting / "checkpoint.pth"
                if ckpt_candidate.exists():
                    ckpt_path = str(ckpt_candidate)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                status = "oom"
                extra["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[PF-ABC] CUDA OOM for stage={stage}, fidelity={fidelity}, config={projected_config}; assigning infinite score.",
                    flush=True,
                )
            else:
                status = "failed"
                extra["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[PF-ABC] evaluation failed for stage={stage}, fidelity={fidelity}, config={projected_config}; assigning infinite score.",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            status = "failed"
            extra["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[PF-ABC] evaluation failed for stage={stage}, fidelity={fidelity}, config={projected_config}; assigning infinite score.",
                flush=True,
            )
        finally:
            if setting:
                extra["setting"] = setting
                ckpt_dir = Path(local_args.checkpoints) / setting
                keep_ckpt = (
                    self.resume_supported
                    and save_ckpt
                    and not getattr(args, "pf_no_resume", False)
                    and ckpt_path is not None
                )
                if ckpt_dir.exists() and not keep_ckpt:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        extra["params"] = int(param_count)
        extra["requested_fidelity"] = int(fidelity)
        extra["effective_patience"] = int(local_args.patience)

        return CandidateRecord(
            uid=uid,
            config=projected_config,
            encoded=search_space.encode_config(projected_config) if search_space is not None else None,
            stage=int(stage),
            fidelity=int(fidelity),
            score=float(score),
            val_mse=float(val_mse),
            val_mae=None,
            key_val_mse=None,
            status=status,
            parent_uid=parent_uid,
            ckpt_path=ckpt_path,
            train_time=float(train_time),
            extra=extra,
        )


def _stage_unique_records(records: Sequence[CandidateRecord]) -> List[CandidateRecord]:
    by_key: Dict[tuple, CandidateRecord] = {}
    for record in records:
        key = make_config_key(record.config)
        current = by_key.get(key)
        if current is None or float(record.score) < float(current.score):
            by_key[key] = record
    return list(by_key.values())


def run_pf_tahc_abc(args, search_space, evaluator):
    """
    PF-ABC main search logic.
    """

    if len(args.race_fidelities) != len(args.race_keep):
        raise ValueError("race_keep length must match race_fidelities length")
    if len(args.race_fidelities) != len(args.race_challengers):
        raise ValueError("race_challengers length must match race_fidelities length")

    verify_seed = args.verify_seed
    if verify_seed is None:
        verify_seed = int(args.search_seeds[0])

    args.search_space = search_space
    args.pf_evaluator = evaluator
    is_hf_abc = bool(getattr(args, "hf_abc", False))
    log_prefix = "HF-ABC" if is_hf_abc else "PF-ABC"

    global_archive: List[CandidateRecord] = []
    selected_per_search_seed: List[Dict[str, object]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for search_seed in args.search_seeds:
        search_seed = int(search_seed)
        args.current_search_seed = search_seed
        args.rng = random.Random(search_seed)
        set_global_seed(search_seed)

        seed_dir = output_dir / f"search_seed_{search_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        archive_by_key: Dict[tuple, CandidateRecord] = {}
        seed_archive: List[CandidateRecord] = []
        active = initialize_pf_sources(
            args=args,
            search_space=search_space,
            n=args.race_init_size,
        )
        stage_summaries: List[Dict[str, object]] = []

        for stage_idx, fidelity in enumerate(args.race_fidelities):
            fidelity = int(fidelity)
            if stage_idx > 0:
                n_chal = int(args.race_challengers[stage_idx])
                if n_chal > 0:
                    if is_hf_abc:
                        challengers = generate_hf_abc_candidates(
                            survivors=active,
                            n_candidates=n_chal,
                            args=args,
                            search_space=search_space,
                            archive_keys=set(archive_by_key.keys()),
                            stage=stage_idx,
                        )
                    else:
                        challengers = generate_challenger_bees(
                            survivors=active,
                            n_challengers=n_chal,
                            args=args,
                            search_space=search_space,
                            archive_keys=set(archive_by_key.keys()),
                            stage=stage_idx,
                        )
                    active = list(active) + challengers

            evaluated: List[CandidateRecord] = []
            for cand in active:
                key = make_config_key(cand.config)
                cached = archive_by_key.get(key)
                if cached is not None and int(cached.fidelity) >= fidelity:
                    rec = cached
                else:
                    rec = evaluate_pf_candidate(
                        config=cand.config,
                        fidelity=fidelity,
                        seed=search_seed,
                        args=args,
                        resume_ckpt=getattr(cand, "ckpt_path", None),
                        save_ckpt=not bool(args.pf_no_resume),
                        stage=stage_idx,
                        parent_uid=cand.parent_uid,
                    )
                    if cand.extra:
                        merged_extra = dict(cand.extra)
                        merged_extra.update(rec.extra)
                        rec.extra = merged_extra
                    archive_by_key[key] = rec
                    seed_archive.append(rec)
                    global_archive.append(rec)
                evaluated.append(rec)

            ranked = sorted(_stage_unique_records(evaluated), key=lambda record: (float(record.score), record.uid))
            keep_n = min(max(1, int(args.race_keep[stage_idx])), len(ranked))
            survivors = ranked[:keep_n]

            best = survivors[0]
            stage_summary = {
                "stage": int(stage_idx),
                "fidelity": int(fidelity),
                "n_active": len(active),
                "n_unique": len(ranked),
                "n_evaluated": len(ranked),
                "keep_n": int(keep_n),
                "best_score": float(best.score),
                "best_val_mse": float(best.val_mse),
                "best_config": dict(best.config),
                "survivor_uids": [record.uid for record in survivors],
            }
            stage_summaries.append(stage_summary)

            print(
                f"[{log_prefix}] seed={search_seed} stage={stage_idx} "
                f"fidelity={fidelity} active={len(active)} keep={keep_n} "
                f"best_val_mse={best.val_mse:.6f}",
                flush=True,
            )

            active = survivors

        ranked_active = sorted(active, key=lambda record: (float(record.score), record.uid))
        verify_top_k = min(max(1, int(args.verify_top_k)), len(ranked_active))
        topk = ranked_active[:verify_top_k]
        verified: List[CandidateRecord] = []

        for cand in topk:
            rec = evaluate_pf_candidate(
                config=cand.config,
                fidelity=int(args.verify_epochs),
                seed=int(verify_seed),
                args=args,
                resume_ckpt=None,
                save_ckpt=False,
                stage=999,
                parent_uid=cand.uid,
            )
            verified.append(rec)
            seed_archive.append(rec)
            global_archive.append(rec)

        verified = sorted(verified, key=lambda record: (float(record.score), record.uid))
        selected = verified[0]
        selected_per_search_seed.append(
            {
                "search_seed": search_seed,
                "selected_config": dict(selected.config),
                "selected_score": float(selected.score),
                "selected_val_mse": float(selected.val_mse),
                "stage_summaries": stage_summaries,
                "verified": [
                    {
                        "uid": record.uid,
                        "score": float(record.score),
                        "val_mse": float(record.val_mse),
                        "config": dict(record.config),
                        "parent_uid": record.parent_uid,
                        "status": record.status,
                    }
                    for record in verified
                ],
                "archive_size": len(seed_archive),
            }
        )

        (seed_dir / "pf_search_summary.json").write_text(
            json.dumps(selected_per_search_seed[-1], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if args.pf_save_archive:
            (seed_dir / "pf_archive.json").write_text(
                json.dumps([_record_to_dict(record) for record in seed_archive], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    selected_per_search_seed = sorted(
        selected_per_search_seed,
        key=lambda item: float(item["selected_score"]),
    )
    overall = selected_per_search_seed[0]
    selected_best_config = dict(overall["selected_config"])

    search_summary = {
        "method": PF_METHOD,
        "display_name": "HF-ABC" if is_hf_abc else PF_DISPLAY_NAME,
        "method_full_name": (
            "Hybrid Fidelity-guided Artificial Bee Colony" if is_hf_abc else PF_METHOD_FULL_NAME
        ),
        "hf_abc": bool(is_hf_abc),
        "race_fidelities": [int(item) for item in args.race_fidelities],
        "race_init_size": int(args.race_init_size),
        "race_keep": [int(item) for item in args.race_keep],
        "race_challengers": [int(item) for item in args.race_challengers],
        "verify_top_k": int(args.verify_top_k),
        "verify_epochs": int(args.verify_epochs),
        "verify_seed": int(verify_seed),
        "selected_search_seed": int(overall["search_seed"]),
        "selected_best_config": selected_best_config,
        "selected_val_mse": float(overall["selected_val_mse"]),
        "per_search_seed": selected_per_search_seed,
        "archive_size": len(global_archive),
    }
    if is_hf_abc:
        search_summary.update(
            {
                "elite_refine_ratio": float(getattr(args, "elite_refine_ratio", 0.70)),
                "elite_struct_ratio": float(getattr(args, "elite_struct_ratio", 0.20)),
                "elite_random_ratio": float(getattr(args, "elite_random_ratio", 0.10)),
                "elite_top_m": int(getattr(args, "elite_top_m", 4)),
                "elite_lr_sigma": float(getattr(args, "elite_lr_sigma", 0.20)),
                "elite_dropout_sigma": float(getattr(args, "elite_dropout_sigma", 0.02)),
                "elite_max_tries": int(getattr(args, "elite_max_tries", getattr(args, "max_duplicate_tries", 50))),
            }
        )

    if args.pf_save_archive:
        (output_dir / "pf_search_summary.json").write_text(
            json.dumps(search_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "pf_search_archive.json").write_text(
            json.dumps([_record_to_dict(record) for record in global_archive], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return selected_best_config, search_summary
