from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .objective import set_global_seed
from .pf_abc import CandidateRecord, make_config_key
from .search_space import SearchSpace


HYPERBAND_METHOD = "hyperband"
ASHA_METHOD = "asha"
HYPERBAND_DISPLAY_NAME = "Hyperband"
ASHA_DISPLAY_NAME = "ASHA"
HYPERBAND_FULL_NAME = "Synchronous Hyperband Successive Halving"
ASHA_FULL_NAME = "Asynchronous Successive Halving"


class MultiFidelitySearchState:
    def __init__(self, budget_epochs: int) -> None:
        self.budget_epochs = max(1, int(budget_epochs))
        self.spent_epochs = 0
        self.archive: List[CandidateRecord] = []
        self.by_resource: Dict[Tuple[tuple, int], CandidateRecord] = {}

    def can_afford(self, fidelity: int) -> bool:
        return self.spent_epochs + int(fidelity) <= self.budget_epochs

    def evaluate(
        self,
        evaluator,
        config: Dict[str, object],
        fidelity: int,
        seed: int,
        args,
        stage: int,
        parent_uid: Optional[str] = None,
    ) -> CandidateRecord:
        fidelity = int(fidelity)
        key = (make_config_key(config), fidelity)
        cached = self.by_resource.get(key)
        if cached is not None:
            return cached

        record = evaluator.evaluate(
            config=config,
            fidelity=fidelity,
            seed=seed,
            args=args,
            resume_ckpt=None,
            save_ckpt=False,
            stage=stage,
            parent_uid=parent_uid,
        )
        self.by_resource[key] = record
        self.archive.append(record)
        self.spent_epochs += fidelity
        return record


def _model_name_from_args(args) -> str:
    return str(getattr(args, "model_name", getattr(args, "model", "iTransformer")))


def _rung_levels(min_epochs: int, max_epochs: int, eta: int) -> List[int]:
    min_epochs = max(1, int(min_epochs))
    max_epochs = max(min_epochs, int(max_epochs))
    eta = max(2, int(eta))
    levels = [min_epochs]
    while levels[-1] < max_epochs:
        nxt = min(max_epochs, max(levels[-1] + 1, int(levels[-1] * eta)))
        if nxt == levels[-1]:
            break
        levels.append(nxt)
    return levels


def _sample_config(
    args,
    search_space: SearchSpace,
    seen_keys: set[tuple],
    include_default_first: bool = False,
) -> Dict[str, object]:
    model_name = _model_name_from_args(args)
    rng: random.Random = args.rng
    default = search_space.get_default_config(model_name)
    default_key = make_config_key(default)
    if include_default_first and default_key not in seen_keys:
        seen_keys.add(default_key)
        return default

    center = search_space.encode_config(default)
    active_names = search_space.get_active_names(model_name, stage=None)
    max_tries = max(1, int(getattr(args, "max_duplicate_tries", 50)))
    fallback = default

    for _ in range(max_tries):
        vector = search_space.sample_vector(
            rng,
            model_name=model_name,
            center=center,
            active_names=active_names,
            radius=None,
        )
        config = search_space.decode_vector(vector, model_name=model_name)
        fallback = config
        key = make_config_key(config)
        if key not in seen_keys:
            seen_keys.add(key)
            return config

    seen_keys.add(make_config_key(fallback))
    return fallback


def _unique_best(records: Sequence[CandidateRecord]) -> List[CandidateRecord]:
    by_key: Dict[tuple, CandidateRecord] = {}
    for record in records:
        key = make_config_key(record.config)
        current = by_key.get(key)
        if current is None or float(record.score) < float(current.score):
            by_key[key] = record
    return list(by_key.values())


def _candidate_pool_for_verify(records: Sequence[CandidateRecord], top_k: int) -> List[CandidateRecord]:
    unique = _unique_best(records)
    if not unique:
        return []
    max_fidelity = max(int(record.fidelity) for record in unique)
    high_fidelity = [record for record in unique if int(record.fidelity) == max_fidelity]
    ranked = sorted(high_fidelity, key=lambda record: (float(record.score), record.uid))
    return ranked[: max(1, min(int(top_k), len(ranked)))]


def _verify_top_configs(
    args,
    evaluator,
    candidates: Sequence[CandidateRecord],
    verify_seed: int,
    method: str,
) -> List[CandidateRecord]:
    verified: List[CandidateRecord] = []
    for cand in candidates:
        record = evaluator.evaluate(
            config=cand.config,
            fidelity=int(args.verify_epochs),
            seed=int(verify_seed),
            args=args,
            resume_ckpt=None,
            save_ckpt=False,
            stage=999,
            parent_uid=cand.uid,
        )
        record.extra["verify_method"] = method
        verified.append(record)
    return sorted(verified, key=lambda record: (float(record.score), record.uid))


def _hyperband_base_brackets(min_epochs: int, max_epochs: int, eta: int) -> List[Dict[str, object]]:
    eta = max(2, int(eta))
    min_epochs = max(1, int(min_epochs))
    max_epochs = max(min_epochs, int(max_epochs))
    if max_epochs <= min_epochs:
        s_max = 0
    else:
        s_max = int(math.floor(math.log(max_epochs / min_epochs, eta)))

    brackets: List[Dict[str, object]] = []
    for s in reversed(range(s_max + 1)):
        n0 = int(math.ceil((s_max + 1) / (s + 1) * (eta ** s)))
        r0 = max(min_epochs, int(round(max_epochs * (eta ** (-s)))))
        rungs = []
        for i in range(s + 1):
            n_i = max(1, int(math.floor(n0 * (eta ** (-i)))))
            r_i = max(min_epochs, int(round(r0 * (eta ** i))))
            if i == s:
                r_i = max_epochs
            rungs.append({"n": n_i, "fidelity": min(max_epochs, r_i)})
        brackets.append({"s": s, "n0": n0, "rungs": rungs})
    return brackets


def _scale_bracket_to_budget(bracket: Dict[str, object], remaining_epochs: int) -> Optional[Dict[str, object]]:
    rungs = list(bracket["rungs"])
    full_cost = sum(int(item["n"]) * int(item["fidelity"]) for item in rungs)
    if full_cost <= remaining_epochs:
        return bracket

    n0 = int(bracket["n0"])
    while n0 > 0:
        scaled = []
        for idx, item in enumerate(rungs):
            n_i = max(1, int(math.floor(n0 / (int(rungs[0]["n"]) / int(item["n"])))))
            scaled.append({"n": n_i, "fidelity": int(item["fidelity"])})
        cost = sum(int(item["n"]) * int(item["fidelity"]) for item in scaled)
        if cost <= remaining_epochs:
            return {"s": int(bracket["s"]), "n0": n0, "rungs": scaled}
        n0 -= 1
    return None


def _run_hyperband_for_seed(args, search_space: SearchSpace, evaluator, search_seed: int) -> Dict[str, object]:
    args.current_search_seed = int(search_seed)
    args.rng = random.Random(int(search_seed))
    set_global_seed(int(search_seed))

    state = MultiFidelitySearchState(args.search_budget_epochs)
    seen_keys: set[tuple] = set()
    base_brackets = _hyperband_base_brackets(args.mf_min_epochs, args.mf_max_epochs, args.mf_eta)
    bracket_summaries: List[Dict[str, object]] = []
    bracket_counter = 0

    while True:
        progressed = False
        for base in base_brackets:
            remaining = state.budget_epochs - state.spent_epochs
            if remaining < int(args.mf_min_epochs):
                break
            bracket = _scale_bracket_to_budget(base, remaining)
            if bracket is None:
                continue

            bracket_counter += 1
            progressed = True
            active_configs = [
                _sample_config(
                    args=args,
                    search_space=search_space,
                    seen_keys=seen_keys,
                    include_default_first=(not seen_keys),
                )
                for _ in range(int(bracket["rungs"][0]["n"]))
            ]
            rung_summaries = []

            for rung_idx, rung in enumerate(bracket["rungs"]):
                fidelity = int(rung["fidelity"])
                evaluated = []
                for config in active_configs:
                    if not state.can_afford(fidelity):
                        break
                    evaluated.append(
                        state.evaluate(
                            evaluator=evaluator,
                            config=config,
                            fidelity=fidelity,
                            seed=int(search_seed),
                            args=args,
                            stage=bracket_counter * 100 + rung_idx,
                        )
                    )

                if not evaluated:
                    break

                ranked = sorted(_unique_best(evaluated), key=lambda record: (float(record.score), record.uid))
                if rung_idx + 1 < len(bracket["rungs"]):
                    next_n = int(bracket["rungs"][rung_idx + 1]["n"])
                    keep_n = max(1, min(next_n, len(ranked)))
                else:
                    keep_n = len(ranked)
                survivors = ranked[:keep_n]
                best = survivors[0]
                rung_summaries.append(
                    {
                        "rung": rung_idx,
                        "fidelity": fidelity,
                        "n_evaluated": len(evaluated),
                        "keep_n": keep_n,
                        "best_score": float(best.score),
                        "best_config": dict(best.config),
                    }
                )
                print(
                    f"[Hyperband] seed={search_seed} bracket={bracket_counter} "
                    f"rung={rung_idx} fidelity={fidelity} evals={len(evaluated)} "
                    f"keep={keep_n} best_val_mse={float(best.val_mse):.6f}",
                    flush=True,
                )
                active_configs = [dict(record.config) for record in survivors]

            if rung_summaries:
                bracket_summaries.append(
                    {
                        "bracket_index": bracket_counter,
                        "s": int(bracket["s"]),
                        "rungs": rung_summaries,
                        "spent_epochs_after_bracket": int(state.spent_epochs),
                    }
                )

        if not progressed:
            break

    verify_seed = int(args.verify_seed if args.verify_seed is not None else search_seed)
    topk = _candidate_pool_for_verify(state.archive, args.verify_top_k)
    verified = _verify_top_configs(args, evaluator, topk, verify_seed, HYPERBAND_METHOD)
    selected = verified[0] if verified else min(state.archive, key=lambda record: (float(record.score), record.uid))

    return {
        "search_seed": int(search_seed),
        "selected_config": dict(selected.config),
        "selected_score": float(selected.score),
        "selected_val_mse": float(selected.val_mse),
        "budget_epochs": int(args.total_budget_epochs),
        "search_budget_epochs": int(args.search_budget_epochs),
        "spent_epochs_before_verify": int(state.spent_epochs),
        "verify_epochs": int(args.verify_epochs),
        "verify_top_k": int(args.verify_top_k),
        "verify_seed": int(verify_seed),
        "bracket_summaries": bracket_summaries,
        "verified": [_verified_to_summary(record) for record in verified],
        "archive": [_record_to_summary(record) for record in state.archive],
        "archive_size": len(state.archive),
    }


def _promotable_record(
    rung_records: Dict[int, List[CandidateRecord]],
    promoted: Dict[int, set[tuple]],
    rung_idx: int,
    eta: int,
) -> Optional[CandidateRecord]:
    records = sorted(_unique_best(rung_records.get(rung_idx, [])), key=lambda record: (float(record.score), record.uid))
    if len(records) < eta:
        return None
    quota = max(1, len(records) // eta)
    for record in records[:quota]:
        key = make_config_key(record.config)
        if key not in promoted.setdefault(rung_idx, set()):
            return record
    return None


def _run_asha_for_seed(args, search_space: SearchSpace, evaluator, search_seed: int) -> Dict[str, object]:
    args.current_search_seed = int(search_seed)
    args.rng = random.Random(int(search_seed))
    set_global_seed(int(search_seed))

    state = MultiFidelitySearchState(args.search_budget_epochs)
    seen_keys: set[tuple] = set()
    rungs = _rung_levels(args.mf_min_epochs, args.mf_max_epochs, args.mf_eta)
    rung_records: Dict[int, List[CandidateRecord]] = {idx: [] for idx in range(len(rungs))}
    promoted: Dict[int, set[tuple]] = {idx: set() for idx in range(len(rungs))}
    trial_count = 0
    event_log: List[Dict[str, object]] = []

    while state.budget_epochs - state.spent_epochs >= int(args.mf_min_epochs):
        action_taken = False

        for rung_idx in reversed(range(len(rungs) - 1)):
            next_fidelity = int(rungs[rung_idx + 1])
            if not state.can_afford(next_fidelity):
                continue
            cand = _promotable_record(rung_records, promoted, rung_idx, int(args.mf_eta))
            if cand is None:
                continue
            promoted[rung_idx].add(make_config_key(cand.config))
            record = state.evaluate(
                evaluator=evaluator,
                config=cand.config,
                fidelity=next_fidelity,
                seed=int(search_seed),
                args=args,
                stage=1000 + rung_idx + 1,
                parent_uid=cand.uid,
            )
            rung_records[rung_idx + 1].append(record)
            event_log.append(
                {
                    "event": "promote",
                    "from_rung": rung_idx,
                    "to_rung": rung_idx + 1,
                    "fidelity": next_fidelity,
                    "score": float(record.score),
                    "config": dict(record.config),
                    "spent_epochs": int(state.spent_epochs),
                }
            )
            print(
                f"[ASHA] seed={search_seed} promote rung={rung_idx}->{rung_idx + 1} "
                f"fidelity={next_fidelity} val_mse={float(record.val_mse):.6f}",
                flush=True,
            )
            action_taken = True
            break

        if action_taken:
            continue

        first_fidelity = int(rungs[0])
        if not state.can_afford(first_fidelity):
            break
        trial_count += 1
        config = _sample_config(
            args=args,
            search_space=search_space,
            seen_keys=seen_keys,
            include_default_first=(trial_count == 1),
        )
        record = state.evaluate(
            evaluator=evaluator,
            config=config,
            fidelity=first_fidelity,
            seed=int(search_seed),
            args=args,
            stage=trial_count,
        )
        rung_records[0].append(record)
        event_log.append(
            {
                "event": "new_trial",
                "trial": trial_count,
                "rung": 0,
                "fidelity": first_fidelity,
                "score": float(record.score),
                "config": dict(record.config),
                "spent_epochs": int(state.spent_epochs),
            }
        )
        print(
            f"[ASHA] seed={search_seed} new_trial={trial_count} "
            f"fidelity={first_fidelity} val_mse={float(record.val_mse):.6f}",
            flush=True,
        )

    verify_seed = int(args.verify_seed if args.verify_seed is not None else search_seed)
    topk = _candidate_pool_for_verify(state.archive, args.verify_top_k)
    verified = _verify_top_configs(args, evaluator, topk, verify_seed, ASHA_METHOD)
    selected = verified[0] if verified else min(state.archive, key=lambda record: (float(record.score), record.uid))

    rung_summaries = []
    for rung_idx, fidelity in enumerate(rungs):
        records = sorted(_unique_best(rung_records.get(rung_idx, [])), key=lambda record: (float(record.score), record.uid))
        if records:
            best = records[0]
            best_score = float(best.score)
            best_config = dict(best.config)
        else:
            best_score = None
            best_config = None
        rung_summaries.append(
            {
                "rung": rung_idx,
                "fidelity": int(fidelity),
                "n_completed": len(records),
                "n_promoted": len(promoted.get(rung_idx, set())),
                "best_score": best_score,
                "best_config": best_config,
            }
        )

    return {
        "search_seed": int(search_seed),
        "selected_config": dict(selected.config),
        "selected_score": float(selected.score),
        "selected_val_mse": float(selected.val_mse),
        "budget_epochs": int(args.total_budget_epochs),
        "search_budget_epochs": int(args.search_budget_epochs),
        "spent_epochs_before_verify": int(state.spent_epochs),
        "rungs": [int(item) for item in rungs],
        "verify_epochs": int(args.verify_epochs),
        "verify_top_k": int(args.verify_top_k),
        "verify_seed": int(verify_seed),
        "rung_summaries": rung_summaries,
        "events": event_log,
        "verified": [_verified_to_summary(record) for record in verified],
        "archive": [_record_to_summary(record) for record in state.archive],
        "archive_size": len(state.archive),
    }


def _record_to_summary(record: CandidateRecord) -> Dict[str, object]:
    return {
        "uid": record.uid,
        "fidelity": int(record.fidelity),
        "score": float(record.score),
        "val_mse": float(record.val_mse),
        "status": record.status,
        "parent_uid": record.parent_uid,
        "train_time": float(record.train_time),
        "config": dict(record.config),
        "extra": dict(record.extra),
    }


def _verified_to_summary(record: CandidateRecord) -> Dict[str, object]:
    return {
        "uid": record.uid,
        "score": float(record.score),
        "val_mse": float(record.val_mse),
        "config": dict(record.config),
        "parent_uid": record.parent_uid,
        "status": record.status,
    }


def _prepare_args(args, search_space: SearchSpace, evaluator) -> None:
    args.search_space = search_space
    args.pf_evaluator = evaluator
    args.mf_min_epochs = max(1, int(getattr(args, "mf_min_epochs", 1)))
    args.mf_max_epochs = max(args.mf_min_epochs, int(getattr(args, "mf_max_epochs", 20)))
    args.mf_eta = max(2, int(getattr(args, "mf_eta", 3)))
    args.verify_top_k = max(1, int(getattr(args, "verify_top_k", 4)))
    args.verify_epochs = max(args.mf_min_epochs, int(getattr(args, "verify_epochs", args.mf_max_epochs)))
    args.total_budget_epochs = max(1, int(getattr(args, "mf_budget_epochs", 312)))
    reserve = args.verify_top_k * args.verify_epochs
    args.search_budget_epochs = max(args.mf_min_epochs, args.total_budget_epochs - reserve)


def run_hyperband(args, search_space: SearchSpace, evaluator):
    return _run_multifidelity_method(args, search_space, evaluator, HYPERBAND_METHOD, _run_hyperband_for_seed)


def run_asha(args, search_space: SearchSpace, evaluator):
    return _run_multifidelity_method(args, search_space, evaluator, ASHA_METHOD, _run_asha_for_seed)


def _run_multifidelity_method(args, search_space: SearchSpace, evaluator, method: str, run_seed_fn):
    _prepare_args(args, search_space, evaluator)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_search_seed = []
    started = time.time()
    for search_seed in args.search_seeds:
        seed_dir = output_dir / f"search_seed_{int(search_seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result = run_seed_fn(args, search_space, evaluator, int(search_seed))
        per_search_seed.append(result)
        (seed_dir / f"{method}_search_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    per_search_seed = sorted(per_search_seed, key=lambda item: float(item["selected_score"]))
    selected = per_search_seed[0]
    selected_best_config = dict(selected["selected_config"])
    display_name = HYPERBAND_DISPLAY_NAME if method == HYPERBAND_METHOD else ASHA_DISPLAY_NAME
    full_name = HYPERBAND_FULL_NAME if method == HYPERBAND_METHOD else ASHA_FULL_NAME

    search_summary = {
        "method": method,
        "display_name": display_name,
        "method_full_name": full_name,
        "mf_min_epochs": int(args.mf_min_epochs),
        "mf_max_epochs": int(args.mf_max_epochs),
        "mf_eta": int(args.mf_eta),
        "budget_epochs": int(args.total_budget_epochs),
        "search_budget_epochs": int(args.search_budget_epochs),
        "verify_top_k": int(args.verify_top_k),
        "verify_epochs": int(args.verify_epochs),
        "selected_search_seed": int(selected["search_seed"]),
        "selected_best_config": selected_best_config,
        "selected_val_mse": float(selected["selected_val_mse"]),
        "per_search_seed": per_search_seed,
        "search_time_sec": float(time.time() - started),
        "archive_size": int(sum(int(item["archive_size"]) for item in per_search_seed)),
    }
    (output_dir / f"{method}_search_summary.json").write_text(
        json.dumps(search_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected_best_config, search_summary
