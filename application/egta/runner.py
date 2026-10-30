from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

from .objective import FinalRetrainer, SearchObjective
from .optimizers import (
    EGTAABCSearchOptimizer,
    GeneticSearchOptimizer,
    OriginalABCSearchOptimizer,
    PSOSearchOptimizer,
    RandomSearchOptimizer,
)
from .pf_abc import (
    PF_DISPLAY_NAME,
    PF_METHOD,
    PF_METHOD_FULL_NAME,
    ProgressiveFidelityEvaluator,
    run_pf_tahc_abc,
)
from .multifidelity import (
    ASHA_DISPLAY_NAME,
    ASHA_FULL_NAME,
    ASHA_METHOD,
    HYPERBAND_DISPLAY_NAME,
    HYPERBAND_FULL_NAME,
    HYPERBAND_METHOD,
    run_asha,
    run_hyperband,
)
from .presets import DATASET_PRESETS
from .search_space import SearchSpace


def build_base_args(
    dataset_name: str,
    pred_len: int,
    model_name: str,
    gpu: int = 0,
    overrides: Optional[Dict[str, object]] = None,
) -> SimpleNamespace:
    if dataset_name not in DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {dataset_name}")
    preset = copy.deepcopy(DATASET_PRESETS[dataset_name])

    args = {
        "is_training": 1,
        "model_id": f"{dataset_name}_{pred_len}",
        "model": model_name,
        "data": preset["data"],
        "root_path": preset["root_path"],
        "data_path": preset["data_path"],
        "features": preset["features"],
        "target": preset["target"],
        "freq": preset["freq"],
        "checkpoints": "./checkpoints/",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": int(pred_len),
        "enc_in": int(preset["enc_in"]),
        "dec_in": int(preset["dec_in"]),
        "c_out": int(preset["c_out"]),
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "do_predict": False,
        "num_workers": 10,
        "eval_batch_size": 512,
        "itr": 1,
        "train_epochs": 20,
        "batch_size": 64,
        "patience": 5,
        "learning_rate": 2e-4,
        "des": "Exp",
        "loss": "MSE",
        "lradj": "type1",
        "use_amp": False,
        "skip_epoch_val": False,
        "use_gpu": True,
        "gpu": int(gpu),
        "use_multi_gpu": False,
        "devices": str(gpu),
        "exp_name": "MTSF",
        "channel_independence": False,
        "inverse": False,
        "class_strategy": "projection",
        "target_root_path": preset["root_path"],
        "target_data_path": preset["data_path"],
        "efficient_training": False,
        "use_norm": 1,
        "partial_start_index": 0,
        "weight_decay": 1e-4,
        "csg_reduction": 4,
        "run_test_during_train": False,
    }
    if overrides:
        args.update(overrides)
    return SimpleNamespace(**args)


class ExperimentRunner:
    def __init__(
        self,
        output_dir: str,
        dataset_name: str,
        pred_len: int,
        model_name: str,
        gpu: int = 0,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.pred_len = int(pred_len)
        self.model_name = model_name
        self.base_args = build_base_args(dataset_name, pred_len, model_name, gpu=gpu, overrides=overrides)
        self.space = SearchSpace()

    def _log_runner(self, message: str) -> None:
        print(f"[Runner] {message}", flush=True)

    def _split_two_stage_budget(self, search_budget: int, colony_size: int) -> tuple[int, int]:
        search_budget = int(search_budget)
        colony_size = int(colony_size)
        if search_budget <= 1:
            return search_budget, 0

        preferred_stage1 = max(colony_size * 3, math.ceil(search_budget * 0.7))
        min_stage2 = min(colony_size, max(1, search_budget - 1))
        stage1_budget = min(search_budget - min_stage2, preferred_stage1)
        stage1_budget = max(1, stage1_budget)
        stage2_budget = search_budget - stage1_budget
        if stage2_budget <= 0:
            stage1_budget = search_budget - 1
            stage2_budget = 1
        return stage1_budget, stage2_budget

    def _center_from_stage1_history(
        self,
        rows: Sequence[Dict[str, object]],
        model_name: str,
        top_m: int,
    ) -> List[float]:
        ranked = sorted(rows, key=lambda row: float(row["score"]))
        top_rows = ranked[: max(1, min(int(top_m), len(ranked)))]
        encoded = [
            self.space.encode_config(self.space.project_config(dict(row["config"]), model_name=model_name))
            for row in top_rows
        ]
        dim = len(encoded[0])
        return [sum(vec[i] for vec in encoded) / len(encoded) for i in range(dim)]

    def _optimizer_from_method(
        self,
        method: str,
        budget: int,
        seed: int,
        colony_size: int,
        limit: int,
        elite_ratio: float,
        stage_name: str,
    ):
        method = method.lower()
        if method == "random_search":
            return RandomSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "ga":
            return GeneticSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "pso":
            return PSOSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "abc":
            return OriginalABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "egta_abc":
            return EGTAABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "w_o_two_stage":
            return EGTAABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method)
        if method == "w_o_elite":
            return EGTAABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method, use_elite=False)
        if method == "w_o_typeaware":
            return EGTAABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method, type_aware=False)
        if method == "w_o_projection":
            return EGTAABCSearchOptimizer(budget, seed, colony_size, limit, elite_ratio, stage_name, method, use_projection=False)
        raise ValueError(f"Unsupported method: {method}")

    def _run_search_for_seed(
        self,
        method: str,
        seed: int,
        seed_dir: Path,
        search_budget: int,
        colony_size: int,
        limit: int,
        elite_ratio: float,
        search_train_epochs: int,
        search_patience: int,
        use_two_stage: bool,
        stage2_radius: float,
    ) -> Dict[str, object]:
        seed_dir.mkdir(parents=True, exist_ok=True)
        self._log_runner(
            f"seed={seed} | method={method} | model={self.model_name} | search_budget={search_budget} | colony_size={colony_size}"
        )

        objective = SearchObjective(
            self.base_args,
            output_dir=str(seed_dir),
            seed=seed,
            search_train_epochs=search_train_epochs,
            search_patience=search_patience,
        )

        if use_two_stage and method.lower() in {"egta_abc", "w_o_elite", "w_o_typeaware", "w_o_projection"}:
            stage1_budget, stage2_budget = self._split_two_stage_budget(search_budget, colony_size)
            self._log_runner(
                f"seed={seed} | two-stage search | stage1_budget={stage1_budget} | stage2_budget={stage2_budget} | stage2_radius={stage2_radius}"
            )

            stage1 = self._optimizer_from_method(method, stage1_budget, seed, colony_size, limit, elite_ratio, "stage1")
            stage1_result = stage1.search(
                objective=objective,
                space=self.space,
                model_name=self.model_name,
                center=self.space.encode_config(self.space.get_default_config(self.model_name)),
                active_names=self.space.get_active_names(self.model_name, stage=1),
                radius=None,
            )

            stage2_center = self._center_from_stage1_history(
                stage1_result.history,
                model_name=self.model_name,
                top_m=max(2, colony_size + 1),
            )
            stage2 = self._optimizer_from_method(method, stage2_budget, seed + 9973, colony_size, limit, elite_ratio, "stage2")
            stage2_result = stage2.search(
                objective=objective,
                space=self.space,
                model_name=self.model_name,
                center=stage2_center,
                active_names=self.space.get_active_names(self.model_name, stage=2),
                radius=stage2_radius,
            )

            if stage2_result.best_score < stage1_result.best_score:
                best_config = stage2_result.best_config
                best_score = stage2_result.best_score
            else:
                best_config = stage1_result.best_config
                best_score = stage1_result.best_score

            search_rows = stage1_result.history + stage2_result.history
            search_total_time = stage1_result.total_time_sec + stage2_result.total_time_sec
        else:
            self._log_runner(f"seed={seed} | single-stage search")
            optimizer = self._optimizer_from_method(method, search_budget, seed, colony_size, limit, elite_ratio, "single_stage")
            result = optimizer.search(
                objective=objective,
                space=self.space,
                model_name=self.model_name,
                center=self.space.encode_config(self.space.get_default_config(self.model_name)),
                active_names=self.space.get_active_names(self.model_name, stage=None),
                radius=None,
            )
            best_config = result.best_config
            best_score = result.best_score
            search_rows = result.history
            search_total_time = result.total_time_sec

        self._log_runner(
            f"seed={seed} | search complete | best_val_mse={float(best_score):.6f} | search_time_sec={float(search_total_time):.2f}"
        )

        search_summary = {
            "seed": seed,
            "best_score": best_score,
            "search_time_sec": search_total_time,
            "best_config": best_config,
        }
        self._dump_json(seed_dir / "best_config.json", best_config)
        self._dump_json(seed_dir / "search_summary.json", search_summary)
        self._dump_history(seed_dir / "search_history.csv", search_rows)
        return {
            "seed": seed,
            "best_config": best_config,
            "best_score": best_score,
            "search_time_sec": search_total_time,
            "search_rows": search_rows,
            "seed_dir": str(seed_dir),
            "summary": search_summary,
        }

    def run_fixed_baseline(
        self,
        seeds: Sequence[int],
        final_train_epochs: int = 20,
        final_patience: int = 5,
    ) -> Dict[str, object]:
        config = self.space.get_default_config(self.model_name)
        retrainer = FinalRetrainer(
            self.base_args,
            output_dir=str(self.output_dir),
            final_train_epochs=final_train_epochs,
            final_patience=final_patience,
        )
        final_rows = [retrainer.run(config, seed=seed, tag="fixed") for seed in seeds]
        return {
            "method": "fixed",
            "best_config": config,
            "search": None,
            "final_rows": final_rows,
            "summary": self._summarize_rows(final_rows),
        }

    def run_search_method(
        self,
        method: str,
        seeds: Sequence[int],
        search_seeds: Optional[Sequence[int]] = None,
        final_seeds: Optional[Sequence[int]] = None,
        search_budget: int = 20,
        colony_size: int = 4,
        limit: int = 3,
        elite_ratio: float = 0.4,
        search_train_epochs: int = 5,
        search_patience: int = 3,
        final_train_epochs: int = 20,
        final_patience: int = 5,
        use_two_stage: bool = True,
        stage2_radius: float = 0.25,
        race_fidelities: Optional[Sequence[int]] = None,
        race_init_size: int = 48,
        race_keep: Optional[Sequence[int]] = None,
        race_challengers: Optional[Sequence[int]] = None,
        verify_top_k: int = 4,
        verify_epochs: Optional[int] = None,
        verify_seed: Optional[int] = None,
        challenge_struct_prob: float = 0.30,
        challenge_local_prob: float = 0.70,
        max_duplicate_tries: int = 50,
        pf_save_archive: bool = False,
        pf_no_resume: bool = False,
        hf_abc: bool = False,
        elite_refine_ratio: float = 0.70,
        elite_struct_ratio: float = 0.20,
        elite_random_ratio: float = 0.10,
        elite_top_m: int = 4,
        elite_lr_sigma: float = 0.20,
        elite_dropout_sigma: float = 0.02,
        elite_max_tries: int = 50,
        mf_min_epochs: int = 1,
        mf_max_epochs: Optional[int] = None,
        mf_eta: int = 3,
        mf_budget_epochs: Optional[int] = None,
    ) -> Dict[str, object]:
        default_seeds = [int(seed) for seed in seeds]
        search_seeds = [int(seed) for seed in (search_seeds or default_seeds)]
        final_seeds = [int(seed) for seed in (final_seeds or default_seeds)]
        if not search_seeds:
            raise ValueError("search_seeds must contain at least one seed")
        if not final_seeds:
            raise ValueError("final_seeds must contain at least one seed")

        if method.lower() == PF_METHOD:
            return self._run_pf_search_method(
                method=method,
                search_seeds=search_seeds,
                final_seeds=final_seeds,
                search_patience=search_patience,
                final_train_epochs=final_train_epochs,
                final_patience=final_patience,
                race_fidelities=race_fidelities,
                race_init_size=race_init_size,
                race_keep=race_keep,
                race_challengers=race_challengers,
                verify_top_k=verify_top_k,
                verify_epochs=verify_epochs,
                verify_seed=verify_seed,
                challenge_struct_prob=challenge_struct_prob,
                challenge_local_prob=challenge_local_prob,
                hf_abc=hf_abc,
                elite_refine_ratio=elite_refine_ratio,
                elite_struct_ratio=elite_struct_ratio,
                elite_random_ratio=elite_random_ratio,
                elite_top_m=elite_top_m,
                elite_lr_sigma=elite_lr_sigma,
                elite_dropout_sigma=elite_dropout_sigma,
                elite_max_tries=elite_max_tries,
                max_duplicate_tries=max_duplicate_tries,
                pf_save_archive=pf_save_archive,
                pf_no_resume=pf_no_resume,
            )
        if method.lower() in {HYPERBAND_METHOD, ASHA_METHOD}:
            return self._run_multifidelity_search_method(
                method=method,
                search_seeds=search_seeds,
                final_seeds=final_seeds,
                search_budget=search_budget,
                search_train_epochs=search_train_epochs,
                search_patience=search_patience,
                final_train_epochs=final_train_epochs,
                final_patience=final_patience,
                verify_top_k=verify_top_k,
                verify_epochs=verify_epochs,
                verify_seed=verify_seed,
                max_duplicate_tries=max_duplicate_tries,
                mf_min_epochs=mf_min_epochs,
                mf_max_epochs=mf_max_epochs,
                mf_eta=mf_eta,
                mf_budget_epochs=mf_budget_epochs,
            )

        all_search_rows = []
        final_rows = []

        split_search_and_final = search_seeds != final_seeds

        if split_search_and_final:
            selected_search = None
            for seed in search_seeds:
                search_run = self._run_search_for_seed(
                    method=method,
                    seed=seed,
                    seed_dir=self.output_dir / f"search_seed_{seed}",
                    search_budget=search_budget,
                    colony_size=colony_size,
                    limit=limit,
                    elite_ratio=elite_ratio,
                    search_train_epochs=search_train_epochs,
                    search_patience=search_patience,
                    use_two_stage=use_two_stage,
                    stage2_radius=stage2_radius,
                )
                all_search_rows.append(search_run["summary"])
                if selected_search is None or float(search_run["best_score"]) < float(selected_search["best_score"]):
                    selected_search = search_run

            assert selected_search is not None
            selected_config = dict(selected_search["best_config"])
            selected_seed = int(selected_search["seed"])
            self._log_runner(
                f"selected config from search_seed={selected_seed} | best_val_mse={float(selected_search['best_score']):.6f}"
            )
            self._dump_json(
                self.output_dir / "selected_best_config.json",
                {
                    "source_search_seed": selected_seed,
                    "best_score": float(selected_search["best_score"]),
                    "best_config": selected_config,
                },
            )

            for seed in final_seeds:
                seed_dir = self.output_dir / f"final_seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                retrainer = FinalRetrainer(
                    self.base_args,
                    output_dir=str(seed_dir),
                    final_train_epochs=final_train_epochs,
                    final_patience=final_patience,
                )
                self._log_runner(f"final_seed={seed} | using config from search_seed={selected_seed} | final retrain start")
                final_rows.append(
                    retrainer.run(selected_config, seed=seed, tag=f"{method.lower()}_search{selected_seed}")
                )
                self._log_runner(f"final_seed={seed} | final retrain complete")
                self._dump_json(seed_dir / "best_config.json", selected_config)
        else:
            for seed in default_seeds:
                seed_dir = self.output_dir / f"seed_{seed}"
                search_run = self._run_search_for_seed(
                    method=method,
                    seed=seed,
                    seed_dir=seed_dir,
                    search_budget=search_budget,
                    colony_size=colony_size,
                    limit=limit,
                    elite_ratio=elite_ratio,
                    search_train_epochs=search_train_epochs,
                    search_patience=search_patience,
                    use_two_stage=use_two_stage,
                    stage2_radius=stage2_radius,
                )
                all_search_rows.append(search_run["summary"])

                retrainer = FinalRetrainer(
                    self.base_args,
                    output_dir=str(seed_dir),
                    final_train_epochs=final_train_epochs,
                    final_patience=final_patience,
                )
                self._log_runner(f"seed={seed} | final retrain start")
                final_rows.append(retrainer.run(search_run["best_config"], seed=seed, tag=method.lower()))
                self._log_runner(f"seed={seed} | final retrain complete")

        summary = {
            "method": method,
            "search_seeds": search_seeds,
            "final_seeds": final_seeds,
            "search_summary": self._summarize_search_rows(all_search_rows),
            "final_summary": self._summarize_rows(final_rows),
        }
        if split_search_and_final:
            summary["selected_search_seed"] = int(selected_seed)
            summary["selected_best_config"] = selected_config
        self._dump_json(self.output_dir / f"{method.lower()}_summary.json", summary)
        return {
            "method": method,
            "search_rows": all_search_rows,
            "final_rows": final_rows,
            "summary": summary,
        }

    def _run_pf_search_method(
        self,
        method: str,
        search_seeds: Sequence[int],
        final_seeds: Sequence[int],
        search_patience: int,
        final_train_epochs: int,
        final_patience: int,
        race_fidelities: Optional[Sequence[int]] = None,
        race_init_size: int = 48,
        race_keep: Optional[Sequence[int]] = None,
        race_challengers: Optional[Sequence[int]] = None,
        verify_top_k: int = 4,
        verify_epochs: Optional[int] = None,
        verify_seed: Optional[int] = None,
        challenge_struct_prob: float = 0.30,
        challenge_local_prob: float = 0.70,
        max_duplicate_tries: int = 50,
        pf_save_archive: bool = False,
        pf_no_resume: bool = False,
        hf_abc: bool = False,
        elite_refine_ratio: float = 0.70,
        elite_struct_ratio: float = 0.20,
        elite_random_ratio: float = 0.10,
        elite_top_m: int = 4,
        elite_lr_sigma: float = 0.20,
        elite_dropout_sigma: float = 0.02,
        elite_max_tries: int = 50,
    ) -> Dict[str, object]:
        pf_args = SimpleNamespace(
            output_dir=str(self.output_dir),
            model_name=self.model_name,
            search_seeds=[int(seed) for seed in search_seeds],
            final_seeds=[int(seed) for seed in final_seeds],
            search_patience=int(search_patience),
            race_fidelities=[int(item) for item in (race_fidelities or [1, 3, 6, 10])],
            race_init_size=int(race_init_size),
            race_keep=[int(item) for item in (race_keep or [16, 8, 4, 4])],
            race_challengers=[int(item) for item in (race_challengers or [0, 8, 4, 0])],
            verify_top_k=int(verify_top_k),
            verify_epochs=int(verify_epochs if verify_epochs is not None else final_train_epochs),
            verify_seed=None if verify_seed is None else int(verify_seed),
            challenge_struct_prob=float(challenge_struct_prob),
            challenge_local_prob=float(challenge_local_prob),
            hf_abc=bool(hf_abc),
            elite_refine_ratio=float(elite_refine_ratio),
            elite_struct_ratio=float(elite_struct_ratio),
            elite_random_ratio=float(elite_random_ratio),
            elite_top_m=int(elite_top_m),
            elite_lr_sigma=float(elite_lr_sigma),
            elite_dropout_sigma=float(elite_dropout_sigma),
            elite_max_tries=int(elite_max_tries),
            max_duplicate_tries=int(max_duplicate_tries),
            pf_save_archive=bool(pf_save_archive),
            pf_no_resume=bool(pf_no_resume),
        )

        evaluator = ProgressiveFidelityEvaluator(
            self.base_args,
            output_dir=str(self.output_dir),
        )

        self._log_runner(
            f"method={method} | {'HF-ABC' if pf_args.hf_abc else 'PF-ABC'} progressive search | "
            f"search_seeds={pf_args.search_seeds} | final_seeds={pf_args.final_seeds}"
        )
        selected_best_config, search_summary = run_pf_tahc_abc(pf_args, self.space, evaluator)
        selected_search_seed = int(search_summary["selected_search_seed"])

        self._dump_json(
            self.output_dir / "selected_best_config.json",
            {
                "source_search_seed": selected_search_seed,
                "best_score": float(search_summary["selected_val_mse"]),
                "best_config": selected_best_config,
            },
        )

        final_rows = []
        for seed in final_seeds:
            seed = int(seed)
            seed_dir = self.output_dir / f"final_seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            retrainer = FinalRetrainer(
                self.base_args,
                output_dir=str(seed_dir),
                final_train_epochs=final_train_epochs,
                final_patience=final_patience,
            )
            self._log_runner(
                f"final_seed={seed} | using PF-ABC config from search_seed={selected_search_seed} | final retrain start"
            )
            final_rows.append(
                retrainer.run(selected_best_config, seed=seed, tag=f"{method.lower()}_search{selected_search_seed}")
            )
            self._log_runner(f"final_seed={seed} | final retrain complete")
            self._dump_json(seed_dir / "best_config.json", selected_best_config)

        summary = {
            "method": PF_METHOD,
            "display_name": "HF-ABC" if pf_args.hf_abc else PF_DISPLAY_NAME,
            "method_full_name": (
                "Hybrid Fidelity-guided Artificial Bee Colony" if pf_args.hf_abc else PF_METHOD_FULL_NAME
            ),
            "search_seeds": [int(seed) for seed in search_seeds],
            "final_seeds": [int(seed) for seed in final_seeds],
            "search_summary": search_summary,
            "final_summary": self._summarize_rows(final_rows),
        }
        self._dump_json(self.output_dir / f"{method.lower()}_summary.json", summary)
        return {
            "method": PF_METHOD,
            "search_rows": list(search_summary.get("per_search_seed", [])),
            "final_rows": final_rows,
            "summary": summary,
        }

    def _run_multifidelity_search_method(
        self,
        method: str,
        search_seeds: Sequence[int],
        final_seeds: Sequence[int],
        search_budget: int,
        search_train_epochs: int,
        search_patience: int,
        final_train_epochs: int,
        final_patience: int,
        verify_top_k: int = 4,
        verify_epochs: Optional[int] = None,
        verify_seed: Optional[int] = None,
        max_duplicate_tries: int = 50,
        mf_min_epochs: int = 1,
        mf_max_epochs: Optional[int] = None,
        mf_eta: int = 3,
        mf_budget_epochs: Optional[int] = None,
    ) -> Dict[str, object]:
        method = method.lower()
        total_budget_epochs = int(
            mf_budget_epochs
            if mf_budget_epochs is not None
            else int(search_budget) * int(search_train_epochs)
        )
        max_epochs = int(mf_max_epochs if mf_max_epochs is not None else (verify_epochs or final_train_epochs))
        mf_args = SimpleNamespace(
            output_dir=str(self.output_dir),
            model_name=self.model_name,
            search_seeds=[int(seed) for seed in search_seeds],
            final_seeds=[int(seed) for seed in final_seeds],
            search_patience=int(search_patience),
            verify_top_k=int(verify_top_k),
            verify_epochs=int(verify_epochs if verify_epochs is not None else max_epochs),
            verify_seed=None if verify_seed is None else int(verify_seed),
            max_duplicate_tries=int(max_duplicate_tries),
            mf_min_epochs=int(mf_min_epochs),
            mf_max_epochs=max_epochs,
            mf_eta=int(mf_eta),
            mf_budget_epochs=total_budget_epochs,
        )
        evaluator = ProgressiveFidelityEvaluator(
            self.base_args,
            output_dir=str(self.output_dir),
        )

        display_name = HYPERBAND_DISPLAY_NAME if method == HYPERBAND_METHOD else ASHA_DISPLAY_NAME
        self._log_runner(
            f"method={method} | {display_name} multi-fidelity search | "
            f"search_seeds={mf_args.search_seeds} | budget_epochs={total_budget_epochs}"
        )
        if method == HYPERBAND_METHOD:
            selected_best_config, search_summary = run_hyperband(mf_args, self.space, evaluator)
            method_full_name = HYPERBAND_FULL_NAME
        elif method == ASHA_METHOD:
            selected_best_config, search_summary = run_asha(mf_args, self.space, evaluator)
            method_full_name = ASHA_FULL_NAME
        else:
            raise ValueError(f"Unsupported multi-fidelity method: {method}")

        selected_search_seed = int(search_summary["selected_search_seed"])
        self._dump_json(
            self.output_dir / "selected_best_config.json",
            {
                "source_search_seed": selected_search_seed,
                "best_score": float(search_summary["selected_val_mse"]),
                "best_config": selected_best_config,
            },
        )

        final_rows = []
        for seed in final_seeds:
            seed = int(seed)
            seed_dir = self.output_dir / f"final_seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            retrainer = FinalRetrainer(
                self.base_args,
                output_dir=str(seed_dir),
                final_train_epochs=final_train_epochs,
                final_patience=final_patience,
            )
            self._log_runner(
                f"final_seed={seed} | using {method} config from search_seed={selected_search_seed} | final retrain start"
            )
            final_rows.append(
                retrainer.run(selected_best_config, seed=seed, tag=f"{method}_search{selected_search_seed}")
            )
            self._log_runner(f"final_seed={seed} | final retrain complete")
            self._dump_json(seed_dir / "best_config.json", selected_best_config)

        summary = {
            "method": method,
            "display_name": display_name,
            "method_full_name": method_full_name,
            "search_seeds": [int(seed) for seed in search_seeds],
            "final_seeds": [int(seed) for seed in final_seeds],
            "search_summary": search_summary,
            "final_summary": self._summarize_rows(final_rows),
        }
        self._dump_json(self.output_dir / f"{method}_summary.json", summary)
        return {
            "method": method,
            "search_rows": list(search_summary.get("per_search_seed", [])),
            "final_rows": final_rows,
            "summary": summary,
        }

    def _dump_json(self, path: Path, obj) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _dump_history(self, path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _summarize_rows(self, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        mses = [float(r["mse"]) for r in rows]
        maes = [float(r["mae"]) for r in rows]
        summary = {
            "mse_mean": mean(mses),
            "mse_std": pstdev(mses) if len(mses) > 1 else 0.0,
            "mae_mean": mean(maes),
            "mae_std": pstdev(maes) if len(maes) > 1 else 0.0,
            "n_seeds": len(rows),
        }
        for key in ("key_mse", "key_mae", "key_rmse"):
            vals = [float(r[key]) for r in rows if key in r]
            if vals:
                summary[f"{key}_mean"] = mean(vals)
                summary[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        return summary

    def _summarize_search_rows(self, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        scores = [float(r["best_score"]) for r in rows]
        times = [float(r["search_time_sec"]) for r in rows]
        return {
            "best_val_mse_mean": mean(scores),
            "best_val_mse_std": pstdev(scores) if len(scores) > 1 else 0.0,
            "search_time_mean": mean(times),
            "search_time_std": pstdev(times) if len(times) > 1 else 0.0,
            "n_seeds": len(rows),
        }
