from __future__ import annotations

import csv
import importlib
import json
import math
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Optional, Sequence

from .objective import FinalRetrainer, SearchObjective
from .search_space import SearchSpace


OPTUNA_TPE_METHOD = "optuna_tpe"
OPTUNA_TPE_DISPLAY_NAME = "Optuna-TPE"
OPTUNA_TPE_FULL_NAME = "Optuna Tree-structured Parzen Estimator with a fixed trial budget"


def _require_optuna():
    try:
        return importlib.import_module("optuna")
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for optuna_tpe experiments. Install it with: "
            "pip install optuna"
        ) from exc


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    return obj


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_for_json(obj), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _finite_score(value: float) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    return 1.0e30


def suggest_config_from_trial(trial, search_space: SearchSpace, model_name: str) -> Dict[str, object]:
    """Map one Optuna trial to the existing SearchSpace contract."""

    config = search_space.get_default_config(model_name)
    for spec in search_space.specs:
        if not search_space.is_searchable(spec.name, model_name):
            continue

        if spec.kind in {"discrete", "conditional_discrete"}:
            config[spec.name] = trial.suggest_categorical(spec.name, list(spec.values or []))
        elif spec.kind == "continuous":
            config[spec.name] = trial.suggest_float(spec.name, float(spec.low), float(spec.high))
        elif spec.kind == "log_continuous":
            config[spec.name] = trial.suggest_float(
                spec.name,
                float(spec.low),
                float(spec.high),
                log=True,
            )
        else:
            raise ValueError(f"Unsupported search-space kind: {spec.kind}")

    return search_space.project_config(config, model_name=model_name)


def _trial_to_record(trial) -> Dict[str, object]:
    user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
    return {
        "trial_number": int(trial.number),
        "state": str(trial.state.name),
        "score": None if trial.value is None else float(trial.value),
        "val_mse": user_attrs.get("val_mse"),
        "cached": int(bool(user_attrs.get("cached", False))),
        "train_time_sec": user_attrs.get("train_time_sec"),
        "params": dict(trial.params),
        "config": dict(user_attrs.get("config") or {}),
        "metrics": dict(user_attrs.get("metrics") or {}),
        "duration_sec": user_attrs.get("duration_sec"),
    }


def _write_trials_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "trial_number",
        "state",
        "score",
        "val_mse",
        "cached",
        "train_time_sec",
        "duration_sec",
        "config",
        "params",
        "metrics",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("config", "params", "metrics"):
                flat[key] = json.dumps(_clean_for_json(flat.get(key, {})), ensure_ascii=False, sort_keys=True)
            writer.writerow(flat)


def run_optuna_tpe_for_seed(
    *,
    base_args,
    search_space: SearchSpace,
    model_name: str,
    output_dir: str | Path,
    search_seed: int,
    n_trials: int = 20,
    search_train_epochs: int = 5,
    search_patience: int = 3,
    tpe_startup_trials: int = 10,
    tpe_multivariate: bool = False,
    tpe_group: bool = False,
    tpe_constant_liar: bool = False,
) -> Dict[str, object]:
    optuna = _require_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_seed = int(search_seed)
    n_trials = int(n_trials)

    objective = SearchObjective(
        base_args,
        output_dir=str(output_dir),
        seed=search_seed,
        search_train_epochs=search_train_epochs,
        search_patience=search_patience,
    )

    effective_multivariate = bool(tpe_multivariate or tpe_group)
    sampler_kwargs = {
        "seed": search_seed,
        "n_startup_trials": int(tpe_startup_trials),
        "multivariate": effective_multivariate,
        "group": bool(tpe_group),
        "constant_liar": bool(tpe_constant_liar),
    }
    sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"{OPTUNA_TPE_METHOD}_seed_{search_seed}",
    )

    def objective_fn(trial) -> float:
        started = time.time()
        config = suggest_config_from_trial(trial, search_space, model_name=model_name)
        trial.set_user_attr("config", config)
        result = objective.evaluate(config, phase=f"optuna_tpe_trial_{trial.number + 1}")

        raw_score = float(result.score)
        score = _finite_score(raw_score)
        metrics = dict(result.metrics)
        trial.set_user_attr("raw_score", raw_score)
        trial.set_user_attr("val_mse", raw_score)
        trial.set_user_attr("score_for_optuna", score)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("cached", bool(result.cached))
        trial.set_user_attr("train_time_sec", metrics.get("train_time_sec"))
        trial.set_user_attr("duration_sec", time.time() - started)

        print(
            f"[Optuna-TPE] seed={search_seed} trial={trial.number + 1}/{n_trials} "
            f"val_mse={raw_score:.6f} cached={int(result.cached)} config={config}",
            flush=True,
        )
        return score

    started = time.time()
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    search_time_sec = time.time() - started

    complete_state = optuna.trial.TrialState.COMPLETE
    completed = [trial for trial in study.trials if trial.state == complete_state]
    if not completed:
        raise RuntimeError("Optuna-TPE completed no successful trials.")

    best_trial = min(completed, key=lambda trial: float(trial.value))
    best_config = dict(best_trial.user_attrs.get("config") or {})
    selected_score = float(best_trial.value)
    selected_val_mse = float(best_trial.user_attrs.get("val_mse", selected_score))
    trial_rows = [_trial_to_record(trial) for trial in study.trials]

    summary = {
        "method": OPTUNA_TPE_METHOD,
        "display_name": OPTUNA_TPE_DISPLAY_NAME,
        "method_full_name": OPTUNA_TPE_FULL_NAME,
        "search_seed": search_seed,
        "fixed_search_budget_trials": n_trials,
        "n_trials": n_trials,
        "search_train_epochs": int(search_train_epochs),
        "search_patience": int(search_patience),
        "tpe_startup_trials": int(tpe_startup_trials),
        "tpe_multivariate": effective_multivariate,
        "tpe_group": bool(tpe_group),
        "tpe_constant_liar": bool(tpe_constant_liar),
        "selected_config": best_config,
        "selected_score": selected_score,
        "selected_val_mse": selected_val_mse,
        "best_trial_number": int(best_trial.number),
        "search_time_sec": float(search_time_sec),
        "trials": trial_rows,
        "archive_size": len(trial_rows),
    }
    _dump_json(output_dir / "optuna_tpe_search_summary.json", summary)
    _dump_json(output_dir / "best_config.json", best_config)
    _write_trials_csv(output_dir / "optuna_tpe_trials.csv", trial_rows)
    return summary


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
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


def run_optuna_tpe_experiment(
    *,
    base_args,
    output_dir: str | Path,
    model_name: str,
    search_space: Optional[SearchSpace] = None,
    search_seeds: Sequence[int] = (2026,),
    final_seeds: Sequence[int] = (2026, 2027, 2028),
    n_trials: int = 20,
    search_train_epochs: int = 5,
    search_patience: int = 3,
    final_train_epochs: int = 20,
    final_patience: int = 5,
    tpe_startup_trials: int = 10,
    tpe_multivariate: bool = False,
    tpe_group: bool = False,
    tpe_constant_liar: bool = False,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = search_space or SearchSpace()
    search_seeds = [int(seed) for seed in search_seeds]
    final_seeds = [int(seed) for seed in final_seeds]
    if not search_seeds:
        raise ValueError("search_seeds must contain at least one seed")
    if not final_seeds:
        raise ValueError("final_seeds must contain at least one seed")

    started = time.time()
    per_search_seed = []
    for seed in search_seeds:
        seed_dir = output_dir / f"search_seed_{seed}"
        print(
            f"[Optuna-TPE] search_seed={seed} | fixed_budget_trials={int(n_trials)} | search start",
            flush=True,
        )
        per_seed = run_optuna_tpe_for_seed(
            base_args=base_args,
            search_space=search_space,
            model_name=model_name,
            output_dir=seed_dir,
            search_seed=seed,
            n_trials=n_trials,
            search_train_epochs=search_train_epochs,
            search_patience=search_patience,
            tpe_startup_trials=tpe_startup_trials,
            tpe_multivariate=tpe_multivariate,
            tpe_group=tpe_group,
            tpe_constant_liar=tpe_constant_liar,
        )
        per_search_seed.append(per_seed)

    per_search_seed = sorted(per_search_seed, key=lambda item: float(item["selected_score"]))
    selected = per_search_seed[0]
    selected_search_seed = int(selected["search_seed"])
    selected_best_config = dict(selected["selected_config"])
    selected_val_mse = float(selected["selected_val_mse"])

    _dump_json(
        output_dir / "selected_best_config.json",
        {
            "source_search_seed": selected_search_seed,
            "best_score": selected_val_mse,
            "best_config": selected_best_config,
        },
    )

    final_rows = []
    for seed in final_seeds:
        seed_dir = output_dir / f"final_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        retrainer = FinalRetrainer(
            base_args,
            output_dir=str(seed_dir),
            final_train_epochs=final_train_epochs,
            final_patience=final_patience,
        )
        print(
            f"[Optuna-TPE] final_seed={seed} | using config from search_seed={selected_search_seed} | start",
            flush=True,
        )
        row = retrainer.run(
            selected_best_config,
            seed=int(seed),
            tag=f"{OPTUNA_TPE_METHOD}_search{selected_search_seed}",
        )
        print(f"[Optuna-TPE] final_seed={seed} | complete", flush=True)
        _dump_json(seed_dir / "best_config.json", selected_best_config)
        _dump_json(seed_dir / "final_metrics.json", row)
        final_rows.append(row)

    search_summary = {
        "method": OPTUNA_TPE_METHOD,
        "display_name": OPTUNA_TPE_DISPLAY_NAME,
        "method_full_name": OPTUNA_TPE_FULL_NAME,
        "fixed_search_budget_trials": int(n_trials),
        "n_trials": int(n_trials),
        "search_train_epochs": int(search_train_epochs),
        "search_patience": int(search_patience),
        "tpe_startup_trials": int(tpe_startup_trials),
        "tpe_multivariate": bool(tpe_multivariate or tpe_group),
        "tpe_group": bool(tpe_group),
        "tpe_constant_liar": bool(tpe_constant_liar),
        "selected_search_seed": selected_search_seed,
        "selected_best_config": selected_best_config,
        "selected_val_mse": selected_val_mse,
        "search_time_sec": float(time.time() - started),
        "per_search_seed": per_search_seed,
        "archive_size": int(sum(int(item.get("archive_size", 0)) for item in per_search_seed)),
    }
    summary = {
        "method": OPTUNA_TPE_METHOD,
        "display_name": OPTUNA_TPE_DISPLAY_NAME,
        "method_full_name": OPTUNA_TPE_FULL_NAME,
        "search_seeds": search_seeds,
        "final_seeds": final_seeds,
        "search_summary": search_summary,
        "final_summary": _summarize_rows(final_rows),
    }
    _dump_json(output_dir / f"{OPTUNA_TPE_METHOD}_summary.json", summary)
    return {
        "method": OPTUNA_TPE_METHOD,
        "search_rows": per_search_seed,
        "final_rows": final_rows,
        "summary": summary,
    }
