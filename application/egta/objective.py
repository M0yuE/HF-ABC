from __future__ import annotations

import copy
import csv
import json
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .exp_ext import Exp_Long_Term_Forecast_Ext
from .key_metrics import compute_key_metrics


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@dataclass
class EvalResult:
    score: float
    metrics: Dict[str, float]
    config: Dict[str, object]
    cached: bool


class SearchObjective:
    def __init__(
        self,
        base_args: SimpleNamespace,
        output_dir: str,
        seed: int,
        search_train_epochs: int = 5,
        search_patience: int = 3,
        cleanup_checkpoints: bool = True,
    ) -> None:
        self.base_args = copy.deepcopy(base_args)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.search_train_epochs = int(search_train_epochs)
        self.search_patience = int(search_patience)
        self.cleanup_checkpoints = cleanup_checkpoints
        self.cache: Dict[str, EvalResult] = {}
        self.records: List[Dict[str, object]] = []
        self.counter = 0
        self.log_path = self.output_dir / "search_trials.csv"

    def _config_key(self, config: Dict[str, object]) -> str:
        return json.dumps(dict(sorted(config.items())), ensure_ascii=False, sort_keys=True)

    def _write_record(self, row: Dict[str, object]) -> None:
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _make_args(self, config: Dict[str, object], candidate_tag: str) -> SimpleNamespace:
        args = copy.deepcopy(self.base_args)
        for k, v in config.items():
            setattr(args, k, v)
        args.train_epochs = self.search_train_epochs
        args.patience = self.search_patience
        args.is_training = 1
        return args

    def _short_phase_tag(self, phase: str) -> str:
        parts = [part[:2] for part in str(phase).split("_") if part]
        short = "".join(parts)[:10]
        return short or "srch"

    def evaluate(self, config: Dict[str, object], phase: str = "search") -> EvalResult:
        key = self._config_key(config)
        if key in self.cache:
            cached = self.cache[key]
            return EvalResult(
                score=cached.score,
                metrics=dict(cached.metrics),
                config=dict(cached.config),
                cached=True,
            )

        self.counter += 1
        candidate_tag = f"{self._short_phase_tag(phase)}_s{self.seed}_c{self.counter}_{uuid.uuid4().hex[:6]}"
        args = self._make_args(config, candidate_tag)
        setting = None
        param_count = 0
        train_time = 0.0

        try:
            set_global_seed(self.seed)
            exp = Exp_Long_Term_Forecast_Ext(args)
            setting = exp.make_setting(extra_tag=candidate_tag)
            param_count = count_parameters(exp.model)

            start = time.time()
            exp.train(setting)
            train_time = time.time() - start

            vali_data, vali_loader = exp._get_data(flag="val")
            criterion = exp._select_criterion()
            score = float(exp.vali(vali_data, vali_loader, criterion))

            metrics = {
                "val_mse": score,
                "train_time_sec": train_time,
                "params": param_count,
                "oom": 0,
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            score = float("inf")
            metrics = {
                "val_mse": score,
                "train_time_sec": train_time,
                "params": param_count,
                "oom": 1,
            }
            print(
                f"[SearchObjective] CUDA OOM for config={config}; assigning infinite score and continuing.",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if self.cleanup_checkpoints and setting:
                ckpt_dir = Path(args.checkpoints) / setting
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir, ignore_errors=True)

        row = {
            "candidate_id": self.counter,
            "phase": phase,
            "score": score,
            "train_time_sec": train_time,
            "params": param_count,
            **metrics,
            **config,
        }
        self.records.append(row)
        self._write_record(row)

        result = EvalResult(
            score=score,
            metrics=metrics,
            config=dict(config),
            cached=False,
        )
        self.cache[key] = result
        return result


class FinalRetrainer:
    def __init__(
        self,
        base_args: SimpleNamespace,
        output_dir: str,
        final_train_epochs: int = 20,
        final_patience: int = 5,
    ) -> None:
        self.base_args = copy.deepcopy(base_args)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_train_epochs = int(final_train_epochs)
        self.final_patience = int(final_patience)

    def _make_args(self, config: Dict[str, object], seed: int, tag: str) -> SimpleNamespace:
        args = copy.deepcopy(self.base_args)
        for k, v in config.items():
            setattr(args, k, v)
        args.train_epochs = self.final_train_epochs
        args.patience = self.final_patience
        args.is_training = 1
        return args

    def run(self, config: Dict[str, object], seed: int, tag: str = "final") -> Dict[str, float]:
        set_global_seed(seed)
        args = self._make_args(config, seed, tag)
        exp = Exp_Long_Term_Forecast_Ext(args)
        setting = exp.make_setting(extra_tag=f"{tag}_seed{seed}")
        start = time.time()
        exp.train(setting)
        train_time = time.time() - start

        exp.test(setting)
        result_dir = Path("./results") / setting
        metrics_path = result_dir / "metrics.npy"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        mae, mse, rmse, mape, mspe = np.load(metrics_path, allow_pickle=True).tolist()
        row = {
            "seed": seed,
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "train_time_sec": float(train_time),
            "setting": setting,
        }

        feature_columns = list(getattr(args, "feature_columns", []) or [])
        key_channels = list(getattr(args, "key_channels", []) or [])
        if feature_columns and key_channels:
            indices = [
                feature_columns.index(name)
                for name in key_channels
                if name in feature_columns
            ]
            if indices:
                key_metrics_path = result_dir / "key_metrics.json"
                if key_metrics_path.exists():
                    key_metrics = json.loads(key_metrics_path.read_text(encoding="utf-8"))
                    row.update(
                        {
                            "key_mse": float(key_metrics["key_mse"]),
                            "key_mae": float(key_metrics["key_mae"]),
                            "key_rmse": float(key_metrics["key_rmse"]),
                        }
                    )
                elif (result_dir / "pred.npy").exists() and (result_dir / "true.npy").exists():
                    pred = np.load(result_dir / "pred.npy")
                    true = np.load(result_dir / "true.npy")
                    row.update(compute_key_metrics(pred, true, indices))

        return row
