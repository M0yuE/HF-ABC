from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    default: object
    values: Optional[Sequence[object]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    conditional_on_model: Optional[str] = None

    def validate(self) -> None:
        if self.kind in {"discrete", "conditional_discrete"} and not self.values:
            raise ValueError(f"{self.name}: discrete parameter requires values")
        if self.kind in {"continuous", "log_continuous"}:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: continuous parameter requires low/high")


class SearchSpace:
    def __init__(self) -> None:
        self.specs: List[ParamSpec] = [
            ParamSpec("seq_len", "discrete", 96, values=[48, 96, 192, 336, 512]),
            ParamSpec("d_model", "discrete", 512, values=[128, 256, 384, 512, 768, 1024]),
            ParamSpec("n_heads", "discrete", 8, values=[2, 4, 8, 16,32]), #需与 n_heads 满足整除关系
            ParamSpec("e_layers", "discrete", 2, values=[1, 2, 3, 4, 6, 8]),
            ParamSpec("d_ff", "discrete", 2048, values=[128,256,512, 1024, 2048, 4096]),
            ParamSpec("dropout", "continuous", 0.10, low=0.05, high=0.20),
            ParamSpec("learning_rate", "log_continuous", 2e-4, low=8e-5, high=6e-4),
            ParamSpec("batch_size", "discrete", 256, values=[64, 128, 256, 512]),
            ParamSpec(
                "csg_reduction",
                "conditional_discrete",
                4,
                values=[4, 8],
                conditional_on_model="iTransformer_CSG",
            ),
        ]
        for spec in self.specs:
            spec.validate()
        self.name_to_spec = {spec.name: spec for spec in self.specs}
        self.stage1_names = ["seq_len", "d_model", "n_heads", "e_layers", "learning_rate"]
        self.stage2_names = ["dropout", "batch_size", "d_ff", "csg_reduction"]

    @property
    def names(self) -> List[str]:
        return [spec.name for spec in self.specs]

    def get_default_config(self, model_name: str) -> Dict[str, object]:
        cfg = {spec.name: spec.default for spec in self.specs}
        if model_name != "iTransformer_CSG":
            cfg["csg_reduction"] = self.name_to_spec["csg_reduction"].default
        return self.project_config(cfg, model_name=model_name)

    def is_searchable(self, name: str, model_name: str) -> bool:
        spec = self.name_to_spec[name]
        if spec.kind in {"discrete", "conditional_discrete"}:
            values = list(spec.values or [])
            if len(values) <= 1:
                return False
            if spec.conditional_on_model and model_name != spec.conditional_on_model:
                return False
            return True
        if spec.kind in {"continuous", "log_continuous"}:
            if spec.low is None or spec.high is None:
                return False
            return float(spec.low) < float(spec.high)
        return False

    def get_active_names(self, model_name: str, stage: Optional[int] = None) -> List[str]:
        if stage is None:
            names = list(self.names)
        elif stage == 1:
            names = list(self.stage1_names)
        elif stage == 2:
            names = [n for n in self.stage2_names]
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        return [name for name in names if self.is_searchable(name, model_name)]

    def is_discrete(self, name: str) -> bool:
        return self.name_to_spec[name].kind in {"discrete", "conditional_discrete"}

    def encode_config(self, config: Dict[str, object]) -> List[float]:
        vector: List[float] = []
        for spec in self.specs:
            value = config.get(spec.name, spec.default)
            if spec.kind in {"discrete", "conditional_discrete"}:
                values = list(spec.values or [])
                idx = values.index(value)
                if len(values) == 1:
                    vector.append(0.0)
                else:
                    vector.append(idx / (len(values) - 1))
            elif spec.kind == "continuous":
                vector.append((float(value) - float(spec.low)) / (float(spec.high) - float(spec.low)))
            elif spec.kind == "log_continuous":
                log_low = math.log(float(spec.low))
                log_high = math.log(float(spec.high))
                vector.append((math.log(float(value)) - log_low) / (log_high - log_low))
            else:
                raise ValueError(spec.kind)
        return [min(1.0, max(0.0, x)) for x in vector]

    def decode_vector(self, vector: Sequence[float], model_name: str) -> Dict[str, object]:
        cfg: Dict[str, object] = {}
        for idx, spec in enumerate(self.specs):
            u = min(1.0, max(0.0, float(vector[idx])))
            if spec.kind in {"discrete", "conditional_discrete"}:
                values = list(spec.values or [])
                if len(values) == 1:
                    cfg[spec.name] = values[0]
                else:
                    value_idx = int(round(u * (len(values) - 1)))
                    value_idx = max(0, min(len(values) - 1, value_idx))
                    cfg[spec.name] = values[value_idx]
            elif spec.kind == "continuous":
                cfg[spec.name] = float(spec.low) + u * (float(spec.high) - float(spec.low))
            elif spec.kind == "log_continuous":
                log_low = math.log(float(spec.low))
                log_high = math.log(float(spec.high))
                cfg[spec.name] = math.exp(log_low + u * (log_high - log_low))
            else:
                raise ValueError(spec.kind)
        return self.project_config(cfg, model_name=model_name)

    def project_config(self, config: Dict[str, object], model_name: str) -> Dict[str, object]:
        cfg = dict(config)
        for spec in self.specs:
            if spec.kind in {"discrete", "conditional_discrete"}:
                values = list(spec.values or [])
                if cfg.get(spec.name) not in values:
                    try:
                        numeric_value = float(cfg.get(spec.name))
                        nearest = min(values, key=lambda v: abs(float(v) - numeric_value))
                    except Exception:
                        nearest = spec.default
                    cfg[spec.name] = nearest
            elif spec.kind == "continuous":
                cfg[spec.name] = min(float(spec.high), max(float(spec.low), float(cfg.get(spec.name, spec.default))))
            elif spec.kind == "log_continuous":
                cfg[spec.name] = min(float(spec.high), max(float(spec.low), float(cfg.get(spec.name, spec.default))))

        # hard constraint: d_model % n_heads == 0
        d_model_values = list(self.name_to_spec["d_model"].values or [])
        n_heads = int(cfg["n_heads"])
        d_model = int(cfg["d_model"])
        if d_model % n_heads != 0:
            legal = [v for v in d_model_values if int(v) % n_heads == 0]
            if legal:
                cfg["d_model"] = min(legal, key=lambda v: abs(int(v) - d_model))
            else:
                cfg["d_model"] = min(d_model_values)

        if model_name != "iTransformer_CSG":
            cfg["csg_reduction"] = self.name_to_spec["csg_reduction"].default

        cfg["seq_len"] = int(cfg["seq_len"])
        cfg["d_model"] = int(cfg["d_model"])
        cfg["n_heads"] = int(cfg["n_heads"])
        cfg["e_layers"] = int(cfg["e_layers"])
        cfg["d_ff"] = int(cfg["d_ff"])
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["csg_reduction"] = int(cfg["csg_reduction"])
        cfg["dropout"] = round(float(cfg["dropout"]), 6)
        cfg["learning_rate"] = float(cfg["learning_rate"])
        return cfg

    def clip_vector(
        self,
        vector: Sequence[float],
        center: Optional[Sequence[float]] = None,
        active_names: Optional[Iterable[str]] = None,
        radius: Optional[float] = None,
    ) -> List[float]:
        result = [min(1.0, max(0.0, float(v))) for v in vector]
        if center is None:
            return result
        center = list(center)
        active_names = set(active_names or self.names)
        for i, spec in enumerate(self.specs):
            if spec.name not in active_names:
                result[i] = float(center[i])
                continue
            if radius is not None:
                lo = max(0.0, center[i] - radius)
                hi = min(1.0, center[i] + radius)
                result[i] = min(hi, max(lo, result[i]))
        return result

    def sample_vector(
        self,
        rng: random.Random,
        model_name: str,
        center: Optional[Sequence[float]] = None,
        active_names: Optional[Iterable[str]] = None,
        radius: Optional[float] = None,
    ) -> List[float]:
        if center is None:
            center = self.encode_config(self.get_default_config(model_name))
        center = list(center)
        active_names = set(active_names or self.get_active_names(model_name, stage=None))
        vec: List[float] = []
        for idx, spec in enumerate(self.specs):
            if spec.name not in active_names:
                vec.append(center[idx])
                continue
            if radius is None:
                vec.append(rng.random())
            else:
                lo = max(0.0, center[idx] - radius)
                hi = min(1.0, center[idx] + radius)
                vec.append(rng.uniform(lo, hi))
        return self.clip_vector(vec, center=center, active_names=active_names, radius=radius)

    def to_key(self, config: Dict[str, object]) -> str:
        return json.dumps(dict(sorted(config.items())), ensure_ascii=False, sort_keys=True)
