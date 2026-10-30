from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results"
FIDELITIES = (9, 36, 72, 144)
KEEP = (16, 8, 4, 4)
CHALLENGERS = (0, 16, 8, 4)
INIT_SIZE = 32
METHODS = (
    "random_projected_full",
    "abc_raw_full",
    "abc_projected_full",
    "abc_projected_mf",
    "hf_abc",
)
ABLATION_METHODS = (
    "hf_abc",
    "hf_no_mf",
    "hf_no_projection",
    "hf_no_egr",
    "hf_no_restart",
)
PROTOCOLS = ("equal_cost", "equal_high_fidelity")


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] = ()
    conditional_on: tuple[str, Any] | None = None
    inactive_value: Any = None


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class WaterTask:
    name = "base"
    max_fidelity = 144

    def __init__(self, variables: Iterable[Variable]):
        self.variables = tuple(variables)
        self.by_name = {v.name: v for v in self.variables}

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for value, spec in zip(np.clip(vector, 0.0, 1.0), self.variables):
            u = float(value)
            if spec.kind == "continuous":
                config[spec.name] = float(spec.low + u * (spec.high - spec.low))
            elif spec.kind == "integer":
                config[spec.name] = int(round(spec.low + u * (spec.high - spec.low)))
            elif spec.kind == "categorical":
                index = min(len(spec.values) - 1, int(math.floor(u * len(spec.values))))
                config[spec.name] = spec.values[index]
            else:
                raise ValueError(spec.kind)
        return config

    def sample_vector(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(len(self.variables))

    def normalize_fields(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(config)
        for spec in self.variables:
            value = normalized.get(spec.name)
            if spec.kind == "continuous":
                normalized[spec.name] = float(np.clip(float(value), spec.low, spec.high))
            elif spec.kind == "integer":
                normalized[spec.name] = int(np.clip(round(float(value)), spec.low, spec.high))
            elif spec.kind == "categorical" and value not in spec.values:
                normalized[spec.name] = spec.values[0]
            if spec.conditional_on is not None:
                parent, active_value = spec.conditional_on
                if normalized.get(parent) != active_value:
                    normalized[spec.name] = spec.inactive_value
        return normalized

    def project(self, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        projected = self.normalize_fields(config)
        projected = self.repair_cross_constraints(projected)
        changed = [name for name in projected if projected.get(name) != config.get(name)]
        return projected, changed

    def repair_cross_constraints(self, config: dict[str, Any]) -> dict[str, Any]:
        return config

    def feasible(self, config: dict[str, Any]) -> bool:
        return True

    def violation(self, config: dict[str, Any]) -> float:
        return 0.0 if self.feasible(config) else 1.0

    def objective(self, config: dict[str, Any], fidelity: int) -> float:
        raise NotImplementedError

    def evaluate(self, config: dict[str, Any], fidelity: int) -> tuple[float, bool]:
        feasible = self.feasible(config)
        if not feasible:
            return 1_000.0 + 100.0 * self.violation(config), False
        return float(self.objective(config, fidelity)), True


class ReservoirTask(WaterTask):
    name = "reservoir_operation"

    def __init__(self):
        super().__init__(
            [
                Variable("policy", "categorical", values=("hedging", "balanced", "flood_control")),
                Variable("release_scale", "continuous", 0.60, 1.40),
                Variable("storage_target", "continuous", 0.35, 0.85),
                Variable("gate_count", "integer", 1, 4),
                Variable("rule_segments", "integer", 2, 8),
                Variable("drought_trigger", "continuous", 0.10, 0.50, conditional_on=("policy", "hedging"), inactive_value=0.0),
                Variable("flood_buffer", "continuous", 0.05, 0.35, conditional_on=("policy", "flood_control"), inactive_value=0.0),
            ]
        )
        rng = np.random.default_rng(7241)
        t = np.arange(self.max_fidelity)
        self.inflow = 17.0 + 10.0 * np.sin(2 * np.pi * (t + 8) / 48) + rng.normal(0, 3.0, len(t))
        self.inflow = np.clip(self.inflow, 2.0, None)
        self.demand = 15.0 + 5.0 * np.sin(2 * np.pi * (t + 27) / 36) + rng.normal(0, 1.2, len(t))
        self.demand = np.clip(self.demand, 6.0, None)
        self.eco = 5.0 + 1.5 * np.sin(2 * np.pi * t / 24)

    def repair_cross_constraints(self, config):
        c = dict(config)
        c["gate_count"] = max(c["gate_count"], int(math.ceil(c["rule_segments"] / 3)))
        if c["policy"] == "flood_control" and c["storage_target"] + c["flood_buffer"] > 1.0:
            c["storage_target"] = round(1.0 - c["flood_buffer"], 8)
        return c

    def feasible(self, c):
        return c["gate_count"] >= math.ceil(c["rule_segments"] / 3) and not (
            c["policy"] == "flood_control" and c["storage_target"] + c["flood_buffer"] > 1.0 + 1e-12
        )

    def violation(self, c):
        v = max(0.0, math.ceil(c["rule_segments"] / 3) - c["gate_count"])
        if c["policy"] == "flood_control":
            v += max(0.0, c["storage_target"] + c["flood_buffer"] - 1.0)
        return v

    def objective(self, c, fidelity):
        storage = 56.0
        losses = []
        gate_cap = 7.5 * c["gate_count"]
        for inflow, demand, eco in zip(self.inflow[:fidelity], self.demand[:fidelity], self.eco[:fidelity]):
            ratio = storage / 100.0
            desired = demand * c["release_scale"] + 5.0 * (ratio - c["storage_target"])
            if c["policy"] == "hedging" and ratio < c["drought_trigger"]:
                desired *= 0.58
            if c["policy"] == "flood_control" and ratio > 1.0 - c["flood_buffer"]:
                desired += 14.0 * (ratio - (1.0 - c["flood_buffer"]))
            desired = max(0.0, desired)
            available = storage + inflow
            release = min(gate_cap, available, max(eco, desired))
            shortage = max(0.0, demand - release) / max(demand, 1.0)
            eco_deficit = max(0.0, eco - release) / max(eco, 1.0)
            pre_clip = storage + inflow - release
            spill = max(0.0, pre_clip - 100.0) / 100.0
            storage = min(100.0, max(0.0, pre_clip))
            deviation = (storage / 100.0 - c["storage_target"]) ** 2
            losses.append(4.0 * shortage**2 + 2.5 * eco_deficit**2 + 1.8 * spill**2 + 0.12 * deviation)
        complexity = 0.002 * c["gate_count"] + 0.001 * c["rule_segments"]
        return float(np.mean(losses) + complexity)


class PumpTask(WaterTask):
    name = "pump_station_scheduling"
    MODEL_CAP = {"A": 17.0, "B": 20.0, "C": 23.0}
    MODEL_EFF = {"A": 0.76, "B": 0.82, "C": 0.79}

    def __init__(self):
        super().__init__(
            [
                Variable("pump_model", "categorical", values=("A", "B", "C")),
                Variable("n_pumps", "integer", 1, 5),
                Variable("speed", "continuous", 0.60, 1.20),
                Variable("schedule_blocks", "integer", 2, 12),
                Variable("tariff_mode", "categorical", values=("fixed", "peak_aware")),
                Variable("peak_speed_limit", "continuous", 0.60, 1.00, conditional_on=("tariff_mode", "peak_aware"), inactive_value=1.0),
                Variable("backup", "categorical", values=("off", "on")),
                Variable("backup_units", "integer", 0, 2, conditional_on=("backup", "on"), inactive_value=0),
            ]
        )
        rng = np.random.default_rng(8117)
        t = np.arange(self.max_fidelity)
        self.demand = 49.0 + 18.0 * np.sin(2 * np.pi * (t + 3) / 24) + 7.0 * np.sin(2 * np.pi * t / 9)
        self.demand += rng.normal(0, 3.0, len(t))
        self.demand = np.clip(self.demand, 20.0, 88.0)
        self.tariff = np.where((t % 24 >= 8) & (t % 24 <= 20), 1.55, 0.72)

    def design_capacity(self, c):
        return c["n_pumps"] * self.MODEL_CAP[c["pump_model"]] * c["speed"] + 9.0 * c["backup_units"]

    def repair_cross_constraints(self, config):
        c = dict(config)
        if c["tariff_mode"] == "peak_aware":
            c["peak_speed_limit"] = min(c["peak_speed_limit"], c["speed"])
        while self.design_capacity(c) < 68.0 and c["n_pumps"] < 5:
            c["n_pumps"] += 1
        if self.design_capacity(c) < 68.0:
            required = (68.0 - 9.0 * c["backup_units"]) / (c["n_pumps"] * self.MODEL_CAP[c["pump_model"]])
            c["speed"] = float(np.clip(required, 0.60, 1.20))
        return c

    def feasible(self, c):
        return self.design_capacity(c) >= 68.0 - 1e-9 and not (
            c["tariff_mode"] == "peak_aware" and c["peak_speed_limit"] > c["speed"] + 1e-12
        )

    def violation(self, c):
        v = max(0.0, 68.0 - self.design_capacity(c)) / 68.0
        if c["tariff_mode"] == "peak_aware":
            v += max(0.0, c["peak_speed_limit"] - c["speed"])
        return v

    def objective(self, c, fidelity):
        losses = []
        efficiency = self.MODEL_EFF[c["pump_model"]]
        for demand, tariff in zip(self.demand[:fidelity], self.tariff[:fidelity]):
            speed = c["speed"]
            if c["tariff_mode"] == "peak_aware" and tariff > 1.0:
                speed = min(speed, c["peak_speed_limit"])
            capacity = c["n_pumps"] * self.MODEL_CAP[c["pump_model"]] * speed + 9.0 * c["backup_units"]
            delivered = min(demand, capacity)
            shortage = max(0.0, demand - delivered) / demand
            pressure = delivered / max(demand, 1.0)
            pressure_penalty = max(0.0, 0.97 - pressure)
            energy = tariff * delivered * (0.45 + 0.55 * speed**2) / (efficiency * 100.0)
            losses.append(energy + 18.0 * shortage**2 + 8.0 * pressure_penalty**2)
        complexity = 0.003 * c["schedule_blocks"] + 0.006 * c["n_pumps"] + 0.004 * c["backup_units"]
        return float(np.mean(losses) + complexity)


class WastewaterTask(WaterTask):
    name = "wastewater_aeration_control"

    def __init__(self):
        super().__init__(
            [
                Variable("controller", "categorical", values=("constant", "ammonia_feedback", "n2o_guard")),
                Variable("kla", "continuous", 0.50, 2.00),
                Variable("air_steps", "integer", 1, 8),
                Variable("sensor_set", "categorical", values=("basic", "nitrogen", "full")),
                Variable("nh4_gain", "continuous", 0.10, 2.00, conditional_on=("controller", "ammonia_feedback"), inactive_value=0.0),
                Variable("n2o_threshold", "continuous", 0.20, 1.20, conditional_on=("controller", "n2o_guard"), inactive_value=0.0),
                Variable("aeration_cap", "integer", 2, 10),
            ]
        )
        rng = np.random.default_rng(9901)
        t = np.arange(self.max_fidelity)
        self.load = 1.20 + 0.38 * np.sin(2 * np.pi * t / 24) + 0.18 * np.sin(2 * np.pi * t / 7)
        self.load += rng.normal(0, 0.05, len(t))
        self.temperature = 18.0 + 3.0 * np.sin(2 * np.pi * (t + 5) / 72)
        hidden = {
            "controller": "ammonia_feedback",
            "kla": 1.28,
            "air_steps": 5,
            "sensor_set": "nitrogen",
            "nh4_gain": 1.05,
            "n2o_threshold": 0.0,
            "aeration_cap": 7,
        }
        nh4, n2o, _ = self.simulate(hidden, self.max_fidelity)
        self.obs_nh4 = nh4 + rng.normal(0, 0.018, len(nh4))
        self.obs_n2o = n2o + rng.normal(0, 0.008, len(n2o))

    def repair_cross_constraints(self, config):
        c = dict(config)
        c["aeration_cap"] = max(c["aeration_cap"], c["air_steps"])
        if c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic":
            c["sensor_set"] = "nitrogen"
        if c["controller"] == "n2o_guard":
            c["sensor_set"] = "full"
        return c

    def feasible(self, c):
        if c["aeration_cap"] < c["air_steps"]:
            return False
        if c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic":
            return False
        if c["controller"] == "n2o_guard" and c["sensor_set"] != "full":
            return False
        return True

    def violation(self, c):
        v = max(0.0, c["air_steps"] - c["aeration_cap"])
        v += float(c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic")
        v += float(c["controller"] == "n2o_guard" and c["sensor_set"] != "full")
        return v

    def simulate(self, c, fidelity):
        nh4 = 1.1
        n2o = 0.34
        out_nh4, out_n2o, energy = [], [], []
        for load, temp in zip(self.load[:fidelity], self.temperature[:fidelity]):
            air = 0.42 * c["air_steps"] * c["kla"]
            if c["controller"] == "ammonia_feedback":
                air += c["nh4_gain"] * max(0.0, nh4 - 0.55)
            elif c["controller"] == "n2o_guard" and n2o > c["n2o_threshold"]:
                air += 0.8 * (n2o - c["n2o_threshold"])
            air = min(float(c["aeration_cap"]), max(0.1, air))
            temp_factor = 1.0 + 0.025 * (temp - 18.0)
            nh4 = max(0.0, 0.78 * nh4 + 0.34 * load - 0.17 * air * temp_factor)
            n2o = max(0.0, 0.70 * n2o + 0.10 * nh4 + 0.025 * load + 0.012 * air**2 - 0.035 * air)
            out_nh4.append(nh4)
            out_n2o.append(n2o)
            energy.append(0.025 * air**2)
        return np.asarray(out_nh4), np.asarray(out_n2o), np.asarray(energy)

    def objective(self, c, fidelity):
        nh4, n2o, energy = self.simulate(c, fidelity)
        mse_nh4 = np.mean((nh4 - self.obs_nh4[:fidelity]) ** 2)
        mse_n2o = np.mean((n2o - self.obs_n2o[:fidelity]) ** 2)
        sensor_cost = {"basic": 0.0, "nitrogen": 0.006, "full": 0.012}[c["sensor_set"]]
        return float(mse_nh4 + 2.5 * mse_n2o + 0.08 * np.mean(energy) + sensor_cost)


TASKS = {
    ReservoirTask.name: ReservoirTask,
    PumpTask.name: PumpTask,
    WastewaterTask.name: WastewaterTask,
}


class Recorder:
    def __init__(self, path: Path, task: WaterTask, method: str, protocol: str, seed: int):
        self.path = path
        self.task = task
        self.method = method
        self.protocol = protocol
        self.seed = seed
        self.call_index = 0
        self.cumulative_cost = 0
        self.rows: list[dict[str, Any]] = []

    def evaluate(self, raw_config, fidelity: int, use_projection: bool, *, stage: int, source: str):
        if use_projection:
            evaluated, changed = self.task.project(raw_config)
        else:
            evaluated, changed = dict(raw_config), []
        start = time.perf_counter()
        score, feasible = self.task.evaluate(evaluated, fidelity)
        elapsed = time.perf_counter() - start
        self.call_index += 1
        self.cumulative_cost += int(fidelity)
        row = {
            "task": self.task.name,
            "method": self.method,
            "protocol": self.protocol,
            "search_seed": self.seed,
            "call_index": self.call_index,
            "stage": stage,
            "source": source,
            "fidelity": int(fidelity),
            "cumulative_cost": self.cumulative_cost,
            "raw_config": jsonable(raw_config),
            "evaluated_config": jsonable(evaluated),
            "projection_applied": bool(use_projection),
            "changed_fields": changed,
            "feasible": bool(feasible),
            "score": float(score),
            "evaluation_seconds": elapsed,
        }
        self.rows.append(row)
        return float(score), evaluated, feasible

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def rank_weights(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    weights = 1.0 / ranks
    return weights / weights.sum()


def mutate_abc(vector: np.ndarray, peer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = vector.copy()
    j = int(rng.integers(0, len(vector)))
    phi = float(rng.uniform(-1.0, 1.0))
    child[j] = child[j] + phi * (child[j] - peer[j])
    return np.clip(child, 0.0, 1.0)


def run_full_search(task, method, protocol, seed, recorder, n_evaluations):
    rng = np.random.default_rng(seed)
    use_projection = method != "abc_raw_full"
    if method == "random_projected_full":
        best = None
        for _ in range(n_evaluations):
            vector = task.sample_vector(rng)
            raw = task.decode(vector)
            score, config, feasible = recorder.evaluate(raw, task.max_fidelity, True, stage=0, source="random")
            if best is None or score < best[0]:
                best = (score, config, feasible)
        return best

    population_size = min(8, max(4, n_evaluations // 3))
    vectors = [task.sample_vector(rng) for _ in range(population_size)]
    scores, configs, feasible_flags = [], [], []
    for vector in vectors:
        score, config, feasible = recorder.evaluate(task.decode(vector), task.max_fidelity, use_projection, stage=0, source="initial")
        scores.append(score)
        configs.append(config)
        feasible_flags.append(feasible)
    trials = [0] * population_size
    calls = population_size
    phase = 0
    while calls < n_evaluations:
        phase += 1
        # Employed-bee updates.
        for i in range(population_size):
            if calls >= n_evaluations:
                break
            peers = [k for k in range(population_size) if k != i]
            k = int(rng.choice(peers))
            child = mutate_abc(vectors[i], vectors[k], rng)
            score, config, feasible = recorder.evaluate(task.decode(child), task.max_fidelity, use_projection, stage=phase, source="employed")
            calls += 1
            if score < scores[i]:
                vectors[i], scores[i], configs[i], feasible_flags[i], trials[i] = child, score, config, feasible, 0
            else:
                trials[i] += 1
        # Onlooker updates with quality-biased source selection.
        weights = rank_weights(np.asarray(scores))
        for _ in range(population_size):
            if calls >= n_evaluations:
                break
            i = int(rng.choice(population_size, p=weights))
            peers = [k for k in range(population_size) if k != i]
            k = int(rng.choice(peers))
            child = mutate_abc(vectors[i], vectors[k], rng)
            score, config, feasible = recorder.evaluate(task.decode(child), task.max_fidelity, use_projection, stage=phase, source="onlooker")
            calls += 1
            if score < scores[i]:
                vectors[i], scores[i], configs[i], feasible_flags[i], trials[i] = child, score, config, feasible, 0
            else:
                trials[i] += 1
        # Scout reset after limit=4.
        for i in range(population_size):
            if calls >= n_evaluations:
                break
            if trials[i] >= 4:
                vector = task.sample_vector(rng)
                score, config, feasible = recorder.evaluate(task.decode(vector), task.max_fidelity, use_projection, stage=phase, source="scout")
                calls += 1
                vectors[i], scores[i], configs[i], feasible_flags[i], trials[i] = vector, score, config, feasible, 0
    best_i = int(np.argmin(scores))
    return scores[best_i], configs[best_i], feasible_flags[best_i]


def typed_child(
    task: WaterTask,
    parent: dict[str, Any],
    rng: np.random.Generator,
    probabilities=(0.70, 0.20, 0.10),
):
    mode = str(rng.choice(("local", "struct", "restart"), p=probabilities))
    if mode == "restart":
        return task.decode(task.sample_vector(rng)), mode
    child = dict(parent)
    if mode == "local":
        choices = [v for v in task.variables if v.kind in {"continuous", "integer"}]
        spec = choices[int(rng.integers(0, len(choices)))]
        if spec.kind == "continuous":
            scale = 0.10 * (spec.high - spec.low)
            base = spec.inactive_value if child.get(spec.name) is None else child.get(spec.name)
            child[spec.name] = float(base) + float(rng.normal(0.0, scale))
        else:
            child[spec.name] = int(child[spec.name]) + int(rng.choice((-1, 1)))
    else:
        choices = [v for v in task.variables if v.kind in {"categorical", "integer"}]
        spec = choices[int(rng.integers(0, len(choices)))]
        if spec.kind == "categorical":
            values = [value for value in spec.values if value != child.get(spec.name)]
            child[spec.name] = values[int(rng.integers(0, len(values)))]
        else:
            child[spec.name] = int(rng.integers(int(spec.low), int(spec.high) + 1))
    return child, mode


def run_multifidelity(task, method, protocol, seed, recorder):
    rng = np.random.default_rng(seed)
    use_projection = method != "hf_no_projection"
    probabilities = {
        "hf_no_egr": (0.0, 0.0, 1.0),
        "hf_no_restart": (0.80, 0.20, 0.0),
    }.get(method, (0.70, 0.20, 0.10))
    active = [{"vector": task.sample_vector(rng)} for _ in range(INIT_SIZE)]
    for item in active:
        item["raw"] = task.decode(item["vector"])
    for stage, fidelity in enumerate(FIDELITIES):
        evaluated = []
        for item in active:
            score, config, feasible = recorder.evaluate(
                item["raw"], fidelity, use_projection, stage=stage, source=item.get("source", "initial")
            )
            evaluated.append({**item, "score": score, "config": config, "feasible": feasible})
        evaluated.sort(key=lambda x: x["score"])
        survivors = evaluated[: KEEP[stage]]
        if stage == len(FIDELITIES) - 1:
            best = survivors[0]
            return best["score"], best["config"], best["feasible"]

        n_children = CHALLENGERS[stage + 1]
        scores = np.asarray([item["score"] for item in survivors])
        weights = rank_weights(scores)
        children = []
        for _ in range(n_children):
            i = int(rng.choice(len(survivors), p=weights))
            parent = survivors[i]
            if method == "abc_projected_mf":
                peers = [k for k in range(len(survivors)) if k != i]
                k = int(rng.choice(peers)) if peers else i
                child_vector = mutate_abc(parent["vector"], survivors[k]["vector"], rng)
                child_raw = task.decode(child_vector)
                children.append({"vector": child_vector, "raw": child_raw, "source": "abc_neighbor"})
            else:
                child_raw, mode = typed_child(task, parent["config"], rng, probabilities=probabilities)
                # Vector is retained only for traceable state; HF-ABC mutates typed configs.
                children.append({"vector": task.sample_vector(rng), "raw": child_raw, "source": mode})
        active = survivors + children
    raise AssertionError("unreachable")


def run_typed_full(task, protocol, seed, recorder, n_evaluations):
    rng = np.random.default_rng(seed)
    population = []
    initial = min(8, n_evaluations)
    for _ in range(initial):
        raw = task.decode(task.sample_vector(rng))
        score, config, feasible = recorder.evaluate(raw, task.max_fidelity, True, stage=0, source="initial")
        population.append({"score": score, "config": config, "feasible": feasible})
    calls = initial
    while calls < n_evaluations:
        population.sort(key=lambda x: x["score"])
        elite = population[: min(4, len(population))]
        weights = rank_weights(np.asarray([x["score"] for x in elite]))
        parent = elite[int(rng.choice(len(elite), p=weights))]
        raw, mode = typed_child(task, parent["config"], rng)
        score, config, feasible = recorder.evaluate(raw, task.max_fidelity, True, stage=1, source=mode)
        calls += 1
        population.append({"score": score, "config": config, "feasible": feasible})
        population.sort(key=lambda x: x["score"])
        population = population[:8]
    best = min(population, key=lambda x: x["score"])
    return best["score"], best["config"], best["feasible"]


def mf_cost() -> int:
    active = INIT_SIZE
    total = 0
    for stage, fidelity in enumerate(FIDELITIES):
        total += active * fidelity
        if stage < len(FIDELITIES) - 1:
            active = KEEP[stage] + CHALLENGERS[stage + 1]
    return total


def run_one(task_name, method, protocol, seed, output):
    task = TASKS[task_name]()
    trace_path = output / "traces" / f"{task_name}__{protocol}__{method}__seed{seed}.jsonl"
    recorder = Recorder(trace_path, task, method, protocol, seed)
    start = time.perf_counter()
    if method in {"abc_projected_mf", "hf_abc", "hf_no_projection", "hf_no_egr", "hf_no_restart"}:
        score, config, feasible = run_multifidelity(task, method, protocol, seed, recorder)
    elif method == "hf_no_mf":
        score, config, feasible = run_typed_full(
            task, protocol, seed, recorder, mf_cost() // task.max_fidelity
        )
    else:
        if protocol == "equal_cost":
            n_evaluations = max(1, mf_cost() // task.max_fidelity)
        else:
            n_evaluations = KEEP[-2] + CHALLENGERS[-1]
        score, config, feasible = run_full_search(task, method, protocol, seed, recorder, n_evaluations)
    elapsed = time.perf_counter() - start
    recorder.save()
    return {
        "task": task_name,
        "method": method,
        "protocol": protocol,
        "search_seed": seed,
        "final_score": float(score),
        "final_feasible": bool(feasible),
        "objective_calls": recorder.call_index,
        "total_cost_units": recorder.cumulative_cost,
        "wall_clock_seconds": elapsed,
        "selected_config": json.dumps(jsonable(config), ensure_ascii=False, sort_keys=True),
        "trace_file": str(trace_path.relative_to(output)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(rows: list[dict[str, Any]], output: Path):
    summary = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["protocol"], row["method"]), []).append(row)
    for (task, protocol, method), items in sorted(grouped.items()):
        values = np.asarray([x["final_score"] for x in items], dtype=float)
        n = len(values)
        sem = stats.sem(values) if n > 1 else 0.0
        margin = stats.t.ppf(0.975, n - 1) * sem if n > 1 else 0.0
        summary.append(
            {
                "task": task,
                "protocol": protocol,
                "method": method,
                "n_search_seeds": n,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if n > 1 else 0.0,
                "ci95_low": float(values.mean() - margin),
                "ci95_high": float(values.mean() + margin),
                "mean_cost_units": float(np.mean([x["total_cost_units"] for x in items])),
                "mean_wall_clock_seconds": float(np.mean([x["wall_clock_seconds"] for x in items])),
                "feasible_selected_rate": float(np.mean([x["final_feasible"] for x in items])),
            }
        )
    write_csv(output / "summary_by_method.csv", summary)

    paired = []
    for task in TASKS:
        for protocol in PROTOCOLS:
            hf = sorted(
                [r for r in rows if r["task"] == task and r["protocol"] == protocol and r["method"] == "hf_abc"],
                key=lambda r: r["search_seed"],
            )
            p_group = []
            start_index = len(paired)
            for method in METHODS:
                if method == "hf_abc":
                    continue
                base = sorted(
                    [r for r in rows if r["task"] == task and r["protocol"] == protocol and r["method"] == method],
                    key=lambda r: r["search_seed"],
                )
                x = np.asarray([r["final_score"] for r in hf])
                y = np.asarray([r["final_score"] for r in base])
                if len(x) != len(y) or len(x) == 0:
                    continue
                if np.allclose(x, y):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                better_probability = float(np.mean(x < y) + 0.5 * np.mean(x == y))
                paired.append(
                    {
                        "task": task,
                        "protocol": protocol,
                        "comparison": f"hf_abc vs {method}",
                        "n_pairs": len(x),
                        "median_paired_difference_hf_minus_baseline": float(np.median(x - y)),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "holm_adjusted_p": 0.0,
                        "probability_hf_better": better_probability,
                    }
                )
                p_group.append(p_value)
            adjusted = holm_adjust(p_group)
            for offset, value in enumerate(adjusted):
                paired[start_index + offset]["holm_adjusted_p"] = value
    write_csv(output / "paired_tests_vs_hf.csv", paired)

    friedman = []
    for task in TASKS:
        for protocol in PROTOCOLS:
            arrays = []
            for method in METHODS:
                items = sorted(
                    [r for r in rows if r["task"] == task and r["protocol"] == protocol and r["method"] == method],
                    key=lambda r: r["search_seed"],
                )
                arrays.append(np.asarray([r["final_score"] for r in items]))
            if arrays and len(arrays[0]) >= 2:
                result = stats.friedmanchisquare(*arrays)
                friedman.append(
                    {
                        "task": task,
                        "protocol": protocol,
                        "n_search_seeds": len(arrays[0]),
                        "n_methods": len(arrays),
                        "statistic": float(result.statistic),
                        "p_value": float(result.pvalue),
                    }
                )
    write_csv(output / "friedman_tests.csv", friedman)


def summarize_ablations(rows: list[dict[str, Any]], output: Path):
    summary = []
    for task in TASKS:
        for method in ABLATION_METHODS:
            items = sorted(
                [r for r in rows if r["task"] == task and r["method"] == method],
                key=lambda r: r["search_seed"],
            )
            values = np.asarray([x["final_score"] for x in items], dtype=float)
            n = len(values)
            sem = stats.sem(values) if n > 1 else 0.0
            margin = stats.t.ppf(0.975, n - 1) * sem if n > 1 else 0.0
            summary.append(
                {
                    "task": task,
                    "method": method,
                    "n_search_seeds": n,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if n > 1 else 0.0,
                    "ci95_low": float(values.mean() - margin),
                    "ci95_high": float(values.mean() + margin),
                    "mean_cost_units": float(np.mean([x["total_cost_units"] for x in items])),
                    "feasible_selected_rate": float(np.mean([x["final_feasible"] for x in items])),
                }
            )
    write_csv(output / "ablation_summary.csv", summary)

    tests = []
    for task in TASKS:
        hf = sorted(
            [r for r in rows if r["task"] == task and r["method"] == "hf_abc"],
            key=lambda r: r["search_seed"],
        )
        x = np.asarray([r["final_score"] for r in hf])
        start = len(tests)
        p_values = []
        for method in ABLATION_METHODS:
            if method == "hf_abc":
                continue
            base = sorted(
                [r for r in rows if r["task"] == task and r["method"] == method],
                key=lambda r: r["search_seed"],
            )
            y = np.asarray([r["final_score"] for r in base])
            result = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            p_values.append(float(result.pvalue))
            tests.append(
                {
                    "task": task,
                    "comparison": f"hf_abc vs {method}",
                    "n_pairs": len(x),
                    "median_paired_difference_hf_minus_ablation": float(np.median(x - y)),
                    "p_value": float(result.pvalue),
                    "holm_adjusted_p": 0.0,
                    "probability_hf_better": float(np.mean(x < y) + 0.5 * np.mean(x == y)),
                }
            )
        for offset, adjusted in enumerate(holm_adjust(p_values)):
            tests[start + offset]["holm_adjusted_p"] = adjusted
    write_csv(output / "ablation_tests_vs_hf.csv", tests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    n_seeds = 2 if args.smoke else args.seeds
    task_names = tuple(TASKS) if not args.smoke else (ReservoirTask.name, WastewaterTask.name)
    protocols = ("equal_cost",) if args.ablations else (PROTOCOLS if not args.smoke else ("equal_cost",))
    methods = ABLATION_METHODS if args.ablations else METHODS
    if args.ablations:
        output = args.output / f"ablations_{n_seeds}_seeds"
    else:
        output = args.output / ("smoke" if args.smoke else f"full_{n_seeds}_seeds")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "tasks": task_names,
        "methods": methods,
        "protocols": protocols,
        "search_seeds": list(range(202600, 202600 + n_seeds)),
        "fidelities": FIDELITIES,
        "initial_population": INIT_SIZE,
        "keep": KEEP,
        "challengers": CHALLENGERS,
        "multi_fidelity_cost_units": mf_cost(),
        "full_fidelity": 144,
        "note": "Surrogate CPU tasks; not field-calibrated models and not replacements for Avedore results.",
    }
    (output / "benchmark_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    total = len(task_names) * len(protocols) * len(methods) * n_seeds
    done = 0
    for task_name in task_names:
        for protocol in protocols:
            for method in methods:
                for seed in config["search_seeds"]:
                    rows.append(run_one(task_name, method, protocol, seed, output))
                    done += 1
                    print(f"[{done}/{total}] {task_name} {protocol} {method} seed={seed}", flush=True)
    write_csv(output / "run_summaries.csv", rows)
    if n_seeds >= 2:
        if args.ablations:
            summarize_ablations(rows, output)
        else:
            summarize(rows, output)
    print(output)


if __name__ == "__main__":
    main()
