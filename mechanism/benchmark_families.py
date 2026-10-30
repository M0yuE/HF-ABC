from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from run_benchmarks import Variable, WaterTask


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "family_instances.json"
TASK_NAMES = (
    "reservoir_operation",
    "pump_station_scheduling",
    "wastewater_aeration_control",
)
DIFFICULTIES = ("low", "medium", "high")
DIFFICULTY_LEVEL = {name: index for index, name in enumerate(DIFFICULTIES)}


@dataclass(frozen=True)
class InstanceSpec:
    instance_key: str
    task_family: str
    difficulty: str
    replicate: int
    data_seed: int
    constraint_strength: float
    regime_shift: float
    coupling_strength: float
    split: str


def _round(value: float) -> float:
    return round(float(np.clip(value, 0.05, 0.95)), 6)


def build_specs() -> list[InstanceSpec]:
    """Create the preregistered 3 x 3 x 5 benchmark grid.

    Parameters depend only on task, difficulty and replicate identifiers.  They
    must not be edited after development results have been inspected.
    """
    base = {
        "low": (0.22, 0.18, 0.24),
        "medium": (0.53, 0.52, 0.54),
        "high": (0.84, 0.86, 0.83),
    }
    specs: list[InstanceSpec] = []
    for task_index, task in enumerate(TASK_NAMES):
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            locked_primary = (task_index + difficulty_index) % 5
            locked_ids = {locked_primary}
            if task_index != difficulty_index:
                locked_ids.add((locked_primary + 2) % 5)
            for replicate in range(5):
                data_seed = 64100 + 1000 * task_index + 100 * difficulty_index + replicate
                rng = np.random.default_rng(data_seed)
                jitter = rng.uniform(-0.055, 0.055, size=3)
                constraint, regime, coupling = (
                    _round(base[difficulty][i] + jitter[i]) for i in range(3)
                )
                specs.append(
                    InstanceSpec(
                        instance_key=f"{task}__{difficulty}__i{replicate}",
                        task_family=task,
                        difficulty=difficulty,
                        replicate=replicate,
                        data_seed=data_seed,
                        constraint_strength=constraint,
                        regime_shift=regime,
                        coupling_strength=coupling,
                        split="locked_test" if replicate in locked_ids else "development",
                    )
                )
    return specs


def registry_payload() -> dict[str, Any]:
    specs = build_specs()
    instances = [asdict(spec) for spec in specs]
    canonical = json.dumps(instances, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.0",
        "design_status": "frozen_before_any_family_benchmark_run",
        "design_rule": "3 water tasks x 3 difficulties x 5 deterministic instances",
        "split_rule": (
            "Within each task-difficulty cell, locked_primary=(task_index+difficulty_index) mod 5; "
            "for off-diagonal cells also lock (locked_primary+2) mod 5."
        ),
        "development_instances": sum(spec.split == "development" for spec in specs),
        "locked_test_instances": sum(spec.split == "locked_test" for spec in specs),
        "instances_sha256": digest,
        "instances": instances,
    }


def freeze_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    expected = registry_payload()
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"Frozen registry mismatch: {path}. Refusing to overwrite an existing registry."
            )
        return observed
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return expected


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Run benchmark_families.py --freeze before experiments: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != registry_payload():
        raise RuntimeError("Instance registry no longer matches the preregistered construction rule.")
    return payload


class FamilyTask(WaterTask):
    def __init__(self, spec: InstanceSpec, variables: list[Variable]):
        super().__init__(variables)
        self.spec = spec
        # Recorder uses task.name in every trace row; make it instance-specific.
        self.name = spec.instance_key


class ReservoirFamilyTask(FamilyTask):
    def __init__(self, spec: InstanceSpec):
        level = DIFFICULTY_LEVEL[spec.difficulty]
        policies = ("hedging", "balanced") if level == 0 else ("hedging", "balanced", "flood_control")
        variables = [
            Variable("policy", "categorical", values=policies),
            Variable("release_scale", "continuous", 0.60, 1.40),
            Variable("storage_target", "continuous", 0.35, 0.85),
            Variable("gate_count", "integer", 1, 5),
            Variable("rule_segments", "integer", 2, 9),
            Variable(
                "drought_trigger", "continuous", 0.10, 0.50,
                conditional_on=("policy", "hedging"), inactive_value=0.0,
            ),
        ]
        if level >= 1:
            variables.append(
                Variable(
                    "flood_buffer", "continuous", 0.05, 0.35,
                    conditional_on=("policy", "flood_control"), inactive_value=0.0,
                )
            )
        if level >= 2:
            variables.extend(
                [
                    Variable("eco_mode", "categorical", values=("standard", "strict")),
                    Variable(
                        "eco_boost", "continuous", 0.00, 0.45,
                        conditional_on=("eco_mode", "strict"), inactive_value=0.0,
                    ),
                ]
            )
        super().__init__(spec, variables)
        self.level = level
        self.gate_divisor = (4, 3, 2)[level]
        rng = np.random.default_rng(spec.data_seed)
        t = np.arange(self.max_fidelity)
        base_inflow = 17.0 + 9.5 * np.sin(2 * np.pi * (t + 8) / 48)
        base_demand = 15.0 + 4.8 * np.sin(2 * np.pi * (t + 27) / 36)
        # Later regimes are invisible to the smallest fidelity. Their strength is
        # fixed by the registry and creates controlled low/high-fidelity mismatch.
        regime = np.zeros_like(t, dtype=float)
        regime[t >= 36] = 0.45
        regime[t >= 72] = -0.25
        regime[t >= 108] = 0.70
        self.inflow = base_inflow * (1.0 - spec.regime_shift * 0.32 * regime)
        self.inflow += rng.normal(0, 2.2 + spec.regime_shift, len(t))
        self.inflow = np.clip(self.inflow, 1.5, None)
        self.demand = base_demand * (1.0 + spec.regime_shift * 0.22 * regime)
        self.demand += rng.normal(0, 0.8 + 0.7 * spec.regime_shift, len(t))
        self.demand = np.clip(self.demand, 5.0, None)
        self.eco = 5.0 + 1.5 * np.sin(2 * np.pi * t / 24)
        self.capacity = 101.0 - 8.0 * spec.constraint_strength

    def required_gates(self, c: dict[str, Any]) -> int:
        required = int(math.ceil(c["rule_segments"] / self.gate_divisor))
        if self.level >= 2 and c["eco_mode"] == "strict":
            required += int(self.spec.constraint_strength > 0.80)
        return min(5, required)

    def repair_cross_constraints(self, config):
        c = dict(config)
        c["gate_count"] = min(5, max(c["gate_count"], self.required_gates(c)))
        if self.level >= 1 and c["policy"] == "flood_control":
            upper = 0.98 - 0.08 * self.spec.constraint_strength
            if c["storage_target"] + c["flood_buffer"] > upper:
                c["storage_target"] = max(0.35, upper - c["flood_buffer"])
        if self.level >= 2 and c["eco_mode"] == "strict":
            c["release_scale"] = max(c["release_scale"], 0.78 + 0.10 * self.spec.constraint_strength)
        return c

    def feasible(self, c):
        if c["gate_count"] < self.required_gates(c):
            return False
        if self.level >= 1 and c["policy"] == "flood_control":
            upper = 0.98 - 0.08 * self.spec.constraint_strength
            if c["storage_target"] + c["flood_buffer"] > upper + 1e-12:
                return False
        if self.level >= 2 and c["eco_mode"] == "strict":
            if c["release_scale"] < 0.78 + 0.10 * self.spec.constraint_strength - 1e-12:
                return False
        return True

    def violation(self, c):
        v = max(0.0, self.required_gates(c) - c["gate_count"])
        if self.level >= 1 and c["policy"] == "flood_control":
            v += max(
                0.0,
                c["storage_target"] + c["flood_buffer"] - (0.98 - 0.08 * self.spec.constraint_strength),
            )
        if self.level >= 2 and c["eco_mode"] == "strict":
            v += max(0.0, 0.78 + 0.10 * self.spec.constraint_strength - c["release_scale"])
        return v

    def objective(self, c, fidelity):
        storage = 0.55 * self.capacity
        losses = []
        gate_cap = (7.3 - 0.4 * self.spec.constraint_strength) * c["gate_count"]
        coupling = self.spec.coupling_strength
        eco_boost = c.get("eco_boost", 0.0)
        for inflow, demand, eco in zip(self.inflow[:fidelity], self.demand[:fidelity], self.eco[:fidelity]):
            ratio = storage / self.capacity
            eco_required = eco * (1.0 + eco_boost)
            desired = demand * c["release_scale"] + (4.0 + 4.0 * coupling) * (ratio - c["storage_target"])
            if c["policy"] == "hedging" and ratio < c["drought_trigger"]:
                desired *= 0.56 + 0.10 * (1.0 - coupling)
            if self.level >= 1 and c["policy"] == "flood_control" and ratio > 1.0 - c["flood_buffer"]:
                desired += (11.0 + 6.0 * coupling) * (ratio - (1.0 - c["flood_buffer"]))
            release = min(gate_cap, storage + inflow, max(eco_required, desired, 0.0))
            shortage = max(0.0, demand - release) / max(demand, 1.0)
            eco_deficit = max(0.0, eco_required - release) / max(eco_required, 1.0)
            pre_clip = storage + inflow - release
            spill = max(0.0, pre_clip - self.capacity) / self.capacity
            storage = min(self.capacity, max(0.0, pre_clip))
            deviation = (storage / self.capacity - c["storage_target"]) ** 2
            losses.append(
                (3.6 + 1.4 * coupling) * shortage**2
                + (2.1 + 1.2 * coupling) * eco_deficit**2
                + 1.8 * spill**2
                + (0.10 + 0.10 * coupling) * deviation
            )
        complexity = 0.0018 * c["gate_count"] + 0.0008 * c["rule_segments"]
        return float(np.mean(losses) + complexity)


class PumpFamilyTask(FamilyTask):
    MODEL_CAP = {"A": 17.0, "B": 20.0, "C": 23.0}
    MODEL_EFF = {"A": 0.76, "B": 0.82, "C": 0.79}

    def __init__(self, spec: InstanceSpec):
        level = DIFFICULTY_LEVEL[spec.difficulty]
        variables = [
            Variable("pump_model", "categorical", values=("A", "B", "C")),
            Variable("n_pumps", "integer", 1, 5),
            Variable("speed", "continuous", 0.60, 1.20),
            Variable("schedule_blocks", "integer", 2, 12),
            Variable("tariff_mode", "categorical", values=("fixed", "peak_aware")),
            Variable(
                "peak_speed_limit", "continuous", 0.60, 1.00,
                conditional_on=("tariff_mode", "peak_aware"), inactive_value=1.0,
            ),
        ]
        if level >= 1:
            variables.extend(
                [
                    Variable("backup", "categorical", values=("off", "on")),
                    Variable(
                        "backup_units", "integer", 0, 2,
                        conditional_on=("backup", "on"), inactive_value=0,
                    ),
                ]
            )
        if level >= 2:
            variables.extend(
                [
                    Variable("maintenance_mode", "categorical", values=("normal", "reserve")),
                    Variable(
                        "reserve_margin", "continuous", 0.02, 0.20,
                        conditional_on=("maintenance_mode", "reserve"), inactive_value=0.0,
                    ),
                ]
            )
        super().__init__(spec, variables)
        self.level = level
        self.capacity_threshold = 58.0 + 12.0 * spec.constraint_strength
        rng = np.random.default_rng(spec.data_seed)
        t = np.arange(self.max_fidelity)
        base = 47.0 + 17.0 * np.sin(2 * np.pi * (t + 3) / 24) + 6.5 * np.sin(2 * np.pi * t / 9)
        regime = np.zeros_like(t, dtype=float)
        regime[t >= 36] = 0.55
        regime[t >= 72] = -0.20
        regime[t >= 108] = 0.75
        self.regime_profile = regime
        self.demand = base * (1.0 + 0.18 * spec.regime_shift * regime)
        self.demand += rng.normal(0, 2.0 + 1.4 * spec.regime_shift, len(t))
        self.demand = np.clip(self.demand, 18.0, 94.0)
        peak = (t % 24 >= 8) & (t % 24 <= 20)
        self.tariff = np.where(peak, 1.45 + 0.25 * spec.regime_shift * (t >= 72), 0.72)

    def design_capacity(self, c):
        backup_units = c.get("backup_units", 0)
        return c["n_pumps"] * self.MODEL_CAP[c["pump_model"]] * c["speed"] + 9.0 * backup_units

    def required_capacity(self, c):
        margin = c.get("reserve_margin", 0.0)
        return self.capacity_threshold * (1.0 + margin)

    def repair_cross_constraints(self, config):
        c = dict(config)
        if c["tariff_mode"] == "peak_aware":
            c["peak_speed_limit"] = min(c["peak_speed_limit"], c["speed"])
        while self.design_capacity(c) < self.required_capacity(c) and c["n_pumps"] < 5:
            c["n_pumps"] += 1
        if self.design_capacity(c) < self.required_capacity(c):
            backup = c.get("backup_units", 0)
            required_speed = (
                self.required_capacity(c) - 9.0 * backup
            ) / (c["n_pumps"] * self.MODEL_CAP[c["pump_model"]])
            c["speed"] = float(np.clip(required_speed, 0.60, 1.20))
        return c

    def feasible(self, c):
        if self.design_capacity(c) < self.required_capacity(c) - 1e-9:
            return False
        if c["tariff_mode"] == "peak_aware" and c["peak_speed_limit"] > c["speed"] + 1e-12:
            return False
        return True

    def violation(self, c):
        v = max(0.0, self.required_capacity(c) - self.design_capacity(c)) / self.required_capacity(c)
        if c["tariff_mode"] == "peak_aware":
            v += max(0.0, c["peak_speed_limit"] - c["speed"])
        return v

    def objective(self, c, fidelity):
        losses = []
        efficiency = self.MODEL_EFF[c["pump_model"]]
        coupling = self.spec.coupling_strength
        backup_units = c.get("backup_units", 0)
        reserve_margin = c.get("reserve_margin", 0.0)
        model_sensitivity = {"A": -0.22, "B": 0.04, "C": 0.16}[c["pump_model"]]
        for demand, tariff, regime in zip(
            self.demand[:fidelity], self.tariff[:fidelity], self.regime_profile[:fidelity]
        ):
            speed = c["speed"]
            if c["tariff_mode"] == "peak_aware" and tariff > 1.0:
                speed = min(speed, c["peak_speed_limit"])
            # Hydraulic-head changes and equipment derating affect pump types
            # differently; this makes short commissioning windows imperfect
            # proxies for full seasonal operation.
            regime_factor = 1.0 + self.spec.regime_shift * regime * model_sensitivity
            capacity = (
                c["n_pumps"] * self.MODEL_CAP[c["pump_model"]] * speed * regime_factor
                + 9.0 * backup_units
            )
            delivered = min(demand, capacity)
            shortage = max(0.0, demand - delivered) / max(demand, 1.0)
            pressure_penalty = max(0.0, 0.97 - delivered / max(demand, 1.0))
            schedule_target = 5.0 + 4.0 * coupling + 3.0 * self.spec.regime_shift * regime
            cycling = abs(c["schedule_blocks"] - schedule_target) / 10.0
            energy = tariff * delivered * (0.43 + (0.50 + 0.12 * coupling) * speed**2)
            effective_efficiency = efficiency * (1.0 - 0.10 * self.spec.regime_shift * abs(regime))
            energy /= effective_efficiency * 100.0
            reserve_cost = reserve_margin * (0.04 + 0.08 * coupling)
            losses.append(
                energy
                + (15.0 + 7.0 * coupling) * shortage**2
                + (6.0 + 5.0 * coupling) * pressure_penalty**2
                + 0.025 * cycling**2
                + reserve_cost
            )
        complexity = 0.0025 * c["schedule_blocks"] + 0.005 * c["n_pumps"] + 0.004 * backup_units
        return float(np.mean(losses) + complexity)


class WastewaterFamilyTask(FamilyTask):
    def __init__(self, spec: InstanceSpec):
        level = DIFFICULTY_LEVEL[spec.difficulty]
        controllers = ("constant", "ammonia_feedback") if level == 0 else (
            "constant", "ammonia_feedback", "n2o_guard"
        )
        sensors = ("basic", "nitrogen") if level == 0 else ("basic", "nitrogen", "full")
        variables = [
            Variable("controller", "categorical", values=controllers),
            Variable("kla", "continuous", 0.50, 2.00),
            Variable("air_steps", "integer", 1, 8),
            Variable("sensor_set", "categorical", values=sensors),
            Variable(
                "nh4_gain", "continuous", 0.10, 2.00,
                conditional_on=("controller", "ammonia_feedback"), inactive_value=0.0,
            ),
            Variable("aeration_cap", "integer", 2, 10),
        ]
        if level >= 1:
            variables.append(
                Variable(
                    "n2o_threshold", "continuous", 0.20, 1.20,
                    conditional_on=("controller", "n2o_guard"), inactive_value=0.0,
                )
            )
        if level >= 2:
            variables.extend(
                [
                    Variable("energy_mode", "categorical", values=("standard", "constrained")),
                    Variable(
                        "energy_weight", "continuous", 0.05, 0.30,
                        conditional_on=("energy_mode", "constrained"), inactive_value=0.08,
                    ),
                ]
            )
        super().__init__(spec, variables)
        self.level = level
        rng = np.random.default_rng(spec.data_seed)
        t = np.arange(self.max_fidelity)
        regime = np.zeros_like(t, dtype=float)
        regime[t >= 36] = 0.50
        regime[t >= 72] = -0.18
        regime[t >= 108] = 0.65
        self.regime_profile = regime
        self.load = 1.18 + 0.36 * np.sin(2 * np.pi * t / 24) + 0.17 * np.sin(2 * np.pi * t / 7)
        self.load *= 1.0 + 0.22 * spec.regime_shift * regime
        self.load += rng.normal(0, 0.035 + 0.035 * spec.regime_shift, len(t))
        self.temperature = 18.0 + 3.0 * np.sin(2 * np.pi * (t + 5) / 72)
        self.temperature += 1.8 * spec.regime_shift * regime
        nh4, n2o, _ = self.simulate_reference()
        self.obs_nh4 = nh4 + rng.normal(0, 0.014 + 0.008 * spec.regime_shift, len(nh4))
        self.obs_n2o = n2o + rng.normal(0, 0.006 + 0.004 * spec.regime_shift, len(n2o))

    def simulate_reference(self):
        """Generate observations under fixed, preregistered operating-regime drift."""
        nh4, n2o = 1.1, 0.34
        out_nh4, out_n2o, energy = [], [], []
        coupling = self.spec.coupling_strength
        for load, temp, regime in zip(self.load, self.temperature, self.regime_profile):
            # Early observations reflect ammonia feedback. Positive later regimes
            # progressively shift operation toward N2O guarding and stronger air.
            kla = 1.12 + 0.38 * self.spec.regime_shift * regime
            gain = 0.86 + 0.28 * coupling - 0.25 * self.spec.regime_shift * regime
            air = 0.42 * 5 * kla
            if regime > 0.30:
                threshold = 0.44 + 0.12 * (1.0 - coupling)
                air += (0.70 + 0.35 * coupling) * max(0.0, n2o - threshold)
            else:
                air += gain * max(0.0, nh4 - (0.58 - 0.08 * coupling))
            air = min(7.0, max(0.1, air))
            temp_factor = 1.0 + 0.025 * (temp - 18.0)
            nh4 = max(0.0, 0.78 * nh4 + 0.34 * load - 0.17 * air * temp_factor)
            n2o = max(
                0.0,
                0.70 * n2o + (0.08 + 0.04 * coupling) * nh4 + 0.025 * load
                + (0.010 + 0.005 * coupling) * air**2 - 0.035 * air,
            )
            out_nh4.append(nh4)
            out_n2o.append(n2o)
            energy.append(0.025 * air**2)
        return np.asarray(out_nh4), np.asarray(out_n2o), np.asarray(energy)

    def repair_cross_constraints(self, config):
        c = dict(config)
        c["aeration_cap"] = max(c["aeration_cap"], c["air_steps"])
        if c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic":
            c["sensor_set"] = "nitrogen"
        if c["controller"] == "n2o_guard":
            c["sensor_set"] = "full"
        if self.level >= 2 and c["energy_mode"] == "constrained":
            minimum_weight = 0.05 + 0.10 * self.spec.constraint_strength
            c["energy_weight"] = max(c["energy_weight"], minimum_weight)
            cap_limit = int(round(10.0 - 2.0 * self.spec.constraint_strength))
            c["aeration_cap"] = min(c["aeration_cap"], max(c["air_steps"], cap_limit))
        return c

    def feasible(self, c):
        if c["aeration_cap"] < c["air_steps"]:
            return False
        if c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic":
            return False
        if c["controller"] == "n2o_guard" and c["sensor_set"] != "full":
            return False
        if self.level >= 2 and c["energy_mode"] == "constrained":
            minimum_weight = 0.05 + 0.10 * self.spec.constraint_strength
            cap_limit = int(round(10.0 - 2.0 * self.spec.constraint_strength))
            if c["energy_weight"] < minimum_weight - 1e-12 or c["aeration_cap"] > cap_limit:
                return False
        return True

    def violation(self, c):
        v = max(0.0, c["air_steps"] - c["aeration_cap"])
        v += float(c["controller"] == "ammonia_feedback" and c["sensor_set"] == "basic")
        v += float(c["controller"] == "n2o_guard" and c["sensor_set"] != "full")
        if self.level >= 2 and c["energy_mode"] == "constrained":
            minimum_weight = 0.05 + 0.10 * self.spec.constraint_strength
            cap_limit = int(round(10.0 - 2.0 * self.spec.constraint_strength))
            v += max(0.0, minimum_weight - c["energy_weight"])
            v += max(0.0, c["aeration_cap"] - cap_limit)
        return v

    def simulate(self, c, fidelity):
        nh4, n2o = 1.1, 0.34
        out_nh4, out_n2o, energy = [], [], []
        coupling = self.spec.coupling_strength
        for load, temp in zip(self.load[:fidelity], self.temperature[:fidelity]):
            air = 0.42 * c["air_steps"] * c["kla"]
            if c["controller"] == "ammonia_feedback":
                air += c["nh4_gain"] * max(0.0, nh4 - (0.58 - 0.08 * coupling))
            elif c["controller"] == "n2o_guard" and n2o > c.get("n2o_threshold", 0.0):
                air += (0.65 + 0.35 * coupling) * (n2o - c["n2o_threshold"])
            air = min(float(c["aeration_cap"]), max(0.1, air))
            temp_factor = 1.0 + 0.025 * (temp - 18.0)
            nh4 = max(0.0, 0.78 * nh4 + 0.34 * load - 0.17 * air * temp_factor)
            n2o = max(
                0.0,
                0.70 * n2o + (0.08 + 0.04 * coupling) * nh4 + 0.025 * load
                + (0.010 + 0.005 * coupling) * air**2 - 0.035 * air,
            )
            out_nh4.append(nh4)
            out_n2o.append(n2o)
            energy.append(0.025 * air**2)
        return np.asarray(out_nh4), np.asarray(out_n2o), np.asarray(energy)

    def objective(self, c, fidelity):
        nh4, n2o, energy = self.simulate(c, fidelity)
        mse_nh4 = np.mean((nh4 - self.obs_nh4[:fidelity]) ** 2)
        mse_n2o = np.mean((n2o - self.obs_n2o[:fidelity]) ** 2)
        sensor_cost = {"basic": 0.0, "nitrogen": 0.006, "full": 0.012}[c["sensor_set"]]
        energy_weight = c.get("energy_weight", 0.08)
        n2o_weight = 2.1 + 0.8 * self.spec.coupling_strength
        return float(mse_nh4 + n2o_weight * mse_n2o + energy_weight * np.mean(energy) + sensor_cost)


TASK_CLASSES = {
    "reservoir_operation": ReservoirFamilyTask,
    "pump_station_scheduling": PumpFamilyTask,
    "wastewater_aeration_control": WastewaterFamilyTask,
}


def instantiate(spec: InstanceSpec) -> FamilyTask:
    return TASK_CLASSES[spec.task_family](spec)


def specs_from_registry(split: str | None = None) -> list[InstanceSpec]:
    payload = load_registry()
    specs = [InstanceSpec(**item) for item in payload["instances"]]
    if split is not None:
        specs = [spec for spec in specs if spec.split == split]
    return specs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("Use --freeze to create or verify the preregistered registry.")
    frozen = freeze_registry()
    print(json.dumps({key: value for key, value in frozen.items() if key != "instances"}, indent=2))
