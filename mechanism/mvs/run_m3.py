from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


MVS_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = MVS_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(MVS_ROOT))

import benchmark_families as families
import run_benchmarks as bench
import run_domain_diagnostics as domain


SCHEMA_VERSION = "1.0"
FIDELITIES = tuple(int(value) for value in bench.FIDELITIES)
KEEP = tuple(int(value) for value in bench.KEEP)
CHALLENGERS = tuple(int(value) for value in bench.CHALLENGERS)
INIT_SIZE = int(bench.INIT_SIZE)
SELECTION_PATH = MVS_ROOT / "results" / "domain_v1_1" / "selection_decision.json"
METHODS = (
    "hf_abc",
    "hf_no_projection",
    "hf_no_mf",
    "hf_uniform_parent_mf",
    "hf_no_restart",
    "abc_projected_mf",
    "random_projected_mf",
)
TYPED_PROBABILITIES = {
    "hf_abc": (0.70, 0.20, 0.10),
    "hf_no_projection": (0.70, 0.20, 0.10),
    "hf_uniform_parent_mf": (0.70, 0.20, 0.10),
    "hf_no_restart": (0.80, 0.20, 0.00),
}


def jsonable(value: Any) -> Any:
    return domain.jsonable(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_key(task: bench.WaterTask, config: dict[str, Any]) -> str:
    ordered = {spec.name: config.get(spec.name) for spec in task.variables}
    return json.dumps(jsonable(ordered), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def field_violation(spec: bench.Variable, config: dict[str, Any]) -> float:
    value = config.get(spec.name)
    if spec.conditional_on is not None:
        parent, active_value = spec.conditional_on
        if config.get(parent) != active_value:
            return 0.0 if value == spec.inactive_value else 1.0
    if value is None:
        return 1.0
    if spec.kind in {"continuous", "integer"}:
        numeric = float(value)
        outside = max(0.0, float(spec.low) - numeric) + max(0.0, numeric - float(spec.high))
        width = max(1e-12, float(spec.high) - float(spec.low))
        integer_error = abs(numeric - round(numeric)) if spec.kind == "integer" else 0.0
        return outside / width + integer_error
    return 0.0 if value in spec.values else 1.0


def raw_violation(task: bench.WaterTask, config: dict[str, Any]) -> float:
    return float(sum(field_violation(spec, config) for spec in task.variables) + task.violation(config))


def raw_feasible(task: bench.WaterTask, config: dict[str, Any]) -> bool:
    return raw_violation(task, config) <= 1e-12 and bool(task.feasible(config))


class Ledger:
    def __init__(
        self,
        task: bench.WaterTask,
        method: str,
        seed: int,
        trace_path: Path,
    ):
        self.task = task
        self.method = method
        self.seed = int(seed)
        self.trace_path = trace_path
        self.request_count = 0
        self.objective_call_count = 0
        self.cache_hit_count = 0
        self.requested_cost = 0
        self.actual_cost = 0
        self.cache: dict[tuple[str, int], tuple[float, bool, dict[str, Any]]] = {}
        self.rows: list[dict[str, Any]] = []

    def evaluate(
        self,
        raw_config: dict[str, Any],
        fidelity: int,
        *,
        use_projection: bool,
        stage: int,
        source: str,
        candidate_id: str,
        parent_id: str | None,
    ) -> tuple[float, dict[str, Any], bool]:
        self.request_count += 1
        self.requested_cost += int(fidelity)
        before_violation = raw_violation(self.task, raw_config)
        if use_projection:
            evaluated, changed = self.task.project(raw_config)
            key = domain.canonical_key(self.task, evaluated)
        else:
            evaluated, changed = dict(raw_config), []
            key = exact_key(self.task, evaluated)
        after_violation = raw_violation(self.task, evaluated)
        cache_key = (key, int(fidelity))
        started = time.perf_counter()
        cache_hit = cache_key in self.cache
        if cache_hit:
            score, feasible, cached_config = self.cache[cache_key]
            evaluated = dict(cached_config)
            elapsed = 0.0
            self.cache_hit_count += 1
        else:
            if use_projection:
                score, feasible = self.task.evaluate(evaluated, int(fidelity))
            else:
                feasible = raw_feasible(self.task, evaluated)
                score = (
                    float(self.task.objective(evaluated, int(fidelity)))
                    if feasible
                    else 1_000.0 + 100.0 * after_violation
                )
            elapsed = time.perf_counter() - started
            self.objective_call_count += 1
            self.actual_cost += int(fidelity)
            self.cache[cache_key] = (float(score), bool(feasible), dict(evaluated))
        self.rows.append(
            {
                "instance_key": self.task.name,
                "method": self.method,
                "search_seed": self.seed,
                "request_index": self.request_count,
                "objective_call_index": None if cache_hit else self.objective_call_count,
                "stage": int(stage),
                "source": source,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "fidelity": int(fidelity),
                "raw_config": jsonable(raw_config),
                "evaluated_config": jsonable(evaluated),
                "canonical_key": key,
                "projection_applied": bool(use_projection),
                "changed_fields": changed,
                "changed_field_count": len(changed),
                "projection_distance": domain.projection_distance(
                    self.task, raw_config, evaluated
                ),
                "raw_violation": before_violation,
                "evaluated_violation": after_violation,
                "feasible": bool(feasible),
                "score": float(score),
                "cache_hit": cache_hit,
                "requested_cost": self.requested_cost,
                "actual_cost": self.actual_cost,
                "evaluation_seconds": elapsed,
            }
        )
        return float(score), evaluated, bool(feasible)

    def save(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def rank_weights(scores: list[float]) -> np.ndarray:
    return bench.rank_weights(np.asarray(scores, dtype=float))


def typed_child(
    task: bench.WaterTask,
    parent: dict[str, Any],
    rng: np.random.Generator,
    probabilities: tuple[float, float, float],
) -> tuple[dict[str, Any], str]:
    return bench.typed_child(task, parent, rng, probabilities=probabilities)


def initial_population(
    task: bench.WaterTask,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    active = []
    for index in range(INIT_SIZE):
        vector = task.sample_vector(rng)
        active.append(
            {
                "candidate_id": f"c{index:04d}",
                "parent_id": None,
                "vector": vector,
                "raw": task.decode(vector),
                "source": "initial",
            }
        )
    return active


def run_multifidelity(
    task: bench.WaterTask,
    method: str,
    seed: int,
    ledger: Ledger,
) -> tuple[float, dict[str, Any], bool]:
    rng = np.random.default_rng(seed)
    active = initial_population(task, rng)
    next_id = INIT_SIZE
    use_projection = method != "hf_no_projection"
    for stage, fidelity in enumerate(FIDELITIES):
        evaluated = []
        for item in active:
            score, config, feasible = ledger.evaluate(
                item["raw"],
                fidelity,
                use_projection=use_projection,
                stage=stage,
                source=item["source"],
                candidate_id=item["candidate_id"],
                parent_id=item["parent_id"],
            )
            evaluated.append(
                {
                    **item,
                    "score": score,
                    "config": config,
                    "feasible": feasible,
                }
            )
        evaluated.sort(key=lambda item: (item["score"], item["candidate_id"]))
        survivors = evaluated[: KEEP[stage]]
        if stage == len(FIDELITIES) - 1:
            best = survivors[0]
            return float(best["score"]), dict(best["config"]), bool(best["feasible"])

        child_count = CHALLENGERS[stage + 1]
        children: list[dict[str, Any]] = []
        if method == "random_projected_mf":
            parent_indices = [None] * child_count
        elif method == "hf_uniform_parent_mf":
            parent_indices = [
                int(rng.integers(0, len(survivors))) for _ in range(child_count)
            ]
        else:
            weights = rank_weights([float(item["score"]) for item in survivors])
            parent_indices = [
                int(rng.choice(len(survivors), p=weights)) for _ in range(child_count)
            ]

        for parent_index in parent_indices:
            candidate_id = f"c{next_id:04d}"
            next_id += 1
            if method == "random_projected_mf":
                vector = task.sample_vector(rng)
                raw = task.decode(vector)
                source = "random"
                parent_id = None
            else:
                parent = survivors[int(parent_index)]
                parent_id = str(parent["candidate_id"])
                if method == "abc_projected_mf":
                    peers = [
                        index for index in range(len(survivors)) if index != parent_index
                    ]
                    peer_index = int(rng.choice(peers)) if peers else int(parent_index)
                    vector = bench.mutate_abc(
                        np.asarray(parent["vector"]),
                        np.asarray(survivors[peer_index]["vector"]),
                        rng,
                    )
                    raw = task.decode(vector)
                    source = "abc_neighbor"
                else:
                    probabilities = TYPED_PROBABILITIES[method]
                    raw, source = typed_child(
                        task, dict(parent["config"]), rng, probabilities
                    )
                    vector = task.sample_vector(rng)
            children.append(
                {
                    "candidate_id": candidate_id,
                    "parent_id": parent_id,
                    "vector": vector,
                    "raw": raw,
                    "source": source,
                }
            )
        active = survivors + children
    raise AssertionError("unreachable")


def run_full_fidelity(
    task: bench.WaterTask,
    seed: int,
    ledger: Ledger,
) -> tuple[float, dict[str, Any], bool]:
    rng = np.random.default_rng(seed)
    evaluation_budget = multifidelity_requested_cost() // task.max_fidelity
    population: list[dict[str, Any]] = []
    next_id = 0
    initial = min(8, evaluation_budget)
    for _ in range(initial):
        candidate_id = f"c{next_id:04d}"
        next_id += 1
        raw = task.decode(task.sample_vector(rng))
        score, config, feasible = ledger.evaluate(
            raw,
            task.max_fidelity,
            use_projection=True,
            stage=0,
            source="initial",
            candidate_id=candidate_id,
            parent_id=None,
        )
        population.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "config": config,
                "feasible": feasible,
            }
        )
    while ledger.request_count < evaluation_budget:
        population.sort(key=lambda item: (item["score"], item["candidate_id"]))
        elite = population[: min(4, len(population))]
        weights = rank_weights([float(item["score"]) for item in elite])
        parent = elite[int(rng.choice(len(elite), p=weights))]
        raw, source = typed_child(
            task,
            dict(parent["config"]),
            rng,
            TYPED_PROBABILITIES["hf_abc"],
        )
        candidate_id = f"c{next_id:04d}"
        next_id += 1
        score, config, feasible = ledger.evaluate(
            raw,
            task.max_fidelity,
            use_projection=True,
            stage=1,
            source=source,
            candidate_id=candidate_id,
            parent_id=str(parent["candidate_id"]),
        )
        population.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "config": config,
                "feasible": feasible,
            }
        )
        population.sort(key=lambda item: (item["score"], item["candidate_id"]))
        population = population[:8]
    best = min(population, key=lambda item: (item["score"], item["candidate_id"]))
    return float(best["score"]), dict(best["config"]), bool(best["feasible"])


def multifidelity_requested_cost() -> int:
    active = INIT_SIZE
    total = 0
    for stage, fidelity in enumerate(FIDELITIES):
        total += active * fidelity
        if stage < len(FIDELITIES) - 1:
            active = KEEP[stage] + CHALLENGERS[stage + 1]
    return total


def run_one(
    spec: families.InstanceSpec,
    method: str,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    task = families.instantiate(spec)
    trace_path = output / "traces" / f"{spec.instance_key}__{method}__seed{seed}.jsonl"
    ledger = Ledger(task, method, seed, trace_path)
    started = time.perf_counter()
    if method == "hf_no_mf":
        score, config, feasible = run_full_fidelity(task, seed, ledger)
    else:
        score, config, feasible = run_multifidelity(task, method, seed, ledger)
    wall_clock = time.perf_counter() - started
    ledger.save()
    return {
        "instance_key": spec.instance_key,
        "task_family": spec.task_family,
        "difficulty": spec.difficulty,
        "method": method,
        "search_seed": seed,
        "final_score": score,
        "final_feasible": feasible,
        "request_count": ledger.request_count,
        "objective_call_count": ledger.objective_call_count,
        "cache_hit_count": ledger.cache_hit_count,
        "requested_cost": ledger.requested_cost,
        "actual_cost": ledger.actual_cost,
        "wall_clock_seconds": wall_clock,
        "unique_cache_keys": len(ledger.cache),
        "selected_config": json.dumps(
            jsonable(config), ensure_ascii=False, sort_keys=True
        ),
        "trace_file": str(trace_path.relative_to(output)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_selected_specs() -> list[families.InstanceSpec]:
    decision = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected = set(decision["selected_instance_keys"])
    specs = [
        spec for spec in families.specs_from_registry() if spec.instance_key in selected
    ]
    if len(specs) != len(selected):
        raise RuntimeError("Selected instance list does not match family registry.")
    return specs


def build_manifest(
    specs: list[families.InstanceSpec],
    methods: tuple[str, ...],
    seeds: list[int],
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "M3 budget-paired mechanism smoke" if smoke else "M3 main",
        "smoke": smoke,
        "instance_keys": [spec.instance_key for spec in specs],
        "methods": list(methods),
        "search_seeds": seeds,
        "fidelities": list(FIDELITIES),
        "keep": list(KEEP),
        "challengers": list(CHALLENGERS),
        "initial_size": INIT_SIZE,
        "requested_cost_per_run": multifidelity_requested_cost(),
        "no_restart_probabilities": list(TYPED_PROBABILITIES["hf_no_restart"]),
        "selection_decision_sha256": sha256_file(SELECTION_PATH),
        "source_sha256": {
            "run_m3.py": sha256_file(Path(__file__)),
            "run_domain_diagnostics.py": sha256_file(
                MVS_ROOT / "run_domain_diagnostics.py"
            ),
            "benchmark_families.py": sha256_file(
                BENCHMARK_ROOT / "benchmark_families.py"
            ),
            "run_benchmarks.py": sha256_file(BENCHMARK_ROOT / "run_benchmarks.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--execute-main", action="store_true")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--output", type=Path, default=MVS_ROOT / "results")
    args = parser.parse_args()
    if args.execute_smoke == args.execute_main:
        raise SystemExit("Choose exactly one of --execute-smoke or --execute-main.")
    smoke = bool(args.execute_smoke)
    if smoke:
        if args.seeds is not None:
            raise SystemExit("Smoke seed is fixed; do not pass --seeds.")
        seeds = [303600]
        specs = load_selected_specs()[:2]
        output = args.output / "m3_smoke_v1"
    else:
        if args.seeds not in {10, 20}:
            raise SystemExit("Main run requires a frozen --seeds 10 or --seeds 20.")
        seeds = list(range(303600, 303600 + int(args.seeds)))
        specs = load_selected_specs()
        output = args.output / f"m3_main_v1_{args.seeds}_seeds"
    methods = METHODS
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(specs, methods, seeds, smoke)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise RuntimeError("Existing M3 manifest differs; refusing mixed run.")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    rows: list[dict[str, Any]] = []
    total = len(specs) * len(methods) * len(seeds)
    done = 0
    for spec in specs:
        for method in methods:
            for seed in seeds:
                done += 1
                print(
                    f"[{done}/{total}] {spec.instance_key} {method} seed={seed}",
                    flush=True,
                )
                rows.append(run_one(spec, method, seed, output))
                write_csv(output / "run_summaries.csv", rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
