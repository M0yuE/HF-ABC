from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


MVS_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = MVS_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

import benchmark_families as families
import run_benchmarks as bench


SCHEMA_VERSION = "1.1"
FIDELITIES = tuple(int(value) for value in bench.FIDELITIES)
DEFAULT_SAMPLES = 500
SELECTION_SALT = "HFABC-MVS-v1.1"
THRESHOLDS = {
    "projected_feasible_rate": 1.0,
    "minimum_projection_trigger_rate": 0.05,
    "minimum_low_high_spearman": 0.20,
    "maximum_low_high_spearman": 0.95,
    "minimum_variable_kinds": 3,
    "maximum_selected_instances": 15,
    "minimum_selected_instances": 6,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_instance_hash(instance_key: str) -> str:
    payload = f"{SELECTION_SALT}|{instance_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_key(task: bench.WaterTask, config: dict[str, Any]) -> str:
    normalized = task.normalize_fields(config)
    ordered = {spec.name: normalized.get(spec.name) for spec in task.variables}
    return json.dumps(jsonable(ordered), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def field_distance(spec: bench.Variable, left: Any, right: Any) -> float:
    if left == right:
        return 0.0
    if left is None or right is None:
        return 1.0
    if spec.kind in {"continuous", "integer"}:
        width = float(spec.high) - float(spec.low)
        if width <= 0:
            return 1.0
        return min(1.0, abs(float(left) - float(right)) / width)
    return 1.0


def projection_distance(
    task: bench.WaterTask,
    raw: dict[str, Any],
    projected: dict[str, Any],
) -> float:
    if not task.variables:
        return 0.0
    return float(
        np.mean(
            [
                field_distance(spec, raw.get(spec.name), projected.get(spec.name))
                for spec in task.variables
            ]
        )
    )


def safe_rank_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if kind == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    if kind == "kendall":
        return float(stats.kendalltau(x, y).statistic)
    raise ValueError(kind)


def top_retention(
    low_scores: np.ndarray,
    high_scores: np.ndarray,
    high_fraction: float = 0.10,
    low_fraction: float = 0.25,
) -> float:
    count = len(high_scores)
    if count == 0:
        return float("nan")
    high_n = max(1, int(math.ceil(count * high_fraction)))
    low_n = max(high_n, int(math.ceil(count * low_fraction)))
    high_top = set(np.argsort(high_scores, kind="stable")[:high_n].tolist())
    low_top = set(np.argsort(low_scores, kind="stable")[:low_n].tolist())
    return len(high_top & low_top) / high_n


def variable_structure(task: bench.WaterTask) -> tuple[int, bool]:
    kinds = len({spec.kind for spec in task.variables})
    has_conditional = any(spec.conditional_on is not None for spec in task.variables)
    has_cross_constraint = type(task).repair_cross_constraints is not bench.WaterTask.repair_cross_constraints
    return kinds, bool(has_conditional or has_cross_constraint)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_instance(
    spec: families.InstanceSpec,
    samples: int,
    candidate_stream,
) -> dict[str, Any]:
    task = families.instantiate(spec)
    rng = np.random.default_rng(880000 + int(spec.data_seed))
    candidates: list[dict[str, Any]] = []
    key_counts: dict[str, int] = {}
    for candidate_index in range(samples):
        raw = task.decode(task.sample_vector(rng))
        raw_feasible = bool(task.feasible(raw))
        raw_violation = float(task.violation(raw))
        projected, changed = task.project(raw)
        projected_feasible = bool(task.feasible(projected))
        projected_violation = float(task.violation(projected))
        key = canonical_key(task, projected)
        key_counts[key] = key_counts.get(key, 0) + 1
        scores = {
            str(fidelity): float(task.evaluate(projected, fidelity)[0])
            for fidelity in FIDELITIES
        }
        row = {
            "instance_key": spec.instance_key,
            "task_family": spec.task_family,
            "difficulty": spec.difficulty,
            "candidate_index": candidate_index,
            "raw_feasible": raw_feasible,
            "raw_violation": raw_violation,
            "projected_feasible": projected_feasible,
            "projected_violation": projected_violation,
            "projection_triggered": bool(changed),
            "changed_fields": changed,
            "changed_field_count": len(changed),
            "projection_distance": projection_distance(task, raw, projected),
            "canonical_key": key,
            "scores": scores,
        }
        candidate_stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        candidates.append(row)

    projected_feasible = np.asarray([row["projected_feasible"] for row in candidates], dtype=float)
    projection_triggered = np.asarray([row["projection_triggered"] for row in candidates], dtype=float)
    high = np.asarray([row["scores"][str(FIDELITIES[-1])] for row in candidates], dtype=float)
    low = np.asarray([row["scores"][str(FIDELITIES[0])] for row in candidates], dtype=float)
    median_high = float(np.median(high))
    iqr_high = float(np.percentile(high, 75) - np.percentile(high, 25))
    iqr_floor = 1e-8 * max(1.0, abs(median_high))
    kinds, has_structure = variable_structure(task)
    unique_keys = len(key_counts)
    return {
        "instance_key": spec.instance_key,
        "task_family": spec.task_family,
        "difficulty": spec.difficulty,
        "replicate": spec.replicate,
        "data_seed": spec.data_seed,
        "n_candidates": samples,
        "raw_feasible_rate": float(np.mean([row["raw_feasible"] for row in candidates])),
        "projected_feasible_rate": float(projected_feasible.mean()),
        "projection_trigger_rate": float(projection_triggered.mean()),
        "mean_raw_violation": float(np.mean([row["raw_violation"] for row in candidates])),
        "mean_projected_violation": float(
            np.mean([row["projected_violation"] for row in candidates])
        ),
        "mean_changed_fields": float(
            np.mean([row["changed_field_count"] for row in candidates])
        ),
        "mean_projection_distance": float(
            np.mean([row["projection_distance"] for row in candidates])
        ),
        "unique_projected_keys": unique_keys,
        "projection_collision_rate": 1.0 - unique_keys / samples,
        "high_fidelity_median": median_high,
        "high_fidelity_iqr": iqr_high,
        "high_fidelity_iqr_floor": iqr_floor,
        "objective_nonconstant": bool(iqr_high > iqr_floor),
        "spearman_low_high": safe_rank_correlation(low, high, "spearman"),
        "kendall_low_high": safe_rank_correlation(low, high, "kendall"),
        "top10_in_low25_retention": top_retention(low, high),
        "variable_kind_count": kinds,
        "has_conditional_or_cross_constraint": has_structure,
    }


def eligible(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    rho = float(row["spearman_low_high"])
    if not math.isclose(
        float(row["projected_feasible_rate"]),
        THRESHOLDS["projected_feasible_rate"],
        abs_tol=1e-12,
    ):
        reasons.append("projected_feasible_rate")
    if float(row["projection_trigger_rate"]) < THRESHOLDS["minimum_projection_trigger_rate"]:
        reasons.append("projection_trigger_rate")
    if not bool(row["objective_nonconstant"]):
        reasons.append("objective_nonconstant")
    if not (
        math.isfinite(rho)
        and THRESHOLDS["minimum_low_high_spearman"]
        <= rho
        <= THRESHOLDS["maximum_low_high_spearman"]
    ):
        reasons.append("spearman_low_high")
    if int(row["variable_kind_count"]) < THRESHOLDS["minimum_variable_kinds"]:
        reasons.append("variable_kind_count")
    if not bool(row["has_conditional_or_cross_constraint"]):
        reasons.append("structural_constraint")
    return not reasons, reasons


def capped_selection(rows: list[dict[str, Any]]) -> list[str]:
    qualified = [row for row in rows if row["eligible_for_m3"]]
    qualified.sort(key=lambda row: stable_instance_hash(str(row["instance_key"])))
    cap = int(THRESHOLDS["maximum_selected_instances"])
    if len(qualified) <= cap:
        return [str(row["instance_key"]) for row in qualified]

    selected: list[dict[str, Any]] = []
    for family in sorted({str(row["task_family"]) for row in qualified}):
        family_rows = [row for row in qualified if row["task_family"] == family]
        selected.extend(family_rows[:3])
    selected_keys = {str(row["instance_key"]) for row in selected}
    for row in qualified:
        key = str(row["instance_key"])
        if len(selected) >= cap:
            break
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    return [str(row["instance_key"]) for row in selected[:cap]]


def build_manifest(samples: int, smoke: bool) -> dict[str, Any]:
    registry = families.load_registry()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "M0-M2 task-property diagnostics without optimizer outcomes",
        "smoke": smoke,
        "samples_per_instance": samples,
        "fidelities": list(FIDELITIES),
        "thresholds": THRESHOLDS,
        "selection_salt": SELECTION_SALT,
        "family_registry_sha256": registry["instances_sha256"],
        "source_sha256": {
            "run_domain_diagnostics.py": sha256_file(Path(__file__)),
            "benchmark_families.py": sha256_file(BENCHMARK_ROOT / "benchmark_families.py"),
            "run_benchmarks.py": sha256_file(BENCHMARK_ROOT / "run_benchmarks.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=MVS_ROOT / "results")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Diagnostics not executed. Pass --execute explicitly.")
    samples = 40 if args.smoke and args.samples == DEFAULT_SAMPLES else args.samples
    if samples < 20:
        raise SystemExit("--samples must be at least 20.")

    all_specs = families.specs_from_registry()
    specs = all_specs[:3] if args.smoke else all_specs
    run_name = "domain_smoke_v1_1" if args.smoke else "domain_v1_1"
    output = args.output / run_name
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(samples, args.smoke)
    manifest["instance_keys"] = [spec.instance_key for spec in specs]
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise RuntimeError("Existing manifest differs; refusing to mix diagnostic runs.")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    rows: list[dict[str, Any]] = []
    candidate_path = output / "candidate_diagnostics.jsonl"
    with candidate_path.open("w", encoding="utf-8") as candidate_stream:
        for index, spec in enumerate(specs, 1):
            print(f"[{index}/{len(specs)}] {spec.instance_key}", flush=True)
            rows.append(evaluate_instance(spec, samples, candidate_stream))

    for row in rows:
        is_eligible, reasons = eligible(row)
        row["eligible_for_m3"] = is_eligible
        row["exclusion_reasons"] = "|".join(reasons)
        row["selection_hash"] = stable_instance_hash(str(row["instance_key"]))
    selected = capped_selection(rows)
    selected_set = set(selected)
    for row in rows:
        row["selected_for_m3"] = str(row["instance_key"]) in selected_set

    write_csv(output / "instance_diagnostics.csv", rows)
    selection = {
        "status": (
            "smoke_only"
            if args.smoke
            else (
                "eligible"
                if len(selected) >= THRESHOLDS["minimum_selected_instances"]
                else "stop_too_few_eligible_instances"
            )
        ),
        "qualified_count_before_cap": sum(bool(row["eligible_for_m3"]) for row in rows),
        "selected_count": len(selected),
        "selected_instance_keys": selected,
        "excluded": {
            str(row["instance_key"]): str(row["exclusion_reasons"]).split("|")
            for row in rows
            if not row["eligible_for_m3"]
        },
    }
    (output / "selection_decision.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
