from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from egta.search_space import SearchSpace
except Exception:  # pragma: no cover - the local fallback is intentionally self-contained.
    SearchSpace = None  # type: ignore[assignment]

EXISTING_SEARCH_SPACE = SearchSpace() if SearchSpace is not None else None


CONFIG_FIELDS = [
    "seq_len",
    "d_model",
    "n_heads",
    "e_layers",
    "d_ff",
    "dropout",
    "learning_rate",
    "batch_size",
    "model_type",
    "csg_reduction",
]

SEQ_LEN_VALUES = (48, 96, 192, 336, 512)
D_MODEL_VALUES = (128, 256, 384, 512, 768, 1024)
N_HEADS_VALUES = (2, 4, 8, 16, 32)
E_LAYERS_VALUES = (1, 2, 3, 4, 6, 8)
D_FF_VALUES = (128, 256, 512, 1024, 2048, 4096)
BATCH_SIZE_VALUES = (64, 128, 256, 512)
CSG_REDUCTION_VALUES = (4, 8)
MODEL_TYPES = ("iTransformer", "iTransformer-CSG")

DISCRETE_VALUES: Mapping[str, Sequence[int]] = {
    "seq_len": SEQ_LEN_VALUES,
    "d_model": D_MODEL_VALUES,
    "n_heads": N_HEADS_VALUES,
    "e_layers": E_LAYERS_VALUES,
    "d_ff": D_FF_VALUES,
    "batch_size": BATCH_SIZE_VALUES,
}

NUMERIC_CLIP_RANGES: Mapping[str, Tuple[float, float]] = {
    "seq_len": (min(SEQ_LEN_VALUES), max(SEQ_LEN_VALUES)),
    "d_model": (min(D_MODEL_VALUES), max(D_MODEL_VALUES)),
    "n_heads": (min(N_HEADS_VALUES), max(N_HEADS_VALUES)),
    "e_layers": (min(E_LAYERS_VALUES), max(E_LAYERS_VALUES)),
    "d_ff": (min(D_FF_VALUES), max(D_FF_VALUES)),
    "dropout": (0.05, 0.20),
    "learning_rate": (8e-5, 6e-4),
    "batch_size": (min(BATCH_SIZE_VALUES), max(BATCH_SIZE_VALUES)),
    "csg_reduction": (min(CSG_REDUCTION_VALUES), max(CSG_REDUCTION_VALUES)),
}

RAW_METHOD = "Raw generator"
CLIP_METHOD = "Clip/Round baseline"
PROJECTION_METHOD = "Full Projection"
METHODS = (RAW_METHOD, CLIP_METHOD, PROJECTION_METHOD)
FULLY_RELAXED_SCENARIO = "fully_relaxed"
NEAR_FEASIBLE_SCENARIO = "near_feasible"
SCENARIOS = (FULLY_RELAXED_SCENARIO, NEAR_FEASIBLE_SCENARIO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline feasibility-projection diagnostic test. Generates relaxed or near-feasible "
            "candidates, compares raw, clip/round, and full projection validity, and writes CSV summaries."
        )
    )
    parser.add_argument("--num-candidates", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--scenarios",
        choices=SCENARIOS,
        nargs="+",
        default=list(SCENARIOS),
        help=(
            "fully_relaxed samples from a deliberately wide stress space; near_feasible starts "
            "from legal configs and applies local optimizer-like perturbations."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/projection_stress_test"))
    parser.add_argument("--no-plot", action="store_true", help="Do not write the optional SVG/PNG figure.")
    args = parser.parse_args()
    if args.num_candidates <= 0:
        parser.error("--num-candidates must be positive")
    if not args.seeds:
        parser.error("--seeds must contain at least one integer")
    return args


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def nearest_value(value: Any, legal_values: Sequence[int]) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return int(legal_values[0])
    return int(min(legal_values, key=lambda candidate: (abs(float(candidate) - numeric_value), candidate)))


def log_clip(value: Any, low: float, high: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = low
    if numeric_value <= 0:
        return low
    log_value = clamp(math.log(numeric_value), math.log(low), math.log(high))
    return math.exp(log_value)


def is_csg_model(model_type: Any) -> bool:
    return model_type_key(model_type) == "iTransformer-CSG"


def model_type_key(model_type: Any) -> str:
    return str(model_type).replace("_", "-")


def normalize_model_type(model_type: Any) -> str:
    normalized = model_type_key(model_type)
    if normalized in MODEL_TYPES:
        return normalized
    return "iTransformer"


def is_inactive_csg(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "inactive")


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def generate_fully_relaxed_candidate(rng: random.Random) -> Dict[str, Any]:
    return {
        "seq_len": rng.randint(24, 600),
        "d_model": rng.randint(64, 1100),
        "n_heads": rng.randint(1, 40),
        "e_layers": rng.randint(1, 10),
        "d_ff": rng.randint(64, 5000),
        "dropout": rng.uniform(0.0, 0.5),
        "learning_rate": sample_log_uniform(rng, 1e-5, 2e-3),
        "batch_size": rng.randint(16, 600),
        "model_type": rng.choice(MODEL_TYPES),
        "csg_reduction": rng.randint(2, 16),
    }


def sample_legal_config(rng: random.Random) -> Dict[str, Any]:
    n_heads = rng.choice(N_HEADS_VALUES)
    d_model = rng.choice([value for value in D_MODEL_VALUES if value % n_heads == 0])
    model_type = rng.choice(MODEL_TYPES)
    return {
        "seq_len": rng.choice(SEQ_LEN_VALUES),
        "d_model": d_model,
        "n_heads": n_heads,
        "e_layers": rng.choice(E_LAYERS_VALUES),
        "d_ff": rng.choice(D_FF_VALUES),
        "dropout": rng.uniform(0.05, 0.20),
        "learning_rate": sample_log_uniform(rng, 8e-5, 6e-4),
        "batch_size": rng.choice(BATCH_SIZE_VALUES),
        "model_type": model_type,
        "csg_reduction": rng.choice(CSG_REDUCTION_VALUES) if model_type == "iTransformer-CSG" else None,
    }


def soft_discrete_perturb(value: int, rng: random.Random) -> float:
    offset = rng.uniform(-0.45, 0.45)
    if abs(offset) < 0.02:
        offset = 0.02 if offset >= 0 else -0.02
    return float(value) + offset


def hard_discrete_perturb(name: str, value: int, rng: random.Random) -> int:
    low, high = NUMERIC_CLIP_RANGES[name]
    if name == "e_layers":
        return rng.choice([5, 7])

    offsets = {
        "seq_len": (-24, -12, 12, 24),
        "d_model": (-96, -64, 64, 96),
        "n_heads": (-3, -1, 1, 3),
        "d_ff": (-96, -64, 64, 96),
        "batch_size": (-48, -32, 32, 48),
    }[name]
    legal = set(DISCRETE_VALUES[name])
    candidates = [
        int(round(value + offset))
        for offset in offsets
        if low <= value + offset <= high and int(round(value + offset)) not in legal
    ]
    if candidates:
        return rng.choice(candidates)

    in_range_illegal = [candidate for candidate in range(int(low), int(high) + 1) if candidate not in legal]
    return rng.choice(in_range_illegal)


def locally_perturb_discrete(name: str, value: int, rng: random.Random) -> int | float:
    draw = rng.random()
    if draw < 0.12:
        return soft_discrete_perturb(value, rng)
    if draw < 0.18:
        return hard_discrete_perturb(name, value, rng)
    return value


def locally_perturb_dropout(value: float, rng: random.Random) -> float:
    if rng.random() >= 0.10:
        return value
    if rng.random() < 0.5:
        return rng.uniform(0.0, 0.049)
    return rng.uniform(0.201, 0.35)


def locally_perturb_learning_rate(value: float, rng: random.Random) -> float:
    if rng.random() >= 0.10:
        return value
    if rng.random() < 0.5:
        return sample_log_uniform(rng, 2e-5, 7.9e-5)
    return sample_log_uniform(rng, 6.1e-4, 1.5e-3)


def locally_perturb_csg(model_type: str, value: Any, rng: random.Random) -> Any:
    if model_type == "iTransformer-CSG":
        if rng.random() < 0.08:
            return rng.choice([None, 3, 6, 10])
        return value

    if rng.random() < 0.12:
        return rng.choice([4, 6, 8])
    return None


def generate_near_feasible_candidate(rng: random.Random) -> Dict[str, Any]:
    cfg = sample_legal_config(rng)
    for name in ("seq_len", "d_model", "n_heads", "e_layers", "d_ff", "batch_size"):
        cfg[name] = locally_perturb_discrete(name, int(cfg[name]), rng)
    cfg["dropout"] = locally_perturb_dropout(float(cfg["dropout"]), rng)
    cfg["learning_rate"] = locally_perturb_learning_rate(float(cfg["learning_rate"]), rng)
    cfg["csg_reduction"] = locally_perturb_csg(str(cfg["model_type"]), cfg["csg_reduction"], rng)
    return cfg


def generate_raw_candidate(rng: random.Random, scenario: str) -> Dict[str, Any]:
    if scenario == FULLY_RELAXED_SCENARIO:
        return generate_fully_relaxed_candidate(rng)
    if scenario == NEAR_FEASIBLE_SCENARIO:
        return generate_near_feasible_candidate(rng)
    raise ValueError(f"Unsupported scenario: {scenario}")


def clip_round_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_config)
    for name in ("seq_len", "d_model", "n_heads", "e_layers", "d_ff", "batch_size"):
        low, high = NUMERIC_CLIP_RANGES[name]
        cfg[name] = int(round(clamp(float(cfg[name]), low, high)))
    if not is_inactive_csg(cfg.get("csg_reduction")):
        low, high = NUMERIC_CLIP_RANGES["csg_reduction"]
        cfg["csg_reduction"] = int(round(clamp(float(cfg["csg_reduction"]), low, high)))
    cfg["dropout"] = round(clamp(float(cfg["dropout"]), *NUMERIC_CLIP_RANGES["dropout"]), 6)
    cfg["learning_rate"] = log_clip(cfg["learning_rate"], *NUMERIC_CLIP_RANGES["learning_rate"])
    cfg["model_type"] = normalize_model_type(cfg.get("model_type"))
    return cfg


def _project_with_existing_search_space(raw_config: Mapping[str, Any]) -> Dict[str, Any] | None:
    if EXISTING_SEARCH_SPACE is None:
        return None

    try:
        model_name = "iTransformer_CSG" if is_csg_model(raw_config.get("model_type")) else "iTransformer"
        projected = EXISTING_SEARCH_SPACE.project_config(dict(raw_config), model_name=model_name)
    except Exception:
        return None

    projected["model_type"] = normalize_model_type(raw_config.get("model_type"))
    if not is_csg_model(projected["model_type"]):
        projected["csg_reduction"] = None
    else:
        projected["csg_reduction"] = nearest_value(projected.get("csg_reduction"), CSG_REDUCTION_VALUES)
    projected["dropout"] = round(clamp(float(projected["dropout"]), 0.05, 0.20), 6)
    projected["learning_rate"] = log_clip(projected["learning_rate"], 8e-5, 6e-4)
    return {name: projected.get(name) for name in CONFIG_FIELDS}


def _project_locally(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_config)
    for name, values in DISCRETE_VALUES.items():
        cfg[name] = nearest_value(cfg.get(name), values)

    cfg["dropout"] = round(clamp(float(cfg.get("dropout", 0.10)), 0.05, 0.20), 6)
    cfg["learning_rate"] = log_clip(cfg.get("learning_rate", 2e-4), 8e-5, 6e-4)
    cfg["model_type"] = normalize_model_type(cfg.get("model_type"))

    n_heads = int(cfg["n_heads"])
    d_model = int(cfg["d_model"])
    if d_model % n_heads != 0:
        legal_models = [value for value in D_MODEL_VALUES if value % n_heads == 0]
        cfg["d_model"] = nearest_value(d_model, legal_models or D_MODEL_VALUES)

    if is_csg_model(cfg["model_type"]):
        cfg["csg_reduction"] = nearest_value(cfg.get("csg_reduction"), CSG_REDUCTION_VALUES)
    else:
        cfg["csg_reduction"] = None

    return {name: cfg.get(name) for name in CONFIG_FIELDS}


def full_projection(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    projected = _project_with_existing_search_space(raw_config)
    if projected is not None:
        return projected
    return _project_locally(raw_config)


def validate_config(cfg: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    violations: List[str] = []

    for name, values in DISCRETE_VALUES.items():
        if cfg.get(name) not in set(values):
            violations.append(f"{name}_not_in_legal_set")

    try:
        d_model = int(cfg.get("d_model"))
        n_heads = int(cfg.get("n_heads"))
        if n_heads == 0 or d_model % n_heads != 0:
            violations.append("d_model_not_divisible_by_n_heads")
    except (TypeError, ValueError):
        violations.append("d_model_not_divisible_by_n_heads")

    try:
        dropout = float(cfg.get("dropout"))
        if not (0.05 <= dropout <= 0.20):
            violations.append("dropout_out_of_range")
    except (TypeError, ValueError):
        violations.append("dropout_out_of_range")

    try:
        learning_rate = float(cfg.get("learning_rate"))
        if not (8e-5 <= learning_rate <= 6e-4):
            violations.append("learning_rate_out_of_range")
    except (TypeError, ValueError):
        violations.append("learning_rate_out_of_range")

    model_type = model_type_key(cfg.get("model_type"))
    if model_type not in MODEL_TYPES:
        violations.append("model_type_invalid")
    elif is_csg_model(model_type):
        if cfg.get("csg_reduction") not in set(CSG_REDUCTION_VALUES):
            violations.append("csg_reduction_not_in_legal_set")
    elif not is_inactive_csg(cfg.get("csg_reduction")):
        violations.append("csg_reduction_active_for_non_csg")

    return len(violations) == 0, violations


def json_cell(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def config_key(cfg: Mapping[str, Any]) -> str:
    return json_cell({name: canonical_value(cfg.get(name)) for name in CONFIG_FIELDS})


def changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    return [name for name in CONFIG_FIELDS if canonical_value(before.get(name)) != canonical_value(after.get(name))]


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_seed(
    scenario: str, seed: int, num_candidates: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    method_stats: Dict[str, MutableMapping[str, Any]] = {
        method: {
            "valid": 0,
            "unique_keys": set(),
            "violations": Counter(),
            "repaired": 0,
            "unrepairable": 0,
            "changed_counts": [],
        }
        for method in METHODS
    }
    example_rows: List[Dict[str, Any]] = []

    for candidate_id in range(num_candidates):
        raw_config = generate_raw_candidate(rng, scenario)
        clip_config = clip_round_config(raw_config)
        projected_config = full_projection(raw_config)

        raw_valid, raw_violations = validate_config(raw_config)
        clip_valid, clip_violations = validate_config(clip_config)
        projected_valid, projected_violations = validate_config(projected_config)

        variants = {
            RAW_METHOD: (raw_config, raw_valid, raw_violations, []),
            CLIP_METHOD: (clip_config, clip_valid, clip_violations, changed_fields(raw_config, clip_config)),
            PROJECTION_METHOD: (
                projected_config,
                projected_valid,
                projected_violations,
                changed_fields(raw_config, projected_config),
            ),
        }

        for method, (cfg, valid, violations, changed) in variants.items():
            stats = method_stats[method]
            stats["valid"] += int(valid)
            stats["unique_keys"].add(config_key(cfg))
            stats["violations"].update(violations)
            stats["changed_counts"].append(len(changed))
            if not raw_valid and valid:
                stats["repaired"] += 1
            if not raw_valid and not valid:
                stats["unrepairable"] += 1

        example_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "candidate_id": candidate_id,
                "raw_config": json_cell(raw_config),
                "clip_round_config": json_cell(clip_config),
                "projected_config": json_cell(projected_config),
                "raw_violations": json_cell(raw_violations),
                "clip_round_violations": json_cell(clip_violations),
                "projected_violations": json_cell(projected_violations),
                "changed_fields": json_cell(variants[PROJECTION_METHOD][3]),
            }
        )

    raw_invalid_count = num_candidates - int(method_stats[RAW_METHOD]["valid"])
    summary_rows: List[Dict[str, Any]] = []
    violation_rows: List[Dict[str, Any]] = []

    for method in METHODS:
        stats = method_stats[method]
        num_valid = int(stats["valid"])
        num_invalid = num_candidates - num_valid
        num_unique = len(stats["unique_keys"])
        changed_counts = list(stats["changed_counts"])
        summary_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "method": method,
                "num_candidates": num_candidates,
                "num_valid": num_valid,
                "valid_rate": ratio(num_valid, num_candidates),
                "invalid_rate": ratio(num_invalid, num_candidates),
                "num_unique": num_unique,
                "duplicate_rate": ratio(num_candidates - num_unique, num_candidates),
                "num_repaired": int(stats["repaired"]),
                "repair_rate": ratio(int(stats["repaired"]), raw_invalid_count),
                "num_unrepairable": int(stats["unrepairable"]),
                "unrepairable_rate": ratio(int(stats["unrepairable"]), raw_invalid_count),
                "mean_changed_fields": round(sum(changed_counts) / len(changed_counts), 6) if changed_counts else 0.0,
                "estimated_wasted_candidates": num_invalid,
                "estimated_wasted_ce_1epoch": num_invalid * 1,
                "estimated_wasted_ce_10epoch": num_invalid * 10,
            }
        )

        for violation_type, count in sorted(stats["violations"].items()):
            violation_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "violation_type": violation_type,
                    "count": int(count),
                    "ratio": ratio(int(count), num_candidates),
                }
            )
        if not stats["violations"]:
            violation_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "violation_type": "none",
                    "count": 0,
                    "ratio": 0.0,
                }
            )

    return summary_rows, violation_rows, example_rows


def compact_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def top_violations_for_method(
    violation_rows: Sequence[Mapping[str, Any]], scenario: str, method: str, top_k: int = 3
) -> str:
    counter: Counter[str] = Counter()
    for row in violation_rows:
        if row["scenario"] == scenario and row["method"] == method and row["violation_type"] != "none":
            counter[str(row["violation_type"])] += int(row["count"])
    if not counter:
        return "none"
    return ", ".join(f"{name}({count})" for name, count in counter.most_common(top_k))


def print_console_summary(summary_rows: Sequence[Mapping[str, Any]], violation_rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["scenario"]), str(row["method"]))].append(row)

    headers = ["scenario", "method", "valid_rate", "invalid_rate", "repair_rate", "duplicate_rate", "top violations"]
    table_rows: List[List[str]] = []
    for scenario in SCENARIOS:
        if not any(row["scenario"] == scenario for row in summary_rows):
            continue
        for method in METHODS:
            rows = grouped[(scenario, method)]
            mean_valid = sum(float(row["valid_rate"]) for row in rows) / len(rows)
            mean_invalid = sum(float(row["invalid_rate"]) for row in rows) / len(rows)
            mean_repair = sum(float(row["repair_rate"]) for row in rows) / len(rows)
            mean_duplicate = sum(float(row["duplicate_rate"]) for row in rows) / len(rows)
            table_rows.append(
                [
                    scenario,
                    method,
                    compact_rate(mean_valid),
                    compact_rate(mean_invalid),
                    compact_rate(mean_repair),
                    compact_rate(mean_duplicate),
                    top_violations_for_method(violation_rows, scenario, method),
                ]
            )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]
    print(" | ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(row[col].ljust(widths[col]) for col in range(len(headers))))


def aggregate_for_paper(
    summary_rows: Sequence[Mapping[str, Any]], violation_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_key[(str(row["scenario"]), str(row["method"]))].append(row)

    raw_invalid_by_scenario: Dict[str, int] = {}
    for scenario in SCENARIOS:
        raw_rows = rows_by_key.get((scenario, RAW_METHOD), [])
        raw_invalid_by_scenario[scenario] = sum(
            int(row["num_candidates"]) - int(row["num_valid"]) for row in raw_rows
        )

    paper_rows: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            rows = rows_by_key.get((scenario, method), [])
            if not rows:
                continue
            total_candidates = sum(int(row["num_candidates"]) for row in rows)
            total_valid = sum(int(row["num_valid"]) for row in rows)
            total_unique = sum(int(row["num_unique"]) for row in rows)
            total_repaired = sum(int(row["num_repaired"]) for row in rows)
            total_unrepairable = sum(int(row["num_unrepairable"]) for row in rows)
            total_wasted_1epoch = sum(int(row["estimated_wasted_ce_1epoch"]) for row in rows)
            total_wasted_10epoch = sum(int(row["estimated_wasted_ce_10epoch"]) for row in rows)
            weighted_changed = sum(
                float(row["mean_changed_fields"]) * int(row["num_candidates"]) for row in rows
            )
            raw_invalid_total = raw_invalid_by_scenario[scenario]
            paper_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "num_candidates": total_candidates,
                    "num_valid": total_valid,
                    "valid_rate": ratio(total_valid, total_candidates),
                    "invalid_rate": ratio(total_candidates - total_valid, total_candidates),
                    "num_unique": total_unique,
                    "duplicate_rate": ratio(total_candidates - total_unique, total_candidates),
                    "num_repaired": total_repaired,
                    "repair_rate": ratio(total_repaired, raw_invalid_total),
                    "num_unrepairable": total_unrepairable,
                    "unrepairable_rate": ratio(total_unrepairable, raw_invalid_total),
                    "mean_changed_fields": round(weighted_changed / total_candidates, 6),
                    "estimated_wasted_ce_1epoch": total_wasted_1epoch,
                    "estimated_wasted_ce_10epoch": total_wasted_10epoch,
                    "top_violations": top_violations_for_method(violation_rows, scenario, method),
                }
            )
    return paper_rows


def write_optional_figure(paper_rows: Sequence[Mapping[str, Any]], violation_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping figure export because matplotlib is unavailable: {exc}")
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["legend.frameon"] = False

    colors = {
        RAW_METHOD: "#9CA3AF",
        CLIP_METHOD: "#60A5FA",
        PROJECTION_METHOD: "#34A853",
    }
    scenario_titles = {
        FULLY_RELAXED_SCENARIO: "Fully relaxed stress space",
        NEAR_FEASIBLE_SCENARIO: "Near-feasible perturbation space",
    }
    paper_by_key = {(row["scenario"], row["method"]): row for row in paper_rows}
    required_keys = [(scenario, method) for scenario in SCENARIOS for method in METHODS]
    if any(key not in paper_by_key for key in required_keys):
        print("Skipping figure export because both default scenarios are required for the multi-panel figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), constrained_layout=True)
    x_positions = range(len(METHODS))

    for scenario_idx, scenario in enumerate(SCENARIOS):
        offset = -0.18 if scenario_idx == 0 else 0.18
        valid_rates = [
            float(paper_by_key[(scenario, method)]["valid_rate"]) * 100
            for method in METHODS
            if (scenario, method) in paper_by_key
        ]
        axes[0].bar(
            [x + offset for x in x_positions],
            valid_rates,
            width=0.34,
            label=scenario_titles[scenario],
            color=["#D1D5DB", "#93C5FD", "#86EFAC"] if scenario_idx == 0 else ["#6B7280", "#2563EB", "#16A34A"],
        )

    axes[0].set_title("a  Valid candidates")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels(["Raw", "Clip/Round", "Projection"], rotation=25, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper left", fontsize=7)

    near_rows = [paper_by_key[(NEAR_FEASIBLE_SCENARIO, method)] for method in METHODS]
    repair_rates = [float(row["repair_rate"]) * 100 for row in near_rows]
    axes[1].bar(
        list(x_positions),
        repair_rates,
        color=[colors[method] for method in METHODS],
        width=0.55,
    )
    axes[1].set_title("b  Repair under near-feasible perturbation")
    axes[1].set_ylabel("Repair rate (%)")
    axes[1].set_xticks(list(x_positions))
    axes[1].set_xticklabels(["Raw", "Clip/Round", "Projection"], rotation=25, ha="right")
    axes[1].set_ylim(0, 105)

    violation_counter: Dict[str, Counter[str]] = {RAW_METHOD: Counter(), CLIP_METHOD: Counter()}
    for row in violation_rows:
        if row["scenario"] != NEAR_FEASIBLE_SCENARIO or row["violation_type"] == "none":
            continue
        if row["method"] in violation_counter:
            violation_counter[str(row["method"])][str(row["violation_type"])] += int(row["count"])
    top_types = [
        name
        for name, _ in (violation_counter[RAW_METHOD] + violation_counter[CLIP_METHOD]).most_common(5)
    ]
    near_total = int(paper_by_key[(NEAR_FEASIBLE_SCENARIO, RAW_METHOD)]["num_candidates"])
    y_positions = range(len(top_types))
    raw_rates = [violation_counter[RAW_METHOD][name] / near_total * 100 for name in top_types]
    clip_rates = [violation_counter[CLIP_METHOD][name] / near_total * 100 for name in top_types]
    axes[2].barh([y + 0.18 for y in y_positions], raw_rates, height=0.34, color=colors[RAW_METHOD], label="Raw")
    axes[2].barh([y - 0.18 for y in y_positions], clip_rates, height=0.34, color=colors[CLIP_METHOD], label="Clip/Round")
    axes[2].set_title("c  Main violations near feasible")
    axes[2].set_xlabel("Violation count / candidates (%)")
    axes[2].set_yticks(list(y_positions))
    short_violation_labels = {
        "batch_size_not_in_legal_set": "batch_size\nnot in set",
        "n_heads_not_in_legal_set": "n_heads\nnot in set",
        "e_layers_not_in_legal_set": "e_layers\nnot in set",
        "d_ff_not_in_legal_set": "d_ff\nnot in set",
        "seq_len_not_in_legal_set": "seq_len\nnot in set",
        "d_model_not_in_legal_set": "d_model\nnot in set",
        "d_model_not_divisible_by_n_heads": "d_model %\nn_heads != 0",
        "dropout_out_of_range": "dropout\nout of range",
        "learning_rate_out_of_range": "learning rate\nout of range",
        "csg_reduction_active_for_non_csg": "CSG variable\nactive",
        "csg_reduction_not_in_legal_set": "CSG reduction\nnot in set",
    }
    axes[2].set_yticklabels([short_violation_labels.get(name, name) for name in top_types], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), ncol=2, fontsize=7)

    for axis in axes:
        axis.grid(axis="x" if axis is axes[2] else "y", color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)

    fig.savefig(out_dir / "projection_stress_test_figure.svg", bbox_inches="tight")
    fig.savefig(out_dir / "projection_stress_test_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: List[Dict[str, Any]] = []
    all_violation_rows: List[Dict[str, Any]] = []
    all_example_rows: List[Dict[str, Any]] = []

    for scenario in args.scenarios:
        for seed in args.seeds:
            summary_rows, violation_rows, example_rows = summarize_seed(scenario, seed, args.num_candidates)
            all_summary_rows.extend(summary_rows)
            all_violation_rows.extend(violation_rows)
            all_example_rows.extend(example_rows)

    paper_rows = aggregate_for_paper(all_summary_rows, all_violation_rows)

    write_csv(
        out_dir / "projection_stress_summary.csv",
        [
            "scenario",
            "seed",
            "method",
            "num_candidates",
            "num_valid",
            "valid_rate",
            "invalid_rate",
            "num_unique",
            "duplicate_rate",
            "num_repaired",
            "repair_rate",
            "num_unrepairable",
            "unrepairable_rate",
            "mean_changed_fields",
            "estimated_wasted_candidates",
            "estimated_wasted_ce_1epoch",
            "estimated_wasted_ce_10epoch",
        ],
        all_summary_rows,
    )
    write_csv(
        out_dir / "projection_violation_breakdown.csv",
        ["scenario", "seed", "method", "violation_type", "count", "ratio"],
        all_violation_rows,
    )
    write_csv(
        out_dir / "projection_examples.csv",
        [
            "scenario",
            "seed",
            "candidate_id",
            "raw_config",
            "clip_round_config",
            "projected_config",
            "raw_violations",
            "clip_round_violations",
            "projected_violations",
            "changed_fields",
        ],
        all_example_rows,
    )
    write_csv(
        out_dir / "projection_paper_table.csv",
        [
            "scenario",
            "method",
            "num_candidates",
            "num_valid",
            "valid_rate",
            "invalid_rate",
            "num_unique",
            "duplicate_rate",
            "num_repaired",
            "repair_rate",
            "num_unrepairable",
            "unrepairable_rate",
            "mean_changed_fields",
            "estimated_wasted_ce_1epoch",
            "estimated_wasted_ce_10epoch",
            "top_violations",
        ],
        paper_rows,
    )
    if not args.no_plot:
        write_optional_figure(paper_rows, all_violation_rows, out_dir)

    print_console_summary(all_summary_rows, all_violation_rows)
    print(f"\nWrote CSV files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
