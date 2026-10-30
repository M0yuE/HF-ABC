from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

import run_m3 as m3


DEFAULT_INPUT = m3.MVS_ROOT / "results" / "m3_main_v1_20_seeds"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(values: list[float]) -> list[float]:
    count = len(values)
    order = np.argsort(values)
    adjusted = np.zeros(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int = 20260728,
    draws: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True)
    medians = np.median(sampled, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def instance_normalization(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    by_instance: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance_key"]].append(float(row["final_score"]))
    normalization = {}
    for instance, values in by_instance.items():
        array = np.asarray(values, dtype=float)
        best = float(array.min())
        scale = float(np.percentile(array, 75) - np.percentile(array, 25))
        if scale <= 1e-12:
            scale = max(1.0, abs(float(np.median(array))))
        normalization[instance] = (best, scale)
    return normalization


def summarize_instance_methods(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    normalization = instance_normalization(rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["instance_key"], row["method"])].append(row)
    output = []
    for (instance, method), items in sorted(grouped.items()):
        scores = np.asarray([float(row["final_score"]) for row in items])
        best, scale = normalization[instance]
        regrets = (scores - best) / scale
        output.append(
            {
                "instance_key": instance,
                "task_family": items[0]["task_family"],
                "difficulty": items[0]["difficulty"],
                "method": method,
                "n_search_seeds": len(items),
                "median_score": float(np.median(scores)),
                "score_iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                "median_normalized_regret": float(np.median(regrets)),
                "mean_normalized_regret": float(np.mean(regrets)),
                "final_feasible_rate": float(
                    np.mean([row["final_feasible"] == "True" for row in items])
                ),
                "mean_cache_hits": float(
                    np.mean([int(row["cache_hit_count"]) for row in items])
                ),
                "mean_actual_cost": float(
                    np.mean([int(row["actual_cost"]) for row in items])
                ),
                "mean_wall_clock_seconds": float(
                    np.mean([float(row["wall_clock_seconds"]) for row in items])
                ),
            }
        )
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_instance[row["instance_key"]].append(row)
    for items in by_instance.values():
        values = np.asarray([row["median_normalized_regret"] for row in items])
        ranks = stats.rankdata(values, method="average")
        for row, rank in zip(items, ranks):
            row["instance_rank"] = float(rank)
    return output


def summarize_methods(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped[row["method"]].append(row)
    output = []
    for method, items in sorted(grouped.items()):
        regrets = np.asarray([row["median_normalized_regret"] for row in items])
        output.append(
            {
                "method": method,
                "n_instances": len(items),
                "median_instance_normalized_regret": float(np.median(regrets)),
                "mean_instance_normalized_regret": float(np.mean(regrets)),
                "mean_instance_rank": float(
                    np.mean([row["instance_rank"] for row in items])
                ),
                "median_instance_rank": float(
                    np.median([row["instance_rank"] for row in items])
                ),
                "mean_final_feasible_rate": float(
                    np.mean([row["final_feasible_rate"] for row in items])
                ),
                "mean_cache_hits_per_run": float(
                    np.mean([row["mean_cache_hits"] for row in items])
                ),
                "mean_actual_cost": float(
                    np.mean([row["mean_actual_cost"] for row in items])
                ),
                "mean_wall_clock_seconds": float(
                    np.mean([row["mean_wall_clock_seconds"] for row in items])
                ),
            }
        )
    return output


def compare_with_hf(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["instance_key"], row["method"]): float(
            row["median_normalized_regret"]
        )
        for row in instance_rows
    }
    instances = sorted({row["instance_key"] for row in instance_rows})
    comparisons = []
    p_values = []
    for method in m3.METHODS:
        if method == "hf_abc":
            continue
        hf = np.asarray([lookup[(instance, "hf_abc")] for instance in instances])
        baseline = np.asarray([lookup[(instance, method)] for instance in instances])
        differences = hf - baseline
        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        ci_low, ci_high = bootstrap_median_ci(differences)
        comparisons.append(
            {
                "comparison": f"hf_abc vs {method}",
                "n_instances": len(instances),
                "median_difference_hf_minus_other": float(np.median(differences)),
                "bootstrap95_low": ci_low,
                "bootstrap95_high": ci_high,
                "hf_better_instances": int(np.sum(differences < 0)),
                "ties": int(np.sum(np.isclose(differences, 0.0))),
                "hf_worse_instances": int(np.sum(differences > 0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "holm_adjusted_p": 0.0,
            }
        )
        p_values.append(p_value)
    for row, adjusted in zip(comparisons, holm_adjust(p_values)):
        row["holm_adjusted_p"] = adjusted
    return comparisons


def trace_mechanism_summary(
    run_rows: list[dict[str, str]],
    root: Path,
) -> list[dict[str, Any]]:
    aggregates: dict[str, Counter] = defaultdict(Counter)
    generated_sources: dict[str, Counter] = defaultdict(Counter)
    unique_candidate_totals: Counter = Counter()
    for run in run_rows:
        method = run["method"]
        trace_path = root / run["trace_file"]
        first_by_candidate: dict[str, dict[str, Any]] = {}
        with trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                aggregates[method]["requests"] += 1
                aggregates[method]["feasible_requests"] += int(row["feasible"])
                aggregates[method]["projection_triggers"] += int(
                    row["changed_field_count"] > 0
                )
                aggregates[method]["cache_hits"] += int(row["cache_hit"])
                candidate_id = row["candidate_id"]
                first_by_candidate.setdefault(candidate_id, row)
        unique_candidate_totals[method] += len(first_by_candidate)
        for row in first_by_candidate.values():
            generated_sources[method][row["source"]] += 1
    output = []
    for method in sorted(aggregates):
        counts = aggregates[method]
        sources = generated_sources[method]
        requests = counts["requests"]
        generated = unique_candidate_totals[method]
        output.append(
            {
                "method": method,
                "total_requests": requests,
                "unique_generated_candidates": generated,
                "request_feasible_rate": counts["feasible_requests"] / requests,
                "projection_trigger_rate": counts["projection_triggers"] / requests,
                "cache_hit_rate": counts["cache_hits"] / requests,
                "initial_share": sources["initial"] / generated,
                "local_share": sources["local"] / generated,
                "structural_share": sources["struct"] / generated,
                "restart_share": sources["restart"] / generated,
                "abc_neighbor_share": sources["abc_neighbor"] / generated,
                "random_share": sources["random"] / generated,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    args = parser.parse_args()
    root = args.input
    run_rows = read_csv(root / "run_summaries.csv")
    instance_rows = summarize_instance_methods(run_rows)
    method_rows = summarize_methods(instance_rows)
    comparisons = compare_with_hf(instance_rows)
    trace_rows = trace_mechanism_summary(run_rows, root)
    write_csv(root / "instance_method_summary.csv", instance_rows)
    write_csv(root / "method_overall_summary.csv", method_rows)
    write_csv(root / "comparisons_vs_hf.csv", comparisons)
    write_csv(root / "trace_mechanism_summary.csv", trace_rows)
    print(root)


if __name__ == "__main__":
    main()
