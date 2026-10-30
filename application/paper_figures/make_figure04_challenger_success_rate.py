from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
HF_ROOT = ROOT / "HF-ABC"
DATA_CSV = SCRIPT_DIR / "fig04_challenger_success_rate_data.csv"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_PNG = OUT_DIR / "fig04_challenger_success_rate.png"
OUT_SVG = OUT_DIR / "fig04_challenger_success_rate.svg"
OUT_PDF = OUT_DIR / "fig04_challenger_success_rate.pdf"

ORDINARY_ARCHIVE = HF_ROOT / "pf_runs_wq_two_hpo" / "search_seed_2026" / "pf_archive.json"

EGR_ARCHIVES = {
    "elite-local": [
        HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_local_only" / "search_seed_2026" / "pf_archive.json",
        HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_wo_restart" / "search_seed_2026" / "pf_archive.json",
    ],
    "elite-struct": [
        HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_struct_only" / "search_seed_2026" / "pf_archive.json",
        HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_wo_restart" / "search_seed_2026" / "pf_archive.json",
    ],
    "random-restart": [
        HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_restart_only" / "search_seed_2026" / "pf_archive.json",
    ],
}

MODE_BY_SOURCE = {
    "elite-local": "elite_local",
    "elite-struct": "elite_struct",
    "random-restart": "random_restart",
}

SOURCE_ORDER = [
    "elite-local",
    "elite-struct",
    "random-restart",
    "ordinary challenger",
]

CONFIG_FIELDS = [
    "seq_len",
    "d_model",
    "n_heads",
    "e_layers",
    "d_ff",
    "dropout",
    "learning_rate",
    "batch_size",
    "csg_reduction",
]

COLORS = {
    "elite-local": "#2F6F9F",
    "elite-struct": "#4E9F76",
    "random-restart": "#9AA3AD",
    "ordinary challenger": "#C0874A",
}

DISPLAY_LABELS = {
    "elite-local": "elite-\nlocal",
    "elite-struct": "elite-\nstruct",
    "random-restart": "random-\nrestart",
    "ordinary challenger": "ordinary\nchallenger",
}


def read_archive(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def is_valid(record: dict[str, object]) -> bool:
    if record.get("status") != "ok":
        return False
    score = record.get("val_mse")
    return score is not None and math.isfinite(float(score))


def config_key(config: dict[str, object]) -> tuple[object, ...]:
    key: list[object] = []
    for name in CONFIG_FIELDS:
        value = config[name]
        if name in {"dropout", "learning_rate"}:
            key.append(round(float(value), 6))
        else:
            key.append(int(value))
    return tuple(key)


def stage_rank_table(records: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, float]]:
    best_by_key: dict[tuple[object, ...], float] = {}
    for record in records:
        if not is_valid(record):
            continue
        key = config_key(record["config"])
        score = float(record["val_mse"])
        previous = best_by_key.get(key)
        if previous is None or score < previous:
            best_by_key[key] = score

    ordered = sorted(best_by_key.items(), key=lambda item: (item[1], json.dumps(item[0])))
    total = len(ordered)
    top_n = max(1, math.ceil(total * 0.25)) if total else 0
    table: dict[tuple[object, ...], dict[str, float]] = {}
    for rank, (key, score) in enumerate(ordered, start=1):
        table[key] = {
            "rank": float(rank),
            "stage_unique_n": float(total),
            "rank_percentile": float(rank) / float(total) * 100.0 if total else float("nan"),
            "success": float(rank <= top_n),
            "stage_best_mse": float(score),
        }
    return table


def append_stage_rows(
    rows: list[dict[str, object]],
    *,
    source: str,
    archive_name: str,
    stage_records: list[dict[str, object]],
    child_records: list[dict[str, object]],
) -> None:
    rank_table = stage_rank_table(stage_records)
    for record in child_records:
        if not is_valid(record):
            continue
        key = config_key(record["config"])
        rank_info = rank_table.get(key, {})
        rows.append(
            {
                "source": source,
                "archive": archive_name,
                "stage": int(record["stage"]),
                "fidelity": int(record["fidelity"]),
                "val_mse": float(record["val_mse"]),
                "rank": int(rank_info.get("rank", 0)),
                "stage_unique_n": int(rank_info.get("stage_unique_n", 0)),
                "rank_percentile": float(rank_info.get("rank_percentile", float("nan"))),
                "success": int(rank_info.get("success", 0)),
                "success_definition": "entered_top_25_percent_within_same_stage",
                "config_key": json.dumps(key),
            }
        )


def ordinary_challenger_rows() -> list[dict[str, object]]:
    records = [record for record in read_archive(ORDINARY_ARCHIVE) if is_valid(record)]
    by_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_stage[int(record["stage"])].append(record)

    rows: list[dict[str, object]] = []
    seen_keys: set[tuple[object, ...]] = set()
    for stage in sorted(by_stage):
        stage_records = by_stage[stage]
        stage_keys = {config_key(record["config"]) for record in stage_records}
        if stage > 0:
            child_records = [record for record in stage_records if config_key(record["config"]) not in seen_keys]
            append_stage_rows(
                rows,
                source="ordinary challenger",
                archive_name=str(ORDINARY_ARCHIVE.relative_to(ROOT)),
                stage_records=stage_records,
                child_records=child_records,
            )
        seen_keys.update(stage_keys)
    return rows


def egr_source_rows(source: str, paths: list[Path]) -> list[dict[str, object]]:
    mode = MODE_BY_SOURCE[source]
    rows: list[dict[str, object]] = []
    for path in paths:
        records = [record for record in read_archive(path) if is_valid(record)]
        by_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            by_stage[int(record["stage"])].append(record)

        first_child_by_key: dict[tuple[object, ...], dict[str, object]] = {}
        for record in records:
            if not isinstance(record.get("extra"), dict) or record["extra"].get("mode") != mode:
                continue
            key = config_key(record["config"])
            previous = first_child_by_key.get(key)
            if previous is None or int(record["stage"]) < int(previous["stage"]):
                first_child_by_key[key] = record

        child_by_stage: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in first_child_by_key.values():
            child_by_stage[int(record["stage"])].append(record)

        for stage, child_records in child_by_stage.items():
            append_stage_rows(
                rows,
                source=source,
                archive_name=str(path.relative_to(ROOT)),
                stage_records=by_stage[stage],
                child_records=child_records,
            )
    return rows


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, paths in EGR_ARCHIVES.items():
        rows.extend(egr_source_rows(source, paths))
    rows.extend(ordinary_challenger_rows())
    return deduplicate_rows(rows)


def deduplicate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        source_key = (str(row["source"]), str(row["config_key"]))
        previous = by_source_key.get(source_key)
        if previous is None:
            by_source_key[source_key] = row
            continue
        current_sort = (int(row["stage"]), float(row["val_mse"]))
        previous_sort = (int(previous["stage"]), float(previous["val_mse"]))
        if current_sort < previous_sort:
            by_source_key[source_key] = row
    return [by_source_key[key] for key in sorted(by_source_key)]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        subset = [row for row in rows if row["source"] == source]
        generated = len(subset)
        successes = sum(int(row["success"]) for row in subset)
        rate = successes / generated if generated else float("nan")
        summary_rows.append(
            {
                "source": source,
                "generated_children": generated,
                "successful_children": successes,
                "success_rate": rate,
            }
        )
    return summary_rows


def write_data_csv(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DATA_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["summary"])
        writer.writerow(["source", "generated_children", "successful_children", "success_rate"])
        for row in summary_rows:
            writer.writerow(
                [
                    row["source"],
                    row["generated_children"],
                    row["successful_children"],
                    f"{float(row['success_rate']):.10f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["child_records"])
        detail_fields = [
            "source",
            "archive",
            "stage",
            "fidelity",
            "val_mse",
            "rank",
            "stage_unique_n",
            "rank_percentile",
            "success",
            "success_definition",
            "config_key",
        ]
        detail_writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        detail_writer.writeheader()
        detail_writer.writerows(rows)


def plot_success_rates(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.85,
            "legend.frameon": False,
        }
    )

    sources = [str(row["source"]) for row in summary_rows]
    rates = np.array([float(row["success_rate"]) * 100.0 for row in summary_rows], dtype=float)
    successes = [int(row["successful_children"]) for row in summary_rows]
    generated = [int(row["generated_children"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [0.9, 1.15]})
    ax = axes[0]
    x = np.arange(len(sources))
    bars = ax.bar(
        x,
        rates,
        width=0.62,
        color=[COLORS[source] for source in sources],
        edgecolor="#29313A",
        linewidth=0.55,
        zorder=3,
    )

    for bar, rate, ok, total in zip(bars, rates, successes, generated):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.15,
            f"{rate:.1f}%\n{ok}/{total}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.12,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[source] for source in sources])
    ax.set_ylabel("Success rate (%)", fontsize=8.4)
    ax.set_ylim(0, max(24.0, float(np.nanmax(rates)) + 8.0))
    ax.grid(axis="y", color="#E8EDF3", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.2, width=0.7, labelsize=7.6)
    ax.set_title("Top-25% hit rate", fontsize=9.2, fontweight="semibold", pad=5)
    ax.text(
        0.01,
        1.02,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )

    ax_rank = axes[1]
    rng = np.random.default_rng(2026)
    ax_rank.axhspan(0, 25, color="#DCEFE6", alpha=0.65, zorder=0)
    ax_rank.axhline(25, color="#6FA88A", lw=0.95, ls=(0, (3, 2)), zorder=1)
    for idx, source in enumerate(sources):
        subset = [row for row in rows if row["source"] == source]
        y = np.array([float(row["rank_percentile"]) for row in subset], dtype=float)
        jitter = rng.uniform(-0.14, 0.14, size=len(y))
        success_mask = np.array([int(row["success"]) == 1 for row in subset], dtype=bool)
        ax_rank.scatter(
            np.full_like(y, idx, dtype=float) + jitter,
            y,
            s=np.where(success_mask, 34, 24),
            facecolor=COLORS[source],
            edgecolor=np.where(success_mask, "#1F252B", "white"),
            linewidth=np.where(success_mask, 0.65, 0.45),
            alpha=np.where(success_mask, 0.96, 0.58),
            zorder=3,
        )
        if len(y):
            median = float(np.nanmedian(y))
            ax_rank.plot([idx - 0.22, idx + 0.22], [median, median], color="#242B33", lw=1.35, zorder=4)

    ax_rank.set_xticks(x)
    ax_rank.set_xticklabels([DISPLAY_LABELS[source] for source in sources])
    ax_rank.set_ylim(102, -2)
    ax_rank.set_yticks([0, 25, 50, 75, 100])
    ax_rank.set_ylabel("Rank percentile\n(lower is better)", fontsize=8.4)
    ax_rank.grid(axis="y", color="#E8EDF3", linewidth=0.65)
    ax_rank.set_axisbelow(True)
    ax_rank.tick_params(direction="out", length=3.2, width=0.7, labelsize=7.6)
    ax_rank.set_title("Generated-child rank distribution", fontsize=9.2, fontweight="semibold", pad=5)
    ax_rank.text(
        0.01,
        1.02,
        "b",
        transform=ax_rank.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_rank.text(
        0.99,
        0.04,
        "Shaded: top 25%",
        transform=ax_rank.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#424B57",
    )

    fig.tight_layout(w_pad=1.1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    if not rows:
        raise RuntimeError("No challenger records were found.")
    summary_rows = summarize(rows)
    write_data_csv(rows, summary_rows)
    plot_success_rates(rows, summary_rows)

    for row in summary_rows:
        print(
            f"{row['source']}: "
            f"{int(row['successful_children'])}/{int(row['generated_children'])} "
            f"({float(row['success_rate']) * 100:.2f}%)"
        )
    print(OUT_PNG.resolve())


if __name__ == "__main__":
    main()
