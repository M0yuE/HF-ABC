from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figure09_overall_prediction import (
    METHOD_COLORS,
    METHOD_ORDER,
    finite,
    lookup,
    to_float,
)

DATASET_ORDER = ["N2O", "Agtrup"]
HORIZON_ORDER = [60, 120, 240]

OUT_DIR = Path(__file__).parent / "outputs"
DATA_CSV = OUT_DIR / "fig09_overall_prediction_performance_data.csv"

METRICS = [
    ("MAE", "mae_mean", "mae_std"),
    ("Key-MAE", "key_mae_mean", "key_mae_std"),
]

STYLE = {
    "figsize": (8.1, 3.55),
    "dpi": 450,
    "bar_height": 0.52,
    "bar_edge_width": 0.7,
    "capsize": 2.2,
    "error_linewidth": 0.9,
    "error_capthick": 0.9,
    "title_fontsize": 8.0,
    "tick_fontsize": 7.0,
    "method_label_fontsize": 4.8,
    "ylabel_fontsize": 8.5,
    "subfigure_label_fontsize": 8.5,
    "legend_fontsize": 6.4,
    "grid_color": "#E8EDF4",
    "tight_layout_rect": (0.035, 0.025, 0.995, 0.88),
    "w_pad": 0.7,
    "h_pad": 0.95,
    "xlim_pad_left": 0.18,
    "xlim_pad_right": 0.08,
    "xlim_min_span": 0.002,
}

NUMERIC_KEYS = [
    "mse_mean", "mse_std", "mae_mean", "mae_std",
    "key_mse_mean", "key_mse_std", "key_mae_mean", "key_mae_std",
]


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rec: dict[str, object] = {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "method": row["method"],
            }
            for key in NUMERIC_KEYS:
                rec[key] = to_float(row.get(key, "nan"))
            rows.append(rec)
    return rows


def candidate_methods(rows: list[dict[str, object]]) -> list[str]:
    methods = sorted(
        {r["method"] for r in rows if r["method"] in METHOD_ORDER},
        key=lambda m: METHOD_ORDER.index(m),
    )
    return methods or list(METHOD_ORDER)


def sorted_methods(
    rows: list[dict[str, object]],
    dataset: str,
    horizon: int,
    mean_key: str,
) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for method in candidate_methods(rows):
        rec = lookup(rows, dataset, horizon, method)
        if rec is None:
            continue
        val = float(rec[mean_key])
        if math.isnan(val):
            continue
        pairs.append((method, val))
    pairs.sort(key=lambda mv: mv[1], reverse=True)
    return [m for m, _ in pairs]


def active_facets(rows: list[dict[str, object]]) -> list[tuple[str, int]]:
    pairs = {(r["dataset"], int(r["horizon"])) for r in rows}
    return sorted(
        pairs,
        key=lambda dh: (
            DATASET_ORDER.index(dh[0]) if dh[0] in DATASET_ORDER else 999,
            HORIZON_ORDER.index(dh[1]) if dh[1] in HORIZON_ORDER else dh[1],
        ),
    )


def subfigure_label(row_idx: int, col_idx: int, ncols: int) -> str:
    return f"({chr(ord('a') + row_idx * ncols + col_idx)})"


def draw_bar_figure(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    methods_for_legend = candidate_methods(rows)
    facets = active_facets(rows)
    nrows = len(METRICS)
    ncols = len(facets)
    figsize = (2.05 * ncols, STYLE["figsize"][1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for row_idx, (metric_label, mean_key, std_key) in enumerate(METRICS):
        for col_idx, (dataset, horizon) in enumerate(facets):
            ax = axes[row_idx, col_idx]
            methods = sorted_methods(rows, dataset, horizon, mean_key)
            if not methods:
                continue
            y_pos = np.arange(len(methods))[::-1]
            y_map = dict(zip(methods, y_pos))
            values_for_xlim: list[float] = []

            for method in methods:
                rec = lookup(rows, dataset, horizon, method)
                if rec is None:
                    continue
                mean = float(rec[mean_key])
                std = float(rec[std_key])
                if math.isnan(mean):
                    continue
                std = 0.0 if math.isnan(std) else std
                ax.barh(
                    y_map[method], mean, xerr=std,
                    height=STYLE["bar_height"],
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=STYLE["bar_edge_width"],
                    capsize=STYLE["capsize"],
                    error_kw={
                        "elinewidth": STYLE["error_linewidth"],
                        "capthick": STYLE["error_capthick"],
                    },
                )
                values_for_xlim.extend([mean - std, mean + std])

            vals = finite(values_for_xlim)
            if vals:
                lo, hi = min(vals), max(vals)
                span = max(hi - lo, STYLE["xlim_min_span"])
                ax.set_xlim(
                    lo - STYLE["xlim_pad_left"] * span,
                    hi + STYLE["xlim_pad_right"] * span,
                )

            ax.set_title(f"{dataset} H={horizon}", fontsize=STYLE["title_fontsize"], pad=3)
            ax.grid(axis="x", color=STYLE["grid_color"], linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=STYLE["tick_fontsize"], length=2)
            ax.tick_params(axis="y", length=0)
            ax.set_yticks(y_pos)
            if col_idx == 0:
                ax.set_yticklabels(methods, fontsize=STYLE["method_label_fontsize"])
                ax.set_ylabel(metric_label, fontsize=STYLE["ylabel_fontsize"],
                              weight="bold", labelpad=7)
            else:
                ax.set_yticklabels([])

            ax.text(
                0.5, -0.22,
                subfigure_label(row_idx, col_idx, ncols),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=STYLE["subfigure_label_fontsize"], weight="bold",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], label=m)
        for m in methods_for_legend
    ]
    fig.legend(
        handles=handles,
        ncol=len(methods_for_legend),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        frameon=False,
        fontsize=STYLE["legend_fontsize"],
        columnspacing=0.85,
    )
    fig.tight_layout(
        rect=STYLE["tight_layout_rect"],
        w_pad=STYLE["w_pad"],
        h_pad=STYLE["h_pad"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / "fig09_overall_prediction_performance_bars_mae.png",
        dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    draw_bar_figure(load_rows(DATA_CSV))


if __name__ == "__main__":
    main()
