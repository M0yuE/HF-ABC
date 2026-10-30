from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).parent / "outputs"
DATA_CSV = OUT_DIR / "fig09_overall_prediction_performance_data.csv"

METHOD_ORDER = [
    "Fixed",
    "iTransformer-CSG",
    "Random",
    "GA",
    "PSO",
    "ABC",
    "ASHA",
    "Hyperband",
    "Optuna-TPE",
    "HF-ABC",
]
FACETS = [("N2O", 60), ("N2O", 120), ("N2O", 240), ("Agtrup", 120)]
METRICS = [
    ("MSE", "mse_mean", "mse_std"),
    ("Key-MSE", "key_mse_mean", "key_mse_std"),
]

COL = {
    "ink": "#243044",
    "grid": "#E8EDF4",
    "fixed": "#9CA3AF",
    "iTransformer-CSG": "#F3B7ED",
    "Random": "#4E79A7",
    "GA": "#59A14F",
    "PSO": "#76B7B2",
    "ABC": "#F28E2B",
    "ASHA": "#B07AA1",
    "Hyperband": "#EB1228",
    "Optuna-TPE": "#8C6D31",
    "HF-ABC": "#0F4D92",
}

METHOD_COLORS = {
    "Fixed": COL["fixed"],
    "iTransformer-CSG": COL["iTransformer-CSG"],
    "Random": COL["Random"],
    "GA": COL["GA"],
    "PSO": COL["PSO"],
    "ABC": COL["ABC"],
    "ASHA": COL["ASHA"],  # stands for ASHA
    "Hyperband": COL["Hyperband"],
    "Optuna-TPE": COL["Optuna-TPE"],
    "HF-ABC": COL["HF-ABC"],
}


def to_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return math.nan
    return float(text)


def load_rows_from_csv() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "method": row["method"],
            }
            for key in [
                "mse_mean",
                "mse_std",
                "mae_mean",
                "mae_std",
                "key_mse_mean",
                "key_mse_std",
                "key_mae_mean",
                "key_mae_std",
            ]:
                converted[key] = to_float(row.get(key, "nan"))
            rows.append(converted)
    return rows


def lookup(rows: list[dict[str, object]], dataset: str, horizon: int, method: str) -> dict[str, object] | None:
    for row in rows:
        if row["dataset"] == dataset and row["horizon"] == horizon and row["method"] == method:
            return row
    return None


def finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def draw_figure(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.4,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(7.8, 3.25), sharey=True)
    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    y_map = dict(zip(METHOD_ORDER, y_positions))

    for row_idx, (metric_label, mean_key, std_key) in enumerate(METRICS):
        for col_idx, (dataset, horizon) in enumerate(FACETS):
            ax = axes[row_idx, col_idx]
            vals_for_limits: list[float] = []

            for method in METHOD_ORDER:
                rec = lookup(rows, dataset, horizon, method)
                if rec is None:
                    continue
                mean = float(rec[mean_key])
                std = float(rec[std_key])
                if math.isnan(mean):
                    continue
                std = 0.0 if math.isnan(std) else std
                vals_for_limits.extend([mean - std, mean + std])
                marker = "D" if method == "HF-ABC" else "o"
                size = 42 if method == "HF-ABC" else 30
                ax.errorbar(
                    mean,
                    y_map[method],
                    xerr=std,
                    fmt=marker,
                    markersize=size**0.5,
                    color=METHOD_COLORS[method],
                    markerfacecolor=METHOD_COLORS[method],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    ecolor=METHOD_COLORS[method],
                    elinewidth=1.0,
                    capsize=2.2,
                    zorder=3 if method != "HF-ABC" else 4,
                )

            vals = finite(vals_for_limits)
            if vals:
                lo, hi = min(vals), max(vals)
                span = max(hi - lo, 0.002)
                ax.set_xlim(lo - 0.16 * span, hi + 0.20 * span)
            ax.set_title(f"{dataset} H={horizon}", fontsize=7.6, pad=3, color=COL["ink"])
            ax.grid(axis="x", color=COL["grid"], lw=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=7.2, length=2)
            ax.tick_params(axis="y", length=0)
            ax.set_yticks(y_positions)
            if col_idx == 0:
                ax.set_yticklabels(METHOD_ORDER, fontsize=7.2)
                ax.set_ylabel(metric_label, fontsize=8.2, weight="bold", color=COL["ink"], labelpad=6)
            else:
                ax.set_yticklabels([])

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="D" if method == "HF-ABC" else "o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markersize=5.6,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        frameon=False,
        fontsize=6.4,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=[0.035, 0.025, 0.995, 0.88], w_pad=0.65, h_pad=0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig09_overall_prediction_performance.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    draw_figure(load_rows_from_csv())


if __name__ == "__main__":
    main()
