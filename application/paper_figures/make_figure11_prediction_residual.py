from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig11_prediction_residual_data.csv"
OUT_PNG = SCRIPT_DIR / "outputs" / "fig11_prediction_residual.png"

COLORS = {"Ground truth": "#202A36", "HF-ABC": "#174A8B", "PSO": "#4FA9A7"}


def load_rows() -> list[dict[str, str]]:
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def grouped_xy(rows: list[dict[str, str]], panel: str, metric: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["panel"] == panel and row["metric"] == metric:
            groups[row["method"]].append((float(row["step"]), float(row["value"])))
    return {
        method: (
            np.array([x for x, _ in sorted(values)]),
            np.array([y for _, y in sorted(values)]),
        )
        for method, values in groups.items()
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6EBF2", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.7, labelsize=7.2)


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.6,
            "axes.linewidth": 0.75,
        }
    )

    pred = grouped_xy(rows, "prediction_curve", "normalized_value")
    err = grouped_xy(rows, "residual_curve", "absolute_error")
    baseline = next((m for m in pred if m not in {"Ground truth", "HF-ABC"}), "PSO")
    sample = next((r["sample_index"] for r in rows if r["sample_index"]), "")

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax = axes[0]
    for method in ["Ground truth", "HF-ABC", baseline]:
        if method not in pred:
            continue
        x, y = pred[method]
        ax.plot(
            x,
            y,
            color=COLORS.get(method, "#777777"),
            lw=2.0 if method in {"Ground truth", "HF-ABC"} else 1.55,
            ls="-" if method in {"Ground truth", "HF-ABC"} else (0, (4, 1.6)),
            label=method,
        )
    ax.set_title("Representative N2O forecast (H=120)", fontsize=9.4, fontweight="semibold", pad=4)
    ax.set_ylabel("Normalized value", fontsize=8.2)
    ax.legend(loc="upper right", fontsize=7.0, frameon=False, ncol=3)
    ax.text(0.01, 0.05, f"sample {sample}", transform=ax.transAxes, fontsize=6.8, color="#5B6573")
    style_axis(ax)

    ax = axes[1]
    for method in ["HF-ABC", baseline]:
        if method not in err:
            continue
        x, y = err[method]
        ax.plot(
            x,
            y,
            color=COLORS.get(method, "#777777"),
            lw=1.85 if method == "HF-ABC" else 1.45,
            ls="-" if method == "HF-ABC" else (0, (4, 1.6)),
            label=method,
        )
    ax.set_title("Absolute residual on the same segment", fontsize=9.4, fontweight="semibold", pad=4)
    ax.set_xlabel("Forecast step", fontsize=8.2)
    ax.set_ylabel("|error|", fontsize=8.2)
    ax.legend(loc="upper right", fontsize=7.0, frameon=False, ncol=2)
    style_axis(ax)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
