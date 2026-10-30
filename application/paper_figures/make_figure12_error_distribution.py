from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig12_error_distribution_data.csv"
OUT_PNG = SCRIPT_DIR / "outputs" / "fig12_error_distribution.png"

METHOD_ORDER = ["Random Search", "ABC", "GA", "PSO", "ASHA", "Hyperband", "HF-ABC"]
COLORS = {
    "Random Search": "#7A8591",
    "ABC": "#F28E2B",
    "GA": "#59A14F",
    "PSO": "#4FA9A7",
    "ASHA": "#B07AA1",
    "Hyperband": "#E15759",
    "HF-ABC": "#174A8B",
}


def load_rows() -> list[dict[str, str]]:
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def short_label(method: str) -> str:
    return method.replace("Random Search", "Random").replace("Hyperband", "HB")


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

    methods = [m for m in METHOD_ORDER if any(row["method"] == m for row in rows)]
    values = [np.array([float(row["value"]) for row in rows if row["method"] == method]) for method in methods]

    fig, ax = plt.subplots(figsize=(6.1, 3.25))
    parts = ax.violinplot(values, showextrema=False, widths=0.78)
    for body, method in zip(parts["bodies"], methods):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("white")
        body.set_alpha(0.72 if method == "HF-ABC" else 0.42)

    for i, vals in enumerate(values, start=1):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color="#111827", lw=1.1)
        ax.plot([i, i], [q1, q3], color="#111827", lw=1.0)

    ax.set_xticks(np.arange(1, len(methods) + 1))
    ax.set_xticklabels([short_label(m) for m in methods], rotation=35, ha="right", fontsize=7.2)
    ax.set_title("Sample-level N2O MAE distribution (H=120)", fontsize=9.4, fontweight="semibold", pad=4)
    ax.set_ylabel("MAE per sample", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6EBF2", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.7, labelsize=7.2)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
