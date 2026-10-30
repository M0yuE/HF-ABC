from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig13_key_variable_heatmap_data.csv"
OUT_PNG = SCRIPT_DIR / "outputs" / "fig13_key_variable_heatmap.png"

METHOD_ORDER = ["Random Search", "ABC", "GA", "PSO", "ASHA", "Hyperband", "HF-ABC"]
KEY_ORDER = ["N2O", "NH4", "NO3", "PO4"]


def load_rows() -> list[dict[str, str]]:
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def short_label(method: str) -> str:
    return method.replace("Random Search", "Random").replace("Hyperband", "HB")


def main() -> None:
    rows = load_rows()
    methods = [m for m in METHOD_ORDER if any(row["method"] == m for row in rows)]
    matrix = np.full((len(KEY_ORDER), len(methods)), np.nan)
    for i, var in enumerate(KEY_ORDER):
        for j, method in enumerate(methods):
            vals = [float(row["value"]) for row in rows if row["method"] == method and row["variable"] == var]
            if vals:
                matrix[i, j] = vals[0]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.6,
            "axes.linewidth": 0.75,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.75))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([short_label(m) for m in methods], rotation=35, ha="right", fontsize=7.0)
    ax.set_yticks(np.arange(len(KEY_ORDER)))
    ax.set_yticklabels(KEY_ORDER, fontsize=7.4)
    ax.set_title("Key-variable MSE by method (H=120)", fontsize=9.4, fontweight="semibold", pad=4)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.1, color="#0F172A")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6.7, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
