from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig15_seed_level_mse_data.csv"
OUT_PNG = SCRIPT_DIR / "outputs" / "fig15_seed_level_mse.png"

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
    fig, ax = plt.subplots(figsize=(6.1, 3.1))
    rng = np.random.default_rng(2026)
    labels = []
    xpos = 0
    for method in METHOD_ORDER:
        vals = np.array([float(row["value"]) for row in rows if row["method"] == method])
        if len(vals) == 0:
            continue
        xpos += 1
        labels.append(short_label(method))
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=24 if method == "HF-ABC" else 18,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            alpha=0.95,
        )
        mean = float(np.mean(vals))
        ax.plot([xpos - 0.23, xpos + 0.23], [mean, mean], color="#111827", lw=1.05)
        if len(vals) == 1:
            ax.text(xpos, vals[0] + 0.00045, "1 seed", ha="center", va="bottom", fontsize=6.0, color="#5B6573")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.1)
    ax.set_title("Final-seed MSE points (N2O H=120)", fontsize=9.4, fontweight="semibold", pad=4)
    ax.set_ylabel("Run MSE", fontsize=8.2)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
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
