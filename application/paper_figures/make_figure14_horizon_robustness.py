from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig14_horizon_robustness_data.csv"
OUT_PNG = SCRIPT_DIR / "outputs" / "fig14_horizon_robustness.png"

METHOD_ORDER = ["iTransformer", "Random Search", "ABC", "GA", "PSO", "ASHA", "Hyperband", "HF-ABC"]
COLORS = {
    "iTransformer": "#9AA3AD",
    "Random Search": "#7A8591",
    "ABC": "#F28E2B",
    "GA": "#59A14F",
    "PSO": "#4FA9A7",
    "ASHA": "#B07AA1",
    "Hyperband": "#E15759",
    "HF-ABC": "#174A8B",
}
LINESTYLES = {
    "iTransformer": (0, (2.0, 1.2)),
    "Random Search": (0, (3.0, 1.5)),
    "ABC": (0, (5.0, 1.4, 1.2, 1.4)),
    "GA": (0, (4.8, 1.6)),
    "PSO": (0, (1.4, 1.4)),
    "ASHA": (0, (2.2, 1.4)),
    "Hyperband": (0, (6.0, 1.8)),
    "HF-ABC": "solid",
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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    metrics = [("mse_mean", "Overall MSE"), ("key_mse_mean", "Key-variable MSE")]
    for ax, (metric, title) in zip(axes, metrics):
        for method in METHOD_ORDER:
            items = sorted(
                [row for row in rows if row["method"] == method and row["metric"] == metric],
                key=lambda r: int(r["horizon"]),
            )
            if not items:
                continue
            xs = np.array([int(row["horizon"]) for row in items])
            ys = np.array([float(row["value"]) for row in items])
            yerr = np.array([float(row["lower"]) if row["lower"] else 0.0 for row in items])
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=COLORS[method],
                lw=2.1 if method == "HF-ABC" else 1.15,
                marker="o",
                ms=4.2 if method == "HF-ABC" else 3.0,
                capsize=2.0,
                linestyle=LINESTYLES[method],
                label=short_label(method),
            )
        ax.set_title(title, fontsize=9.4, fontweight="semibold", pad=4)
        ax.set_xlabel("Prediction horizon", fontsize=8.2)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", color="#E6EBF2", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3.0, width=0.7, labelsize=7.2)
        ax.set_xticks([60, 120, 240])
    axes[0].set_ylabel("Error", fontsize=8.2)
    axes[0].legend(loc="upper left", fontsize=6.3, frameon=False, ncol=2)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
