from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = OUT / "fig10_anytime_performance_data.csv"


METHOD_ORDER = ["Random", "GA", "ABC", "PSO", "ASHA", "Hyperband", "HF-ABC"]
METHOD_LABELS = {
    "Random": "Random",
    "GA": "GA",
    "ABC": "ABC",
    "PSO": "PSO",
    "ASHA": "ASHA",
    "Hyperband": "Hyperband",
    "HF-ABC": "HF-ABC",
}

NEUTRAL = {
    "Random": "#9AA3AD",
    "GA": "#8FA58B",
    "ABC": "#B7A085",
    "PSO": "#7EA6A5",
    "ASHA": "#AD98B5",
    "Hyperband": "#B88484",
}
HF_BLUE = "#174A8B"
BASELINE = "#222222"
GRID = "#E8EDF3"
TEXT = "#222222"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 420,
        }
    )


def step_xy(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    group = group.sort_values("candidate_epoch")
    return group["candidate_epoch"].to_numpy(), group["best_val_mse"].to_numpy()


def endpoint_label(ax, x, y, label, color, dy=0.0, weight="normal", size=7.0):
    ax.text(
        x,
        y + dy,
        label,
        color=color,
        ha="left",
        va="center",
        fontsize=size,
        fontweight=weight,
        clip_on=False,
    )


def main() -> None:
    setup_style()
    df = pd.read_csv(DATA)
    df = df[df["horizon"].astype(int) == 120].copy()

    final_scores = (
        df.sort_values("candidate_epoch")
        .groupby("method", sort=False)["best_val_mse"]
        .last()
    )
    non_hf_methods = [m for m in METHOD_ORDER if m != "HF-ABC"]
    best_non_hf_method = final_scores.loc[non_hf_methods].idxmin()
    best_non_hf = float(final_scores.loc[best_non_hf_method])

    fig = plt.figure(figsize=(7.05, 3.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.32, 1.0],
        left=0.075,
        right=0.955,
        bottom=0.20,
        top=0.84,
        wspace=0.28,
    )
    ax_full = fig.add_subplot(gs[0, 0])
    ax_excess = fig.add_subplot(gs[0, 1])

    # Panel a: full-budget incumbent dynamics.
    for method in non_hf_methods:
        group = df[df["method"] == method]
        x, y = step_xy(group)
        ax_full.step(
            x,
            y,
            where="post",
            color=NEUTRAL[method],
            lw=1.25,
            alpha=0.62,
            zorder=2,
        )

    hf = df[df["method"] == "HF-ABC"]
    x_hf, y_hf = step_xy(hf)
    ax_full.step(x_hf, y_hf, where="post", color=HF_BLUE, lw=2.4, zorder=5)
    ax_full.scatter(x_hf, y_hf, s=16, color=HF_BLUE, edgecolor="white", linewidth=0.55, zorder=6)
    ax_full.axhline(best_non_hf, color=BASELINE, lw=0.95, ls=(0, (1.5, 1.8)), zorder=1)

    ax_full.set_xlim(0, 342)
    ax_full.set_ylim(0.6048, 0.6518)
    ax_full.set_xlabel("Candidate epochs (CE)")
    ax_full.set_ylabel("Best observed validation MSE")
    ax_full.set_title("Full-budget incumbent trajectory", loc="left", fontweight="bold", pad=8)
    ax_full.grid(True, axis="y", color=GRID, lw=0.8)
    ax_full.grid(True, axis="x", color=GRID, lw=0.55, alpha=0.55)
    ax_full.xaxis.set_major_locator(MultipleLocator(50))
    ax_full.yaxis.set_major_locator(MultipleLocator(0.01))

    ax_full.plot([], [], color="#A8B0B8", lw=1.5, alpha=0.75, label="Non-HF baselines")
    ax_full.plot([], [], color=HF_BLUE, lw=2.4, label="HF-ABC")
    ax_full.plot([], [], color=BASELINE, lw=0.95, ls=(0, (1.5, 1.8)), label="Best non-HF final")
    ax_full.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        handlelength=1.7,
        columnspacing=1.3,
        handletextpad=0.45,
        fontsize=6.8,
        borderaxespad=0.0,
    )
    endpoint_label(
        ax_full,
        314.5,
        float(final_scores.loc["HF-ABC"]),
        "HF-ABC",
        HF_BLUE,
        dy=-0.00010,
        weight="bold",
        size=7.2,
    )
    ax_full.text(
        198,
        best_non_hf + 0.00080,
        f"best non-HF final: {best_non_hf_method}",
        color=BASELINE,
        ha="left",
        va="bottom",
        fontsize=6.6,
    )

    # Panel b: excess validation MSE against the best non-HF final baseline.
    for method in non_hf_methods:
        group = df[df["method"] == method]
        x, y = step_xy(group)
        ax_excess.step(
            x,
            (y - best_non_hf) * 1000,
            where="post",
            color=NEUTRAL[method],
            lw=1.2,
            alpha=0.58,
            zorder=2,
        )
    ax_excess.step(
        x_hf,
        (y_hf - best_non_hf) * 1000,
        where="post",
        color=HF_BLUE,
        lw=2.35,
        zorder=5,
    )
    ax_excess.scatter(
        x_hf,
        (y_hf - best_non_hf) * 1000,
        s=16,
        color=HF_BLUE,
        edgecolor="white",
        linewidth=0.55,
        zorder=6,
    )
    ax_excess.axhline(0, color=BASELINE, lw=0.95, ls=(0, (1.5, 1.8)), zorder=1)
    ax_excess.fill_between(
        [40, 322],
        [-2.6, -2.6],
        [0, 0],
        color=HF_BLUE,
        alpha=0.055,
        zorder=0,
    )

    ax_excess.set_xlim(40, 322)
    ax_excess.set_ylim(-2.45, 7.25)
    ax_excess.set_xlabel("Candidate epochs (CE)")
    ax_excess.set_ylabel("Excess validation MSE (x10$^{-3}$)")
    ax_excess.set_title("Excess over strongest non-HF baseline", loc="left", fontweight="bold", pad=8)
    ax_excess.grid(True, axis="y", color=GRID, lw=0.8)
    ax_excess.grid(True, axis="x", color=GRID, lw=0.55, alpha=0.55)
    ax_excess.xaxis.set_major_locator(MultipleLocator(50))
    ax_excess.yaxis.set_major_locator(MultipleLocator(2))
    ax_excess.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

    ax_excess.text(
        64,
        -2.08,
        "below the strongest\nnon-HF final baseline",
        color=HF_BLUE,
        fontsize=6.7,
        ha="left",
        va="center",
        fontweight="bold",
    )

    endpoint_label(
        ax_excess,
        306,
        (float(final_scores.loc["HF-ABC"]) - best_non_hf) * 1000,
        "HF-ABC",
        HF_BLUE,
        dy=-0.10,
        weight="bold",
        size=7.1,
    )
    endpoint_label(
        ax_excess,
        306,
        0,
        f"{best_non_hf_method} final baseline",
        BASELINE,
        dy=0.34,
        size=6.6,
    )

    for ax, letter in [(ax_full, "a"), (ax_excess, "b")]:
        ax.text(
            -0.12,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.tick_params(length=3.2, width=0.8, color=TEXT)
        ax.spines["left"].set_color(TEXT)
        ax.spines["bottom"].set_color(TEXT)

    fig.text(
        0.075,
        0.045,
        "Incumbent curves report the best validation MSE observed up to each candidate-epoch budget; "
        "negative excess indicates performance better than the strongest non-HF final baseline.",
        fontsize=6.6,
        color="#4A5561",
        ha="left",
        va="bottom",
    )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig10_anytime_performance_nature.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
