from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = OUT / "fig10_anytime_performance_data.csv"


METHOD_ORDER = ["Random", "GA", "ABC", "PSO", "ASHA", "Hyperband", "HF-ABC"]
NON_HF = [m for m in METHOD_ORDER if m != "HF-ABC"]
HF_BLUE = "#164A8B"
BASELINE = "#262626"
GRID = "#E9EEF4"
TEXT = "#222222"

PALETTE = {
    "Random": "#A8B0B8",
    "GA": "#92A98E",
    "ABC": "#C2AA88",
    "PSO": "#7FAEAD",
    "ASHA": "#B29BBB",
    "Hyperband": "#C18C8C",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.3,
            "axes.linewidth": 0.85,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 420,
        }
    )


def compress_curve(group: pd.DataFrame) -> pd.DataFrame:
    """Keep only incumbent-improvement points plus the final budget endpoint."""
    group = group.sort_values("candidate_epoch").reset_index(drop=True)
    keep = [0]
    best = float(group.loc[0, "best_val_mse"])
    for idx in range(1, len(group)):
        val = float(group.loc[idx, "best_val_mse"])
        if val < best - 1e-12:
            keep.append(idx)
            best = val
    if keep[-1] != len(group) - 1:
        keep.append(len(group) - 1)
    return group.loc[keep].drop_duplicates("candidate_epoch")


def smooth_xy(x: np.ndarray, y: np.ndarray, n: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving interpolation when SciPy is present; linear fallback otherwise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, idx = np.unique(x, return_index=True)
    x, y = unique_x, y[idx]
    xi = np.linspace(float(x.min()), float(x.max()), n)
    if len(x) < 3:
        return xi, np.interp(xi, x, y)
    try:
        from scipy.interpolate import PchipInterpolator

        yi = PchipInterpolator(x, y)(xi)
    except Exception:
        yi = np.interp(xi, x, y)
    # Incumbent traces should never visually improve beyond the observed best.
    yi = np.maximum(yi, float(np.min(y)))
    yi = np.minimum.accumulate(yi)
    return xi, yi


def draw_method(ax, group: pd.DataFrame, color: str, lw: float, alpha: float, zorder: int, markers: bool = False):
    compact = compress_curve(group)
    x = compact["candidate_epoch"].to_numpy()
    y = compact["best_val_mse"].to_numpy()
    xi, yi = smooth_xy(x, y)
    ax.plot(xi, yi, color=color, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")
    if markers:
        ax.scatter(x, y, s=18, color=color, edgecolor="white", linewidth=0.55, zorder=zorder + 1)
    return compact, xi, yi


def main() -> None:
    setup_style()
    df = pd.read_csv(DATA)
    df = df[df["horizon"].astype(int) == 120].copy()

    final_scores = df.sort_values("candidate_epoch").groupby("method")["best_val_mse"].last()
    best_non_hf_method = final_scores.loc[NON_HF].idxmin()
    best_non_hf = float(final_scores.loc[best_non_hf_method])
    hf = df[df["method"] == "HF-ABC"]

    fig = plt.figure(figsize=(7.1, 3.05))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.28, 1.0],
        left=0.075,
        right=0.955,
        bottom=0.20,
        top=0.83,
        wspace=0.29,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for method in NON_HF:
        group = df[df["method"] == method]
        draw_method(ax1, group, PALETTE[method], lw=1.55, alpha=0.47, zorder=2)

    compact_hf, xi_hf, yi_hf = draw_method(ax1, hf, HF_BLUE, lw=2.55, alpha=1.0, zorder=5, markers=True)
    ax1.axhline(best_non_hf, color=BASELINE, lw=0.95, ls=(0, (2, 2.2)), zorder=1)

    ax1.set_title("Anytime improvement under a fixed search budget", loc="left", fontweight="bold", pad=8)
    ax1.set_xlabel("Candidate epochs (CE)")
    ax1.set_ylabel("Best observed validation MSE")
    ax1.set_xlim(0, 340)
    ax1.set_ylim(0.6048, 0.6518)
    ax1.xaxis.set_major_locator(MultipleLocator(50))
    ax1.yaxis.set_major_locator(MultipleLocator(0.01))
    ax1.grid(True, axis="y", color=GRID, lw=0.8)
    ax1.grid(True, axis="x", color=GRID, lw=0.55, alpha=0.55)

    ax1.plot([], [], color="#AAB3BC", lw=1.8, alpha=0.7, label="Non-HF baselines")
    ax1.plot([], [], color=HF_BLUE, lw=2.6, label="HF-ABC")
    ax1.plot([], [], color=BASELINE, lw=0.95, ls=(0, (2, 2.2)), label="Best non-HF final")
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=3,
        fontsize=6.8,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    ax1.text(211, best_non_hf + 0.00075, f"best non-HF final ({best_non_hf_method})", fontsize=6.6, color=TEXT)
    ax1.text(320, float(final_scores.loc["HF-ABC"]) - 0.001, "HF-ABC", color=HF_BLUE, fontsize=7.3, fontweight="bold")

    # Right panel: smooth excess curves relative to the best non-HF endpoint.
    for method in NON_HF:
        group = compress_curve(df[df["method"] == method])
        x = group["candidate_epoch"].to_numpy()
        y = (group["best_val_mse"].to_numpy() - best_non_hf) * 1000
        xi, yi = smooth_xy(x, y)
        ax2.plot(xi, yi, color=PALETTE[method], lw=1.45, alpha=0.43, solid_capstyle="round", zorder=2)

    hfg = compress_curve(hf)
    x = hfg["candidate_epoch"].to_numpy()
    y = (hfg["best_val_mse"].to_numpy() - best_non_hf) * 1000
    xi, yi = smooth_xy(x, y)
    ax2.plot(xi, yi, color=HF_BLUE, lw=2.55, zorder=5, solid_capstyle="round")
    ax2.scatter(x, y, s=18, color=HF_BLUE, edgecolor="white", linewidth=0.55, zorder=6)

    ax2.axhline(0, color=BASELINE, lw=0.95, ls=(0, (2, 2.2)), zorder=1)
    ax2.fill_between([40, 322], [-2.65, -2.65], [0, 0], color=HF_BLUE, alpha=0.055, zorder=0)
    ax2.set_title("Margin against the strongest non-HF final", loc="left", fontweight="bold", pad=8)
    ax2.set_xlabel("Candidate epochs (CE)")
    ax2.set_ylabel("Excess validation MSE (x10$^{-3}$)")
    ax2.set_xlim(40, 322)
    ax2.set_ylim(-2.45, 7.2)
    ax2.xaxis.set_major_locator(MultipleLocator(50))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.grid(True, axis="y", color=GRID, lw=0.8)
    ax2.grid(True, axis="x", color=GRID, lw=0.55, alpha=0.55)
    ax2.text(80, -1.15, "lower than the strongest\nnon-HF final baseline", color=HF_BLUE, fontsize=6.7, fontweight="bold")
    ax2.text(304, 0.23, f"{best_non_hf_method} final", color=TEXT, fontsize=6.6, ha="right")
    ax2.text(305, float(y[-1]) - 0.5, "HF-ABC", color=HF_BLUE, fontsize=7.3, fontweight="bold", va="center")

    for ax, letter in [(ax1, "a"), (ax2, "b")]:
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="bottom")
        ax.tick_params(length=3.0, width=0.8, color=TEXT)

    fig.text(
        0.075,
        0.045,
        "Smoothed curves are shown only as visual guides; all comparisons are based on the observed incumbent MSE values.",
        fontsize=6.6,
        color="#4A5561",
        ha="left",
        va="bottom",
    )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig10_anytime_performance_smooth.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
