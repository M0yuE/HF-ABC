from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).parent
DATA_CSV = SCRIPT_DIR / "fig16_algorithm_search_convergence_data.csv"
OUT_PATH = SCRIPT_DIR / "outputs" / "fig16_algorithm_search_convergence.png"

METHOD_ORDER = ["Random", "GA", "PSO", "ABC", "ASHA", "Hyperband", "HF-ABC"]
METHOD_COLORS = {
    "Random": "#8A96A3",
    "GA": "#59A14F",
    "PSO": "#4FA9A7",
    "ABC": "#F28E2B",
    "ASHA": "#B07AA1",
    "Hyperband": "#E15759",
    "HF-ABC": "#174A8B",
}
METHOD_LINESTYLES = {
    "Random": (0, (3.0, 1.5)),
    "GA": (0, (4.8, 1.6)),
    "PSO": (0, (1.4, 1.4)),
    "ABC": (0, (5.0, 1.4, 1.2, 1.4)),
    "ASHA": (0, (2.2, 1.4)),
    "Hyperband": (0, (6.0, 1.8)),
    "HF-ABC": "solid",
}


def load_curves() -> dict[str, tuple[list[float], list[float]]]:
    curves: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with DATA_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            curves[row["method"]].append(
                (float(row["candidate_epoch"]), float(row["best_val_mse"]))
            )

    result: dict[str, tuple[list[float], list[float]]] = {}
    missing = [method for method in METHOD_ORDER if method not in curves]
    if missing:
        raise ValueError(f"Missing methods in data table: {missing}")

    for method in METHOD_ORDER:
        points = sorted(curves[method], key=lambda item: item[0])
        result[method] = ([p[0] for p in points], [p[1] for p in points])
    return result


def improvement_points(
    xs: list[float], ys: list[float], tol: float = 1e-12
) -> tuple[list[float], list[float]]:
    keep = [0]
    for idx in range(1, len(ys)):
        if ys[idx] < ys[idx - 1] - tol:
            keep.append(idx)
    if keep[-1] != len(xs) - 1:
        keep.append(len(xs) - 1)
    return [xs[idx] for idx in keep], [ys[idx] for idx in keep]


def smooth_monotone_path(
    xs: list[float], ys: list[float], points_per_segment: int = 28
) -> tuple[list[float], list[float]]:
    px, py = improvement_points(xs, ys)
    if len(px) < 2:
        return px, py

    smooth_x = [px[0]]
    smooth_y = [py[0]]
    for x0, x1, y0, y1 in zip(px[:-1], px[1:], py[:-1], py[1:]):
        if x1 <= x0:
            continue
        for i in range(1, points_per_segment + 1):
            t = i / points_per_segment
            ease = t * t * (3.0 - 2.0 * t)
            smooth_x.append(x0 + (x1 - x0) * t)
            smooth_y.append(y0 + (y1 - y0) * ease)

    best = float("inf")
    for idx, val in enumerate(smooth_y):
        best = min(best, val)
        smooth_y[idx] = best
    return smooth_x, smooth_y


def first_crossing(xs: list[float], ys: list[float], target: float) -> tuple[float, float] | None:
    for x, y in zip(xs, ys):
        if y <= target:
            return x, y
    return None


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(direction="out", length=3.0, width=0.75, labelsize=7.3, pad=2.0)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.65)
    ax.grid(axis="x", color="#EEF2F7", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def plot_panel(
    ax: plt.Axes,
    curves: dict[str, tuple[list[float], list[float]]],
    best_competing: float,
    zoom: bool,
) -> None:
    for method in METHOD_ORDER:
        xs, ys = curves[method]
        sx, sy = smooth_monotone_path(xs, ys)
        is_hf = method == "HF-ABC"
        lw = 2.55 if is_hf else 1.65 if method in {"ASHA", "Hyperband"} else 1.35
        alpha = 1.0 if is_hf else 0.86
        zorder = 8 if is_hf else 5 if method in {"ASHA", "Hyperband"} else 3
        ax.plot(
            sx,
            sy,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )

        mx, my = improvement_points(xs, ys)
        ax.scatter(
            mx,
            my,
            s=10 if is_hf else 6,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.96 if is_hf else 0.72,
            zorder=zorder + 1,
        )

    ax.axhline(
        best_competing,
        color="#596273",
        linestyle=(0, (1.4, 1.4)),
        linewidth=0.9,
        alpha=0.80,
        zorder=1,
    )
    style_axes(ax)
    ax.set_xlim(0, 312)
    ax.set_xlabel("Candidate epochs (CE)", fontsize=8.2, labelpad=3.0)

    if zoom:
        ax.set_xlim(20, 312)
        ax.set_ylim(0.6055, 0.6235)
        ax.set_xticks([50, 100, 150, 200, 250, 300])
        ax.set_yticks([0.606, 0.610, 0.614, 0.618, 0.622])
    else:
        ax.set_ylim(0.6045, 0.6525)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.set_yticks([0.606, 0.616, 0.626, 0.636, 0.646])


def main() -> None:
    curves = load_curves()
    best_competing = min(min(ys) for m, (_, ys) in curves.items() if m != "HF-ABC")
    hf_cross = first_crossing(*curves["HF-ABC"], best_competing)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.85,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 2.85),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.24},
    )

    plot_panel(axes[0], curves, best_competing, zoom=False)
    plot_panel(axes[1], curves, best_competing, zoom=True)

    axes[0].set_ylabel("Best observed validation MSE", fontsize=8.2, labelpad=4.0)
    axes[0].set_title("Full search budget", fontsize=8.8, pad=5.0)
    axes[1].set_title("Late-budget zoom", fontsize=8.8, pad=5.0)

    if hf_cross is not None:
        cross_x, cross_y = hf_cross
        for ax in axes:
            ax.axvline(
                cross_x,
                color=METHOD_COLORS["HF-ABC"],
                linestyle=(0, (2.4, 1.4)),
                linewidth=0.85,
                alpha=0.55,
                zorder=2,
            )
        axes[1].annotate(
            "HF-ABC crosses\nbest baseline",
            xy=(cross_x, cross_y),
            xytext=(70, 0.6212),
            textcoords="data",
            fontsize=6.6,
            color="#253245",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.7,
                "color": "#253245",
                "shrinkA": 2,
                "shrinkB": 3,
            },
        )

    axes[0].text(
        -0.12,
        1.06,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="top",
        ha="left",
    )
    axes[1].text(
        -0.10,
        1.06,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="top",
        ha="left",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.55 if method == "HF-ABC" else 1.65 if method in {"ASHA", "Hyperband"} else 1.35,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#596273",
            linestyle=(0, (1.4, 1.4)),
            linewidth=0.9,
            label="Best non-HF final",
        )
    )
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        fontsize=6.8,
        handlelength=2.35,
        handletextpad=0.52,
        columnspacing=0.85,
    )

    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.20, top=0.80, wspace=0.24)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
