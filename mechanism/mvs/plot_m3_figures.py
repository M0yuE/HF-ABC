from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "m3_main_v1_20_seeds"
OUTPUT = ROOT.parents[2] / "paper" / "figures" / "mvs"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

COLORS = {
    "hf": "#0F4D92",
    "supported": "#2E9E44",
    "unsupported": "#E53935",
    "trend": "#42949E",
    "variant": "#B4C0E4",
    "baseline": "#767676",
    "light": "#D8D8D8",
    "dark": "#272727",
}

METHOD_LABELS = {
    "hf_abc": "HF-ABC",
    "hf_no_projection": "No projection",
    "hf_no_mf": "No multi-fidelity",
    "hf_uniform_parent_mf": "Uniform parent",
    "hf_no_restart": "No restart",
    "abc_projected_mf": "Projected ABC-MF",
    "random_projected_mf": "Projected random-MF",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def forest(
    ax,
    rows: list[dict[str, str]],
    labels: list[str],
    xlim: tuple[float, float],
) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, row, label in zip(y, rows, labels):
        estimate = float(row["median_difference_hf_minus_other"])
        low = float(row["bootstrap95_low"])
        high = float(row["bootstrap95_high"])
        adjusted = float(row["holm_adjusted_p"])
        if estimate < 0 and adjusted < 0.05:
            color = COLORS["supported"]
        elif estimate < 0:
            color = COLORS["trend"]
        else:
            color = COLORS["unsupported"]
        ax.plot([low, high], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter(
            [estimate],
            [yi],
            s=32,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            xlim[1],
            yi,
            f"{estimate:+.3f}",
            ha="right",
            va="center",
            fontsize=7,
            color=COLORS["dark"],
        )
    ax.axvline(0, color=COLORS["baseline"], ls="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.35, len(rows) - 0.55)
    ax.set_xlabel("Median regret difference (HF-ABC − comparator)")
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.01,
        0.97,
        "← HF better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=COLORS["supported"],
    )
    ax.text(
        0.99,
        0.97,
        "Other better →",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["unsupported"],
    )


def make_effect_figure() -> None:
    comparisons = read_csv(RESULTS / "comparisons_vs_hf.csv")
    lookup = {row["comparison"].split(" vs ", 1)[1]: row for row in comparisons}
    large_methods = ["hf_no_projection", "hf_no_mf"]
    local_methods = [
        "abc_projected_mf",
        "random_projected_mf",
        "hf_uniform_parent_mf",
        "hf_no_restart",
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        gridspec_kw={"width_ratios": [0.92, 1.38], "wspace": 0.68},
    )
    forest(
        axes[0],
        [lookup[name] for name in large_methods],
        [METHOD_LABELS[name] for name in large_methods],
        (-7.0, 0.6),
    )
    axes[0].set_title("Core mechanisms", loc="left", fontsize=9, fontweight="bold")
    panel_label(axes[0], "a")
    forest(
        axes[1],
        [lookup[name] for name in local_methods],
        [METHOD_LABELS[name] for name in local_methods],
        (-0.27, 0.27),
    )
    axes[1].set_title(
        "Neighborhood and search allocation",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    panel_label(axes[1], "b")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.86)
    save_all(fig, "figure_5_1_mvs_effects")


def make_overview_figure() -> None:
    methods = read_csv(RESULTS / "method_overall_summary.csv")
    traces = read_csv(RESULTS / "trace_mechanism_summary.csv")
    method_lookup = {row["method"]: row for row in methods}
    trace_lookup = {row["method"]: row for row in traces}
    order = sorted(
        METHOD_LABELS,
        key=lambda name: float(method_lookup[name]["mean_instance_rank"]),
        reverse=True,
    )
    y = np.arange(len(order))
    colors = [
        COLORS["hf"]
        if name == "hf_abc"
        else COLORS["variant"]
        if name.startswith("hf_")
        else COLORS["baseline"]
        for name in order
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.55),
        gridspec_kw={"width_ratios": [1.25, 0.85, 1.0], "wspace": 0.62},
    )
    ranks = [float(method_lookup[name]["mean_instance_rank"]) for name in order]
    bars = axes[0].barh(y, ranks, color=colors, edgecolor="white", height=0.68)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD_LABELS[name] for name in order])
    axes[0].set_xlabel("Mean instance rank (lower is better)")
    axes[0].set_xlim(0, 7.45)
    axes[0].tick_params(axis="y", length=0)
    for bar, value in zip(bars, ranks):
        axes[0].text(
            value + 0.10,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=7,
        )
    axes[0].set_title("Overall rank", loc="left", fontsize=9, fontweight="bold")
    panel_label(axes[0], "a")

    feasible = [float(trace_lookup[name]["request_feasible_rate"]) for name in order]
    bars = axes[1].barh(y, feasible, color=colors, edgecolor="white", height=0.68)
    axes[1].set_yticks([])
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Feasible request rate")
    axes[1].set_title("Feasibility", loc="left", fontsize=9, fontweight="bold")
    for bar, value in zip(bars, feasible):
        axes[1].text(
            min(value + 0.025, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value < 0.96 else "right",
            fontsize=7,
        )
    panel_label(axes[1], "b")

    actual = [float(method_lookup[name]["mean_actual_cost"]) for name in order]
    bars = axes[2].barh(y, actual, color=colors, edgecolor="white", height=0.68)
    axes[2].axvline(3744, color=COLORS["dark"], ls="--", lw=1.0)
    axes[2].set_yticks([])
    axes[2].set_xlim(0, 4000)
    axes[2].set_xlabel("Mean executed fidelity cost")
    axes[2].set_title("Archive reuse", loc="left", fontsize=9, fontweight="bold")
    for bar, name, value in zip(bars, order, actual):
        hits = float(method_lookup[name]["mean_cache_hits_per_run"])
        axes[2].text(
            value - 55,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}\n({hits:.1f} hits)",
            ha="right",
            va="center",
            fontsize=6.5,
            color="white" if value > 1800 else COLORS["dark"],
            linespacing=0.85,
        )
    axes[2].text(
        3744,
        len(order) - 0.25,
        "Nominal budget",
        rotation=90,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["dark"],
    )
    panel_label(axes[2], "c")
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.18, top=0.88)
    save_all(fig, "figure_5_2_mvs_overview")


def save_all(fig, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(
            OUTPUT / f"{stem}.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    make_effect_figure()
    make_overview_figure()
    print(OUTPUT)


if __name__ == "__main__":
    main()
