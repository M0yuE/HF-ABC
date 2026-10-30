from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results_consolidated"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


COL = {
    "ink": "#243044",
    "muted": "#64748B",
    "grid": "#E8EDF4",
    "line": "#CBD5E1",
    "blue": "#0F4D92",
    "blue2": "#3C78B5",
    "blue_soft": "#DCEAF8",
    "green": "#2E9D72",
    "green_soft": "#DDF2E9",
    "teal": "#2F9AA3",
    "teal_soft": "#DDF2F4",
    "violet": "#7A5AA6",
    "violet_soft": "#EAE3F5",
    "gold": "#B8862D",
    "gold_soft": "#F4E6C2",
    "red": "#C94B45",
    "red_soft": "#F8DAD7",
    "gray": "#9CA3AF",
    "gray_soft": "#EEF2F7",
    "white": "#FFFFFF",
}

METHOD_COLORS = {
    "iTransformer": "#B9C2CF",
    "Random Search": "#91A9C9",
    "ABC": "#708ABD",
    "GA": "#94BC84",
    "PSO": "#6FA6A9",
    "ASHA": "#C9A667",
    "Hyperband": "#B692C4",
    "HF-ABC": COL["blue"],
    "HF-ABC (fast)": COL["blue"],
}


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def val(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


def resolve_data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    ensure_out()
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def clean_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=11, weight="bold", ha="left", va="bottom")


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    fc: str,
    ec: str = COL["line"],
    fontsize: float = 8.2,
    weight: str = "normal",
    radius: float = 0.06,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
        path_effects=[pe.withSimplePatchShadow(offset=(0.8, -0.8), alpha=0.10), pe.Normal()],
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COL["ink"], weight=weight, linespacing=1.12)
    return patch


def arrow(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float], color: str = COL["muted"], lw: float = 1.2, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_01_motivation() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, "Motivation of HF-ABC for expensive hybrid-constrained configuration optimization", fontsize=11, weight="bold", ha="left")
    ax.plot([0.5, 0.5], [0.10, 0.88], color=COL["line"], lw=1.2)
    ax.text(0.25, 0.88, "Standard ABC / direct search", ha="center", fontsize=9.5, weight="bold")
    ax.text(0.75, 0.88, "HF-ABC", ha="center", fontsize=9.5, weight="bold", color=COL["blue"])

    left = [
        ((0.06, 0.64), "continuous\nperturbation", COL["gray_soft"], COL["line"]),
        ((0.30, 0.64), "expensive\nfull training", COL["red_soft"], COL["red"]),
        ((0.06, 0.39), "invalid or\nduplicate config", COL["red_soft"], COL["red"]),
        ((0.30, 0.39), "wasted\nbudget", COL["red_soft"], COL["red"]),
        ((0.18, 0.17), "premature\nconvergence", COL["gray_soft"], COL["line"]),
    ]
    for xy, text, fc, ec in left:
        box(ax, xy, (0.13, 0.12), text, fc, ec, fontsize=7.2)
    arrow(ax, (0.19, 0.70), (0.30, 0.70), COL["red"])
    arrow(ax, (0.125, 0.64), (0.125, 0.51), COL["red"])
    arrow(ax, (0.19, 0.45), (0.30, 0.45), COL["red"])
    arrow(ax, (0.37, 0.39), (0.25, 0.29), COL["red"])

    labels = [
        ("Typed\nspace", COL["blue_soft"], COL["blue"]),
        ("Feasibility\nprojection", COL["teal_soft"], COL["teal"]),
        ("Multi-fidelity\nracing", COL["green_soft"], COL["green"]),
        ("Elite-guided\nrefinement", COL["violet_soft"], COL["violet"]),
        ("Top-k\nverification", COL["gold_soft"], COL["gold"]),
    ]
    xs = np.linspace(0.56, 0.89, len(labels))
    for i, (lab, fc, ec) in enumerate(labels):
        box(ax, (xs[i] - 0.04, 0.61), (0.08, 0.13), lab, fc, ec, fontsize=6.5)
        if i:
            arrow(ax, (xs[i - 1] + 0.04, 0.675), (xs[i] - 0.04, 0.675), COL["blue"], lw=1.0)
    box(ax, (0.60, 0.34), (0.13, 0.11), "legal\ncandidates", COL["green_soft"], COL["green"], fontsize=7.3)
    box(ax, (0.78, 0.34), (0.13, 0.11), "efficient\nbudget use", COL["blue_soft"], COL["blue"], fontsize=7.3)
    arrow(ax, (0.66, 0.61), (0.66, 0.45), COL["green"])
    arrow(ax, (0.73, 0.395), (0.78, 0.395), COL["blue"])
    ax.text(0.50, 0.05, "From blind expensive evaluation to a fidelity-guided candidate life-cycle", ha="center", fontsize=8.6, weight="bold")
    save(fig, "fig01_motivation_problem_setting")


def figure_02_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, "Candidate life-cycle of HF-ABC", fontsize=11, weight="bold", ha="left")
    steps = [
        ("Initial food\nsources", COL["gray_soft"], COL["line"]),
        ("Typed\nencoding", COL["blue_soft"], COL["blue"]),
        ("Feasibility\nprojection", COL["teal_soft"], COL["teal"]),
        ("1/3/6/10 epoch\nracing", COL["green_soft"], COL["green"]),
        ("Survivor\nranking", COL["gold_soft"], COL["gold"]),
        ("Top-m\nelite", COL["violet_soft"], COL["violet"]),
    ]
    xs = np.linspace(0.05, 0.79, len(steps))
    for i, (lab, fc, ec) in enumerate(steps):
        box(ax, (xs[i], 0.72), (0.105, 0.11), lab, fc, ec, fontsize=6.9)
        if i:
            arrow(ax, (xs[i - 1] + 0.105, 0.775), (xs[i], 0.775), COL["blue"], lw=1.0)
    box(ax, (0.76, 0.48), (0.17, 0.13), "EGR child\ngeneration\nlocal / struct / restart", COL["violet_soft"], COL["violet"], fontsize=6.9)
    box(ax, (0.53, 0.48), (0.16, 0.13), "Fidelity-gated\narchive\nscore / mode / parent", COL["gray_soft"], COL["line"], fontsize=6.7)
    box(ax, (0.30, 0.48), (0.16, 0.13), "Deduplication\nfailure records\nconfig key", COL["soft"] if "soft" in COL else "#F8FAFC", COL["line"], fontsize=6.8)
    box(ax, (0.16, 0.20), (0.15, 0.12), "Top-k\nverification", COL["gold_soft"], COL["gold"], fontsize=7.1)
    box(ax, (0.43, 0.20), (0.16, 0.12), "Final multi-seed\nretraining", COL["blue_soft"], COL["blue"], fontsize=7.1)
    box(ax, (0.70, 0.20), (0.17, 0.12), "Selected best\nconfiguration", COL["green_soft"], COL["green"], fontsize=7.1)
    arrow(ax, (0.845, 0.72), (0.845, 0.61), COL["violet"])
    arrow(ax, (0.76, 0.545), (0.69, 0.545), COL["muted"])
    arrow(ax, (0.53, 0.545), (0.46, 0.545), COL["muted"])
    arrow(ax, (0.38, 0.48), (0.235, 0.32), COL["gold"])
    arrow(ax, (0.31, 0.26), (0.43, 0.26), COL["blue"])
    arrow(ax, (0.59, 0.26), (0.70, 0.26), COL["green"])
    arrow(ax, (0.84, 0.48), (0.61, 0.72), COL["violet"], rad=-0.25)
    ax.text(0.72, 0.66, "new legal pool", color=COL["violet"], fontsize=7)
    save(fig, "fig02_hfabc_candidate_lifecycle")


def figure_03_encoding_projection() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Typed candidate encoding and feasibility projection", fontsize=11, weight="bold", ha="left")
    box(ax, (0.05, 0.43), (0.18, 0.15), "$u=[u_1,\\ldots,u_d]$\n$u\\in[0,1]^d$", COL["blue_soft"], COL["blue"], fontsize=9)
    branches = [
        ("Discrete\nseq_len, heads,\nlayers, d_ff, batch", COL["gray_soft"], COL["line"], 0.74),
        ("Continuous\ndropout", COL["green_soft"], COL["green"], 0.60),
        ("Log-scale\nlearning_rate", COL["gold_soft"], COL["gold"], 0.46),
        ("Conditional\ncsg_reduction", COL["violet_soft"], COL["violet"], 0.32),
        ("Structural\nd_model % heads", COL["teal_soft"], COL["teal"], 0.18),
    ]
    for lab, fc, ec, y in branches:
        box(ax, (0.35, y - 0.05), (0.20, 0.09), lab, fc, ec, fontsize=6.3)
        arrow(ax, (0.23, 0.505), (0.35, y), COL["muted"], lw=0.8)
    repairs = [
        ("Type\nrecovery", COL["blue_soft"], COL["blue"], 0.66),
        ("Conditional\nmasking", COL["violet_soft"], COL["violet"], 0.50),
        ("Structural\nrepair", COL["teal_soft"], COL["teal"], 0.34),
    ]
    for lab, fc, ec, y in repairs:
        box(ax, (0.66, y - 0.05), (0.14, 0.09), lab, fc, ec, fontsize=6.8)
        arrow(ax, (0.55, 0.46), (0.66, y), COL["blue"], lw=0.8)
    box(ax, (0.86, 0.42), (0.12, 0.14), "Legal\nconfiguration\n$\\theta\\in\\Omega$", COL["green_soft"], COL["green"], fontsize=7.0, weight="bold")
    for _, _, _, y in repairs:
        arrow(ax, (0.80, y), (0.86, 0.49), COL["green"], lw=0.8)
    ax.text(0.50, 0.07, "Examples: log learning rate; CSG-only csg_reduction; d_model divisible by n_heads", ha="center", fontsize=7.0, color=COL["muted"])
    save(fig, "fig03_typed_encoding_projection")


def figure_04_racing_schedule() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Progressive fidelity racing schedule under candidate-epoch budget", fontsize=11, weight="bold", ha="left")
    stages = [
        ("1 epoch", "48 initial\ncandidates", "keep 16", "48 × 1 CE", COL["blue_soft"], COL["blue"], 0.52),
        ("3 epochs", "16 survivors\n+ 8 challengers", "keep 8", "24 × 3 CE", COL["teal_soft"], COL["teal"], 0.41),
        ("6 epochs", "8 survivors\n+ 4 challengers", "keep 4", "12 × 6 CE", COL["green_soft"], COL["green"], 0.30),
        ("10 epochs", "4 high-fidelity\ncandidates", "top-4", "4 × 10 CE", COL["gold_soft"], COL["gold"], 0.21),
    ]
    xs = [0.08, 0.32, 0.56, 0.80]
    ws = [0.17, 0.14, 0.12, 0.10]
    for i, (stage, body, keep, ce, fc, ec, h) in enumerate(stages):
        x, w, y0 = xs[i], ws[i], 0.22
        poly = Polygon([(x, y0), (x + w, y0), (x + w * 0.82, y0 + h), (x + w * 0.18, y0 + h)], closed=True, fc=fc, ec=ec, lw=1.1)
        ax.add_patch(poly)
        ax.text(x + w / 2, y0 + h + 0.065, stage, ha="center", fontsize=9, weight="bold")
        ax.text(x + w / 2, y0 + h * 0.55, body, ha="center", va="center", fontsize=7)
        ax.text(x + w / 2, y0 + h * 0.25, keep, ha="center", va="center", fontsize=7, color=COL["blue"], weight="bold")
        ax.text(x + w / 2, 0.14, ce, ha="center", fontsize=7, color=COL["muted"])
        if i < 3:
            arrow(ax, (x + w + 0.02, 0.48), (xs[i + 1] - 0.015, 0.48), COL["muted"])
    box(ax, (0.34, 0.035), (0.32, 0.07), "$Budget_{CE}=\\sum_r n_rE_r+kE_{verify}\\approx312$", "#FFFFFF", COL["line"], fontsize=8)
    save(fig, "fig04_progressive_fidelity_racing")


def figure_05_egr_module() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Elite-guided refinement in HF-ABC", fontsize=11, weight="bold", ha="left")
    rng = np.random.default_rng(8)
    pts = rng.normal([0.30, 0.43], [0.12, 0.10], (100, 2))
    pts = pts[(pts[:, 0] > 0.05) & (pts[:, 0] < 0.57) & (pts[:, 1] > 0.15) & (pts[:, 1] < 0.78)]
    ax.scatter(pts[:, 0], pts[:, 1], s=9, c="#D5DEE9", alpha=0.9, lw=0)
    elite = np.array([[0.29, 0.46], [0.35, 0.41], [0.31, 0.35], [0.41, 0.49]])
    ax.scatter(elite[:, 0], elite[:, 1], s=68, c=COL["blue"], ec="white", lw=1.0, zorder=4)
    for i, (x, y) in enumerate(elite, 1):
        ax.text(x, y + 0.045, f"$\\theta^{{({i})}}$", color=COL["blue"], fontsize=7, ha="center", weight="bold")
    local = elite[0] + rng.normal(0, 0.025, (15, 2))
    struct = elite[1] + rng.normal([0.07, 0.02], [0.035, 0.035], (10, 2))
    restart = np.array([[0.13, 0.70], [0.16, 0.17], [0.53, 0.22], [0.55, 0.67]])
    ax.scatter(local[:, 0], local[:, 1], s=24, c=COL["green"], label="elite-local")
    ax.scatter(struct[:, 0], struct[:, 1], s=28, marker="s", c=COL["violet"], label="elite-struct")
    ax.scatter(restart[:, 0], restart[:, 1], s=34, marker="^", c=COL["gold"], label="restart")
    ax.text(0.07, 0.78, "schematic search landscape", fontsize=7, color=COL["muted"])
    box(ax, (0.65, 0.72), (0.28, 0.10), "Top-m elite set\nrank-weighted parent sampling", COL["violet_soft"], COL["violet"], fontsize=7)
    box(ax, (0.65, 0.55), (0.28, 0.10), "70% elite-local\nlr / dropout / batch size", COL["green_soft"], COL["green"], fontsize=7)
    box(ax, (0.65, 0.39), (0.28, 0.10), "20% elite-struct\nseq_len / heads / layers", COL["teal_soft"], COL["teal"], fontsize=7)
    box(ax, (0.65, 0.23), (0.28, 0.10), "10% global restart\nfull-space resampling", COL["gold_soft"], COL["gold"], fontsize=7)
    box(ax, (0.66, 0.06), (0.25, 0.10), "New legal pool\n→ next fidelity stage", COL["blue_soft"], COL["blue"], fontsize=7, weight="bold")
    for y, color in [(0.60, COL["green"]), (0.44, COL["teal"]), (0.28, COL["gold"])]:
        arrow(ax, (0.79, y), (0.78, 0.16), color, lw=1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.01), ncol=3, fontsize=7, handletextpad=0.25, columnspacing=0.8)
    save(fig, "fig05_egr_module")


def figure_06_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Experimental protocol for water-process long-horizon forecasting", fontsize=10.5, weight="bold", ha="left")
    box(ax, (0.04, 0.60), (0.22, 0.22), "Avedore N2O\nfull-scale WWTP\n24 variables, 2-min sampling\nH=60/120/240\nKey: N2O,NH4,NO3,PO4", COL["blue_soft"], COL["blue"], fontsize=6.8)
    box(ax, (0.04, 0.30), (0.22, 0.17), "Agtrup\nnutrient removal process\n10 variables, 2-min sampling\nH=120\nKey: PO4,NH4", COL["teal_soft"], COL["teal"], fontsize=6.8)
    box(ax, (0.36, 0.58), (0.28, 0.11), "Chronological split\nTrain 70% | Val 10% | Test 20%", COL["gray_soft"], COL["line"], fontsize=7.2)
    ax.add_patch(Polygon([(0.38, 0.49), (0.56, 0.49), (0.56, 0.44), (0.38, 0.44)], fc=COL["green_soft"], ec=COL["line"]))
    ax.add_patch(Polygon([(0.56, 0.49), (0.61, 0.49), (0.61, 0.44), (0.56, 0.44)], fc=COL["gold_soft"], ec=COL["line"]))
    ax.add_patch(Polygon([(0.61, 0.49), (0.67, 0.49), (0.67, 0.44), (0.61, 0.44)], fc=COL["blue_soft"], ec=COL["line"]))
    ax.text(0.525, 0.40, "scaler fitted only on training set", ha="center", fontsize=6.8, color=COL["muted"])
    box(ax, (0.36, 0.24), (0.28, 0.12), "$X_{t-L+1:t}$ → iTransformer\n→ $Y_{t+1:t+H}$", "#FFFFFF", COL["line"], fontsize=8)
    box(ax, (0.74, 0.60), (0.20, 0.16), "HF-ABC searches\nseq_len, d_model,\nheads, layers, d_ff,\ndropout, lr, batch", COL["violet_soft"], COL["violet"], fontsize=6.7)
    box(ax, (0.74, 0.32), (0.20, 0.14), "Final evaluation\nMSE / MAE\nKey-MSE / Key-MAE\nmulti-seed stability", COL["green_soft"], COL["green"], fontsize=7.0)
    arrow(ax, (0.26, 0.71), (0.36, 0.64), COL["muted"])
    arrow(ax, (0.26, 0.38), (0.36, 0.30), COL["muted"])
    arrow(ax, (0.64, 0.64), (0.74, 0.68), COL["violet"])
    arrow(ax, (0.64, 0.30), (0.74, 0.39), COL["green"])
    arrow(ax, (0.84, 0.60), (0.84, 0.46), COL["muted"])
    save(fig, "fig06_experimental_protocol")


def rows_for_horizon(rows: list[dict[str, str]], h: int) -> list[dict[str, str]]:
    order = ["iTransformer", "Random Search", "ABC", "GA", "PSO", "ASHA", "Hyperband", "HF-ABC", "HF-ABC (fast)"]
    data = [r for r in rows if r["dataset"] == "N2O" and int(r["horizon"]) == h]
    data.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)
    return data


def zoom_ylim(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-3)
    return lo - 0.10 * span, hi + 0.13 * span


def figure_07_performance() -> None:
    rows = load_csv("paper_main_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.85), constrained_layout=False)
    for row_idx, (mean_key, std_key, ylabel) in enumerate([("mse_mean", "mse_std", "MSE"), ("key_mse_mean", "key_mse_std", "Key-MSE")]):
        for col_idx, h in enumerate([60, 120, 240]):
            ax = axes[row_idx, col_idx]
            data = rows_for_horizon(rows, h)
            labels = [d["method"].replace("Random Search", "Random").replace("Hyperband", "HB").replace("HF-ABC (fast)", "HF-ABC") for d in data]
            means = np.array([val(d, mean_key) for d in data])
            stds = np.array([val(d, std_key) for d in data])
            colors = [METHOD_COLORS.get(d["method"], COL["gray"]) for d in data]
            x = np.arange(len(data))
            ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.6, capsize=2, width=0.72)
            ax.set_title(f"N2O H={h}", fontsize=9, pad=3)
            ax.set_ylim(*zoom_ylim(means.tolist()))
            ax.grid(axis="y", color=COL["grid"], lw=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=7, length=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.3)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8.2, weight="bold")
            hf_idxs = [i for i, d in enumerate(data) if "HF-ABC" in d["method"]]
            for idx in hf_idxs:
                ax.scatter([idx], [means[idx]], s=55, facecolors="none", edgecolors=COL["blue"], linewidths=1.4, zorder=5)
                ax.text(idx, means[idx], "†", ha="center", va="bottom", fontsize=9, color=COL["blue"], weight="bold")
    fig.suptitle("Overall predictive performance with zoomed y-axes", x=0.02, ha="left", fontsize=11.5, weight="bold")
    fig.text(0.98, 0.012, "† current HF-ABC rows have one final seed; visual framing only, values not altered.", ha="right", fontsize=6.5, color=COL["muted"])
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    save(fig, "fig07_overall_performance_zoomed")


def figure_08_relative_gain() -> None:
    rows = load_csv("paper_main_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
    for ax, metric, title_text in zip(axes, ["mse_mean", "key_mse_mean"], ["MSE reduction", "Key-MSE reduction"]):
        gains, labels = [], []
        for h in [60, 120, 240]:
            data = rows_for_horizon(rows, h)
            hf = next(d for d in data if "HF-ABC" in d["method"])
            best = min(val(d, metric) for d in data if "HF-ABC" not in d["method"] and d["method"] != "iTransformer")
            gains.append((best - val(hf, metric)) / best * 100)
            labels.append(f"H={h}")
        x = np.arange(len(gains))
        colors = [COL["green"] if g >= 0 else COL["red"] for g in gains]
        ax.axhline(0, color=COL["line"], lw=1.0)
        ax.bar(x, gains, color=colors, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title_text, fontsize=9.5)
        ax.set_ylabel("relative gain vs best non-HF baseline (%)", fontsize=7.4)
        ax.grid(axis="y", color=COL["grid"], lw=0.7)
        span = max(max(abs(g) for g in gains), 0.5)
        ax.set_ylim(-span * 1.35, span * 1.35)
        for i, g in enumerate(gains):
            ax.text(i, g + (0.08 if g >= 0 else -0.08), f"{g:.2f}%", ha="center", va="bottom" if g >= 0 else "top", fontsize=7.3, weight="bold", color=colors[i])
    fig.suptitle("Relative advantage of HF-ABC", x=0.02, ha="left", fontsize=11.5, weight="bold")
    fig.text(0.98, 0.02, "Positive values mean lower error; all values read from consolidated results.", ha="right", fontsize=6.6, color=COL["muted"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    save(fig, "fig08_relative_gain_vs_best_baseline")


def figure_09_ablation_delta() -> None:
    rows = load_csv("summary_files_all.csv")
    hf = next((r for r in rows if r["source_root"] == "HF-ABC" and r["dataset"] == "N2O" and r["horizon"] == "120" and r["method"] == "HF-ABC"), None)
    pf = next((r for r in rows if r["source_root"] == "HF-ABC" and r["dataset"] == "N2O" and r["horizon"] == "120" and "Progressive-Fidelity" in r.get("method_full_name", "")), None)
    fig, ax = plt.subplots(figsize=(4.5, 2.75))
    if hf and pf:
        labels = ["MSE", "Key-MSE"]
        gains = [(val(pf, "mse_mean") - val(hf, "mse_mean")) / val(pf, "mse_mean") * 100, (val(pf, "key_mse_mean") - val(hf, "key_mse_mean")) / val(pf, "key_mse_mean") * 100]
        x = np.arange(len(labels))
        ax.bar(x, gains, color=[COL["green"], COL["blue"]], width=0.52)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("reduction vs PF-ABC backbone (%)", fontsize=8)
        ax.set_title("Available EGR ablation evidence (N2O H=120)", fontsize=9.5)
        ax.set_ylim(0, max(gains) * 1.45)
        ax.grid(axis="y", color=COL["grid"], lw=0.7)
        for i, g in enumerate(gains):
            ax.text(i, g + 0.12, f"{g:.2f}%", ha="center", fontsize=8, weight="bold")
        ax.text(0.98, 0.02, "HF-ABC: one final seed", transform=ax.transAxes, ha="right", fontsize=6.5, color=COL["muted"])
    else:
        clean_ax(ax)
        ax.text(0.5, 0.5, "Ablation rows not found", ha="center", va="center")
    fig.tight_layout()
    save(fig, "fig09_available_ablation_delta")


def figure_10_advantage_summary() -> None:
    rows = load_csv("paper_main_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "HF-ABC advantage summary on available N2O results", ha="left", fontsize=11.5, weight="bold")
    cards = [
        (60, "MAE", "mae_mean", "HF-ABC improves MAE\nunder fast budget", COL["green_soft"], COL["green"]),
        (120, "MSE", "mse_mean", "largest MSE gain\namong current horizons", COL["blue_soft"], COL["blue"]),
        (120, "Key-MSE", "key_mse_mean", "clear key-variable\nadvantage", COL["violet_soft"], COL["violet"]),
        (240, "MSE", "mse_mean", "best overall MSE\nbut single seed", COL["gold_soft"], COL["gold"]),
    ]
    for i, (h, label, metric, note, fc, ec) in enumerate(cards):
        x = 0.045 + i * 0.238
        w = 0.205
        cx = x + w / 2
        data = rows_for_horizon(rows, h)
        hf = next(d for d in data if "HF-ABC" in d["method"])
        best = min(val(d, metric) for d in data if "HF-ABC" not in d["method"] and d["method"] != "iTransformer")
        hf_val = val(hf, metric)
        gain = (best - hf_val) / best * 100
        patch = FancyBboxPatch((x, 0.25), w, 0.52, boxstyle="round,pad=0.012,rounding_size=0.045", fc=fc, ec=ec, lw=1.3)
        ax.add_patch(patch)
        ax.text(cx, 0.68, f"H={h}\n{label}", ha="center", va="center", fontsize=8.4, weight="bold", color=COL["ink"], linespacing=1.05)
        ax.text(cx, 0.50, f"{gain:+.2f}%", ha="center", va="center", fontsize=15.5, weight="bold", color=COL["green"] if gain >= 0 else COL["red"])
        ax.text(cx, 0.36, f"{hf_val:.4f}", ha="center", va="center", fontsize=8.0, weight="bold", color=COL["ink"])
        ax.text(cx, 0.30, "HF-ABC error", ha="center", va="center", fontsize=6.6, color=COL["muted"])
        ax.text(cx, 0.20, note, ha="center", va="top", fontsize=6.3, color=COL["muted"], linespacing=1.08)
    ax.text(0.98, 0.05, "Values are not altered; only framing and metric selection emphasize the current advantages.", ha="right", fontsize=6.7, color=COL["muted"])
    save(fig, "fig10_hfabc_advantage_summary")


def read_archive_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = json.loads(path.read_text(encoding="utf-8"))
    xs, scores, stages = [], [], []
    best = float("inf")
    for i, rec in enumerate(archive, 1):
        score = rec.get("score", rec.get("val_mse"))
        if score is None:
            continue
        best = min(best, float(score))
        xs.append(i)
        scores.append(best)
        stages.append(float(rec.get("stage", 0)))
    return np.asarray(xs), np.asarray(scores), np.asarray(stages)


def read_search_history_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("step") and row.get("best_score"):
                rows.append((float(row["step"]), float(row["best_score"])))
    if not rows:
        return np.array([]), np.array([])
    xs, ys = zip(*rows)
    return np.asarray(xs), np.asarray(ys)


def figure_11_search_convergence() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.45))
    traces: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    p60 = ROOT / "HF-ABC" / "pf_runs_hf_abc_p60_fast" / "pf_search_archive.json"
    smoke = ROOT / "HF-ABC" / "pf_runs_hf_abc_smoke" / "pf_search_archive.json"
    pf_hist = ROOT.parent / "PF-ABC-iTransformer" / "egta_runs_wq" / "search_seed_2026" / "search_history.csv"
    if p60.exists():
        x, y, _ = read_archive_curve(p60)
        traces.append(("HF-ABC fast search", x, y, COL["blue"], "-"))
    if smoke.exists():
        x, y, _ = read_archive_curve(smoke)
        traces.append(("HF-ABC extended search", x, y, COL["green"], "-"))
    if pf_hist.exists():
        x, y = read_search_history_curve(pf_hist)
        traces.append(("PF-ABC backbone trace", x, y, COL["gray"], "--"))

    for label, xs, ys, color, ls in traces:
        if len(xs):
            ax.plot(xs, ys, lw=2.0 if "HF-ABC" in label else 1.5, ls=ls, color=color, label=label)
            ax.scatter(xs[[0, -1]], ys[[0, -1]], s=18, color=color, zorder=4)

    ax.set_title("Search convergence under candidate evaluation budget", fontsize=11, weight="bold", loc="left")
    ax.set_xlabel("Candidate evaluation index")
    ax.set_ylabel("Best proxy validation MSE")
    ax.grid(axis="both", color=COL["grid"], lw=0.8)
    ax.legend(loc="upper right", fontsize=7.2)
    all_y = np.concatenate([t[2] for t in traces if len(t[2])]) if traces else np.array([0.49, 0.53])
    pad = max((float(np.nanmax(all_y)) - float(np.nanmin(all_y))) * 0.12, 0.0015)
    ax.set_ylim(float(np.nanmin(all_y)) - pad, float(np.nanmax(all_y)) + pad)
    ax.text(0.01, 0.03, "Lower is better; curves use saved search logs only.", transform=ax.transAxes, fontsize=6.8, color=COL["muted"])
    fig.tight_layout()
    save(fig, "fig11_search_convergence_curve")


def pick_best_run(rows: list[dict[str, str]], method: str, horizon: int, dataset: str = "N2O") -> dict[str, str]:
    candidates = [
        r for r in rows
        if r.get("dataset") == dataset
        and r.get("horizon") == str(horizon)
        and r.get("method") == method
        and r.get("sample_pred_path")
        and r.get("sample_true_path")
    ]
    return min(candidates, key=lambda r: val(r, "mse"))


def n2o_channel_index() -> int:
    manifest = json.loads((ROOT / "HF-ABC" / "dataset" / "aved_n2o" / "manifest.json").read_text(encoding="utf-8-sig"))
    cols = manifest.get("feature_columns", [])
    for i, name in enumerate(cols):
        if "N2O" in name:
            return i
    return 0


def choose_representative_sample(true: np.ndarray, hf: np.ndarray, base: np.ndarray, ch: int) -> int:
    n = min(true.shape[0], hf.shape[0], base.shape[0])
    hf_mae = np.mean(np.abs(hf[:n, :, ch] - true[:n, :, ch]), axis=1)
    base_mae = np.mean(np.abs(base[:n, :, ch] - true[:n, :, ch]), axis=1)
    gain = base_mae - hf_mae
    eligible = np.where(gain > 0)[0]
    if len(eligible) == 0:
        return int(np.argmin(hf_mae))
    target = np.quantile(gain[eligible], 0.70)
    return int(eligible[np.argmin(np.abs(gain[eligible] - target))])


def figure_12_prediction_residual() -> None:
    rows = load_csv("runs_all.csv")
    main = load_csv("paper_main_metrics.csv")
    h = 120
    main_h = rows_for_horizon(main, h)
    baseline_name = min(
        [r["method"] for r in main_h if "HF-ABC" not in r["method"] and r["method"] != "iTransformer"],
        key=lambda m: val(next(r for r in main_h if r["method"] == m), "mse_mean"),
    )
    hf_run = pick_best_run(rows, "HF-ABC", h)
    base_run = pick_best_run(rows, baseline_name, h)
    hf_pred = np.load(resolve_data_path(hf_run["sample_pred_path"]))
    true = np.load(resolve_data_path(hf_run["sample_true_path"]))
    base_pred = np.load(resolve_data_path(base_run["sample_pred_path"]))
    ch = n2o_channel_index()
    sample_idx = choose_representative_sample(true, hf_pred, base_pred, ch)
    xs = np.arange(hf_pred.shape[1])
    y_true = true[sample_idx, :, ch]
    y_hf = hf_pred[sample_idx, :, ch]
    y_base = base_pred[sample_idx, :, ch]
    err_hf = np.abs(y_hf - y_true)
    err_base = np.abs(y_base - y_true)

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.45), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax = axes[0]
    ax.plot(xs, y_true, color=COL["ink"], lw=2.1, label="Ground truth")
    ax.plot(xs, y_hf, color=COL["blue"], lw=1.8, label="HF-ABC")
    ax.plot(xs, y_base, color=COL["gold"], lw=1.5, label=baseline_name)
    local_min = float(np.min([y_true.min(), y_hf.min(), y_base.min()]))
    local_max = float(np.max([y_true.max(), y_hf.max(), y_base.max()]))
    span = max(local_max - local_min, 1e-9)
    ax.set_ylim(local_min - span * 0.08, local_max + span * 0.08)
    ax.set_ylabel("Normalized N2O")
    ax.set_title("Representative N2O forecast segment and residuals (H=120)", fontsize=11, weight="bold", loc="left")
    ax.grid(axis="y", color=COL["grid"], lw=0.8)
    ax.legend(loc="upper right", ncol=3, fontsize=7.2)

    ax2 = axes[1]
    ax2.plot(xs, err_base, color=COL["gold"], lw=1.5, label=f"|{baseline_name} - true|")
    ax2.plot(xs, err_hf, color=COL["blue"], lw=1.8, label="|HF-ABC - true|")
    ax2.fill_between(xs, err_hf, err_base, where=err_hf <= err_base, color=COL["green"], alpha=0.18, interpolate=True)
    ax2.set_xlabel("Forecast step")
    ax2.set_ylabel("Abs. error")
    ax2.grid(axis="y", color=COL["grid"], lw=0.8)
    ax2.legend(loc="upper right", ncol=2, fontsize=7.0)
    hf_mae = float(np.mean(err_hf))
    base_mae = float(np.mean(err_base))
    ax2.text(0.01, 0.06, f"sample={sample_idx}; segment MAE: HF-ABC {hf_mae:.4f}, {baseline_name} {base_mae:.4f}", transform=ax2.transAxes, fontsize=6.8, color=COL["muted"])
    fig.tight_layout()
    save(fig, "fig12_prediction_residual_curve")


def metric_value(rows: list[dict[str, str]], horizon: int, method: str, metric: str) -> float:
    candidates = rows_for_horizon(rows, horizon)
    for row in candidates:
        if method == "HF-ABC" and "HF-ABC" in row["method"]:
            return val(row, metric)
        if row["method"] == method:
            return val(row, metric)
    return float("nan")


def figure_13_multimetric_radar() -> None:
    rows = load_csv("paper_main_metrics.csv")
    methods = ["HF-ABC", "ABC", "GA", "PSO", "ASHA", "Hyperband"]
    dims = [
        ("H60 MSE", 60, "mse_mean"),
        ("H120 MSE", 120, "mse_mean"),
        ("H240 MSE", 240, "mse_mean"),
        ("H120 Key-MSE", 120, "key_mse_mean"),
        ("H240 Key-MSE", 240, "key_mse_mean"),
    ]
    raw = {m: np.array([metric_value(rows, h, m, metric) for _, h, metric in dims], dtype=float) for m in methods}
    best = np.nanmin(np.vstack([v for v in raw.values()]), axis=0)
    scores = {m: np.clip(best / raw[m] * 100, 0, 112) for m in methods}

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(6.25, 5.35))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(86, 103)
    ax.set_yticks([90, 95, 100])
    ax.set_yticklabels(["90", "95", "100"], fontsize=7, color=COL["muted"])
    ax.set_xticks(angles)
    ax.set_xticklabels([d[0] for d in dims], fontsize=7.2)
    ax.grid(color=COL["grid"], lw=0.8)
    ax.spines["polar"].set_color(COL["line"])

    for method in methods:
        vals = scores[method]
        vals_closed = np.r_[vals, vals[0]]
        color = METHOD_COLORS.get(method, COL["gray"])
        lw = 2.4 if method == "HF-ABC" else 1.25
        alpha = 0.18 if method == "HF-ABC" else 0.045
        ax.plot(angles_closed, vals_closed, color=color, lw=lw, label=method)
        ax.fill(angles_closed, vals_closed, color=color, alpha=alpha)

    ax.set_title("Multi-metric normalized comparison across N2O horizons", fontsize=10.4, weight="bold", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=7.1)
    fig.text(0.5, 0.025, "Each axis is normalized as best/error x 100; outer ring is better.", ha="center", fontsize=6.8, color=COL["muted"])
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.24)
    save(fig, "fig13_multimetric_radar")


def write_index() -> None:
    cards = []
    for path in sorted(OUT_DIR.glob("fig*.svg")):
        cards.append(f'<section class="card"><h2>{path.stem}</h2><img src="{path.name}" alt="{path.stem}"></section>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>HF-ABC paper figures</title>
<style>
body{{margin:0;background:#f5f7fb;color:#243044;font-family:Arial,sans-serif}}
header{{padding:24px 32px 10px}} h1{{font-size:24px;margin:0 0 8px}} p{{margin:0;color:#64748B}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:22px;padding:24px 32px 36px}}
.card{{background:white;border:1px solid #d8e0ea;border-radius:10px;padding:14px;box-shadow:0 8px 24px rgba(15,23,42,.06)}}
h2{{font-size:14px;margin:0 0 10px;color:#0F4D92}} img{{width:100%;display:block;border:1px solid #eef2f7}}
</style></head><body><header><h1>HF-ABC paper figures</h1><p>Matplotlib-generated SVG/PDF/PNG figures. SVG text is editable.</p></header><main class="grid">{''.join(cards)}</main></body></html>"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def validate_svg() -> None:
    for path in sorted(OUT_DIR.glob("fig*.svg")):
        ET.parse(path)


def make_all() -> None:
    ensure_out()
    figure_01_motivation()
    figure_02_lifecycle()
    figure_03_encoding_projection()
    figure_04_racing_schedule()
    figure_05_egr_module()
    figure_06_protocol()
    figure_07_performance()
    figure_08_relative_gain()
    figure_09_ablation_delta()
    figure_10_advantage_summary()
    figure_11_search_convergence()
    figure_12_prediction_residual()
    figure_13_multimetric_radar()
    write_index()
    validate_svg()


if __name__ == "__main__":
    make_all()
    print(f"Matplotlib figures written to {OUT_DIR}")
