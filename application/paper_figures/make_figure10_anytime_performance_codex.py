from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "outputs"

TOTAL_CE = 312.0
VERIFY_EPOCHS = 20.0
DEFAULT_VAL_MSE = 0.6498344540596008

METHOD_ORDER = ["Random", "GA", "PSO", "ABC", "ASHA", "Hyperband", "HF-ABC"]
METHOD_COLORS = {
    "Random": "#7A8591",
    "GA": "#59A14F",
    "PSO": "#4FA9A7",
    "ABC": "#F28E2B",
    "ASHA": "#B07AA1",
    "Hyperband": "#E15759",
    "HF-ABC": "#174A8B",
}
METHOD_MARKERS = {
    "Random": "o",
    "GA": "s",
    "PSO": "^",
    "ABC": "D",
    "ASHA": "P",
    "Hyperband": "h",
    "HF-ABC": "D",
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

TRIALS_CSV = (
    PROJECT_DIR.parent
    / "PF-ABC-iTransformer"
    / "egta_runs_wq"
    / "search_seed_2026"
    / "search_trials.csv"
)
SUMMARY_FILES = {
    "Random": PROJECT_DIR.parent
    / "PF-ABC-iTransformer"
    / "egta_runs_wq"
    / "water_quality_120_iTransformer_CSG_random_search.json",
    "GA": PROJECT_DIR.parent
    / "PF-ABC-iTransformer"
    / "egta_runs_wq"
    / "water_quality_120_iTransformer_CSG_ga.json",
    "PSO": PROJECT_DIR.parent
    / "PF-ABC-iTransformer"
    / "egta_runs_wq"
    / "water_quality_120_iTransformer_CSG_pso.json",
    "ABC": PROJECT_DIR.parent
    / "PF-ABC-iTransformer"
    / "egta_runs_wq"
    / "water_quality_120_iTransformer_CSG_abc.json",
    "ASHA": PROJECT_DIR
    / "HF-ABC"
    / "pf_runs_wq_two_hpo"
    / "water_quality_120_iTransformer_CSG_asha.json",
    "Hyperband": PROJECT_DIR
    / "HF-ABC"
    / "pf_runs_wq_two_hpo"
    / "water_quality_120_iTransformer_CSG_hyperband.json",
}
HF_ARCHIVE_JSON = (
    PROJECT_DIR
    / "HF-ABC"
    / "pf_runs_hf_abc_smoke"
    / "pf_search_archive.json"
)
HF_SUMMARY_JSON = (
    PROJECT_DIR
    / "HF-ABC"
    / "pf_runs_hf_abc_smoke"
    / "water_quality_120_iTransformer_CSG_pf_abc.json"
)

PHASE_PREFIX = {
    "Random": "single_stage_random",
    "GA": "single_stage_ga",
    "PSO": "single_stage_pso",
    "ABC": "single_stage_abc",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_min(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        raise ValueError("No finite values found.")
    return min(vals)


def summary_best(method: str) -> float:
    summary = load_json(SUMMARY_FILES[method])
    return float(summary["search_summary"]["best_val_mse_mean"])


def validate_best(method: str, values: list[float], tol: float = 1e-8) -> None:
    observed = finite_min(values)
    expected = summary_best(method)
    if abs(observed - expected) > tol:
        raise ValueError(
            f"{method} trajectory best {observed:.12f} does not match summary {expected:.12f}."
        )


def prepend_default(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    if xs and xs[0] == 0:
        return xs, ys
    return [0.0, *xs], [DEFAULT_VAL_MSE, *ys]


def best_so_far_curve(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        raise ValueError("Empty trajectory.")

    xs: list[float] = []
    ys: list[float] = []
    best = math.inf
    step_cost = TOTAL_CE / len(values)
    for idx, val in enumerate(values, start=1):
        best = min(best, val)
        xs.append(idx * step_cost)
        ys.append(best)
    return prepend_default(xs, ys)


def load_classic_curves() -> dict[str, tuple[list[float], list[float]]]:
    with TRIALS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        trial_rows = list(csv.DictReader(f))

    # H=120 logs are stored as: ABC first, then Random/GA/PSO interleaved.
    classic_windows = {
        "ABC": trial_rows[:112],
        "Random": trial_rows[112:368],
        "GA": trial_rows[112:368],
        "PSO": trial_rows[112:368],
    }

    curves: dict[str, tuple[list[float], list[float]]] = {}
    for method, rows in classic_windows.items():
        prefix = PHASE_PREFIX[method]
        values = [
            float(row["val_mse"])
            for row in rows
            if row["phase"].startswith(prefix) and row.get("oom", "0") == "0"
        ]
        validate_best(method, values)
        curves[method] = best_so_far_curve(values)
    return curves


def archive_curve(method: str) -> tuple[list[float], list[float]]:
    search_summary = load_json(SUMMARY_FILES[method])["search_summary"]
    summary = search_summary["per_search_seed"][0]
    xs: list[float] = []
    ys: list[float] = []
    ce = 0.0
    best = DEFAULT_VAL_MSE

    for item in summary.get("archive", []):
        if item.get("status", "ok") != "ok":
            continue
        val = item.get("val_mse", item.get("score"))
        if val is None or not math.isfinite(float(val)):
            continue
        ce += float(item.get("fidelity", 1.0))
        best = min(best, float(val))
        xs.append(ce)
        ys.append(best)

    verify_epochs = float(summary.get("verify_epochs", VERIFY_EPOCHS))
    for item in summary.get("verified", []):
        val = item.get("val_mse", item.get("score"))
        if val is None or not math.isfinite(float(val)):
            continue
        ce += verify_epochs
        best = min(best, float(val))
        xs.append(ce)
        ys.append(best)

    if xs and abs(xs[-1] - TOTAL_CE) > 1e-8:
        scale = TOTAL_CE / xs[-1]
        xs = [x * scale for x in xs]

    if not ys or not math.isfinite(min(ys)):
        raise ValueError(f"{method} has no finite validation trajectory.")

    return prepend_default(xs, ys)


def hf_summary_curve() -> tuple[list[float], list[float]]:
    summary = load_json(HF_SUMMARY_JSON)["search_summary"]["per_search_seed"][0]
    xs: list[float] = []
    ys: list[float] = []
    ce = 0.0
    best = DEFAULT_VAL_MSE

    for stage in summary["stage_summaries"]:
        ce += float(stage["n_evaluated"]) * float(stage["fidelity"])
        best = min(best, float(stage["best_val_mse"]))
        xs.append(ce)
        ys.append(best)

    verified = summary.get("verified", [])
    if verified:
        ce += len(verified) * VERIFY_EPOCHS
        best = min(best, finite_min(float(item["val_mse"]) for item in verified))
        xs.append(ce)
        ys.append(best)

    if xs and abs(xs[-1] - TOTAL_CE) > 1e-8:
        scale = TOTAL_CE / xs[-1]
        xs = [x * scale for x in xs]

    return prepend_default(xs, ys)


def hf_archive_curve_or_summary() -> tuple[list[float], list[float]]:
    archive = load_json(HF_ARCHIVE_JSON)
    xs: list[float] = []
    ys: list[float] = []
    ce = 0.0
    best = DEFAULT_VAL_MSE

    for item in archive:
        if item.get("status", "ok") != "ok":
            continue
        val = item.get("val_mse", item.get("score"))
        if val is None or not math.isfinite(float(val)):
            continue
        ce += float(item.get("fidelity", 1.0))
        best = min(best, float(val))
        xs.append(ce)
        ys.append(best)

    if not ys or not math.isfinite(min(ys)):
        raise ValueError("HF-ABC archive has no finite candidate-level val_mse trajectory.")
    if abs(xs[-1] - TOTAL_CE) > 1e-8:
        raise ValueError(f"HF-ABC archive budget is {xs[-1]:.6f}, expected {TOTAL_CE:.6f}.")

    # The root pf_search_archive.json can be overwritten by another horizon/run.
    # For H=120, reject it unless it matches the method wrapper summary.
    summary = load_json(HF_SUMMARY_JSON)["search_summary"]
    expected = float(summary["selected_val_mse"])
    if abs(min(ys) - expected) > 0.02:
        return hf_summary_curve()

    return prepend_default(xs, ys)


def load_curves() -> dict[str, tuple[list[float], list[float]]]:
    curves = load_classic_curves()
    curves["ASHA"] = archive_curve("ASHA")
    curves["Hyperband"] = archive_curve("Hyperband")
    curves["HF-ABC"] = hf_archive_curve_or_summary()
    return curves


def write_curve_csv(curves: dict[str, tuple[list[float], list[float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig10_anytime_performance_data.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "horizon", "method", "candidate_epoch", "best_val_mse"])
        for method in METHOD_ORDER:
            xs, ys = curves[method]
            for x, y in zip(xs, ys):
                writer.writerow(["N2O", 120, method, f"{x:.6f}", f"{y:.12f}"])


def improvement_points(xs: list[float], ys: list[float], tol: float = 1e-12) -> tuple[list[float], list[float]]:
    if not xs:
        return [], []

    keep = [0]
    for idx in range(1, len(ys)):
        if ys[idx] < ys[idx - 1] - tol:
            keep.append(idx)

    if keep[-1] != len(xs) - 1:
        keep.append(len(xs) - 1)

    return [xs[idx] for idx in keep], [ys[idx] for idx in keep]


def smooth_monotone_path(xs: list[float], ys: list[float], points_per_segment: int = 26) -> tuple[list[float], list[float]]:
    px, py = improvement_points(xs, ys)
    if len(px) < 2:
        return px, py

    smooth_x: list[float] = [px[0]]
    smooth_y: list[float] = [py[0]]
    for x0, x1, y0, y1 in zip(px[:-1], px[1:], py[:-1], py[1:]):
        if x1 <= x0:
            continue
        for i in range(1, points_per_segment + 1):
            t = i / points_per_segment
            s = t * t * (3.0 - 2.0 * t)
            smooth_x.append(x0 + (x1 - x0) * t)
            smooth_y.append(y0 + (y1 - y0) * s)

    best = math.inf
    for idx, val in enumerate(smooth_y):
        best = min(best, val)
        smooth_y[idx] = best
    return smooth_x, smooth_y


def draw_figure(curves: dict[str, tuple[list[float], list[float]]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.6,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    zoom_xlim = (35.0, TOTAL_CE)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.28, 2.82),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.25},
    )
    all_y = [y for _, ys in curves.values() for y in ys]
    y_min = min(all_y)
    y_max = max(all_y)
    y_pad = max((y_max - y_min) * 0.065, 0.0020)
    archive_scale = y_min < 0.55
    zoom_ylim = (0.484, 0.624) if archive_scale else (0.6050, 0.6232)
    full_yticks = [0.49, 0.53, 0.57, 0.61, 0.65] if archive_scale else [0.606, 0.616, 0.626, 0.636, 0.646]
    zoom_yticks = [0.49, 0.52, 0.55, 0.58, 0.61] if archive_scale else [0.606, 0.610, 0.614, 0.618, 0.622]

    def plot_panel(ax: plt.Axes, zoom: bool = False) -> None:
        for method in METHOD_ORDER:
            xs, ys = curves[method]
            smooth_x, smooth_y = smooth_monotone_path(xs, ys)
            is_featured = method == "HF-ABC"
            lw = 2.05 if method == "HF-ABC" else 1.52 if method in {"ASHA", "Hyperband"} else 1.22
            alpha = 1.0 if is_featured else 0.86
            zorder = 6 if method == "HF-ABC" else 4 if method in {"ASHA", "Hyperband"} else 2

            ax.plot(
                smooth_x,
                smooth_y,
                color=METHOD_COLORS[method],
                linewidth=lw,
                linestyle=METHOD_LINESTYLES[method],
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )

        ax.set_xlim(0, TOTAL_CE)
        ax.grid(axis="y", color="#E4E9F0", linewidth=0.62)
        ax.grid(axis="x", color="#EEF2F6", linewidth=0.42)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.8, width=0.65, labelsize=7.1, pad=2.0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("Candidate epochs (CE)", fontsize=8.0, labelpad=3.0)
        if zoom:
            ax.set_xlim(*zoom_xlim)
            ax.set_ylim(*zoom_ylim)
            ax.set_xticks([50, 100, 150, 200, 250, 300])
            ax.set_yticks(zoom_yticks)
        else:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
            ax.set_yticks(full_yticks)

    plot_panel(axes[0], zoom=False)
    plot_panel(axes[1], zoom=True)

    axes[0].set_ylabel("Best-so-far validation MSE", fontsize=8.0, labelpad=4.0)
    axes[0].set_title("Full search budget", fontsize=8.4, fontweight="semibold", pad=4)
    axes[1].set_title("Late-budget zoom", fontsize=8.4, fontweight="semibold", pad=4)
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
            color=METHOD_COLORS[m],
            lw=2.05 if m == "HF-ABC" else 1.52 if m in {"ASHA", "Hyperband"} else 1.22,
            linestyle=METHOD_LINESTYLES[m],
            label=m,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=METHOD_ORDER,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.05),
        fontsize=6.8,
        handlelength=2.3,
        handletextpad=0.55,
        columnspacing=0.85,
    )

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.72, wspace=0.25)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(
            OUT_DIR / f"fig10_anytime_performance{suffix}",
            bbox_inches="tight",
            pad_inches=0.035,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    curves = load_curves()
    write_curve_csv(curves)
    draw_figure(curves)


if __name__ == "__main__":
    main()
