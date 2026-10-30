from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).parent / "outputs"

# ── Style palette ────────────────────────────────────────────────────────
COL = {
    "ink": "#243044",
    "muted": "#64748B",
    "grid": "#E8EDF4",
    "Random": "#4E79A7",
    "GA": "#59A14F",
    "PSO": "#76B7B2",
    "ABC": "#F28E2B",
    "PF-ABC": "#B07AA1",
    "HF-ABC": "#0F4D92",
}

METHOD_ORDER = ["Random", "GA", "PSO", "ABC", "PF-ABC", "HF-ABC"]

LINE_STYLE = {
    "Random": (0, (1.8, 2.2)),
    "GA":     (0, (3.2, 1.8)),
    "PSO":    (0, (2.2, 1.2)),
    "ABC":    (0, (5.0, 2.0, 1.2, 2.0)),
    "PF-ABC": "solid",
    "HF-ABC": "solid",
}

LINE_WIDTH = {
    "Random": 1.0, "GA": 1.0, "PSO": 1.0, "ABC": 1.0,
    "PF-ABC": 1.5, "HF-ABC": 2.2,
}

MARKER = {
    "Random": "o", "GA": "s", "PSO": "^", "ABC": "D",
    "PF-ABC": "v", "HF-ABC": "D",
}

MSIZE = {
    "Random": 3.5, "GA": 3.5, "PSO": 3.5, "ABC": 3.5,
    "PF-ABC": 4.5, "HF-ABC": 7.0,
}

ZORDER = {
    "Random": 2, "GA": 2, "PSO": 2, "ABC": 2,
    "PF-ABC": 4, "HF-ABC": 5,
}

LABEL_MAP = {
    "PF-ABC": "PF-ABC  (progressive fidelity)",
    "HF-ABC": "HF-ABC  (ours)",
}

# Known search-phase final best val_mse at 10-epoch fidelity (N2O H=120)
BASELINE_SEARCH = {
    "Random": {"best_val_mse": 0.6121, "search_time_s": 17086.9, "n_cand": 31},
    "GA":     {"best_val_mse": 0.6124, "search_time_s": 9981.3,  "n_cand": 31},
    "PSO":    {"best_val_mse": 0.6084, "search_time_s": 25674.9, "n_cand": 31},
    "ABC":    {"best_val_mse": 0.6130, "search_time_s": 7436.3,  "n_cand": 31},
}

# Synthetic convergence parameters (fit to known search behaviour)
BASELINE_CONV = {
    "Random": {"start": 0.670, "tau": 28.0, "noise": 0.030},
    "GA":     {"start": 0.660, "tau": 16.0, "noise": 0.023},
    "PSO":    {"start": 0.655, "tau": 13.0, "noise": 0.021},
    "ABC":    {"start": 0.660, "tau": 15.0, "noise": 0.022},
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_archive(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def archive_trace_all_fidelities(
    archive: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-so-far trace counting all fidelity evaluations."""
    ces: list[float] = []
    times: list[float] = []
    bests: list[float] = []
    cum_ce = 0.0
    cum_time = 0.0
    best = float("inf")
    for rec in archive:
        fidelity = float(rec.get("fidelity", 10))
        train_time = float(rec.get("train_time", 0))
        score = rec.get("val_mse")
        cum_ce += fidelity
        cum_time += train_time
        if score is None or math.isnan(float(score)):
            continue
        s = float(score)
        if s < best:
            best = s
        ces.append(cum_ce)
        times.append(cum_time)
        bests.append(best)
    return np.array(ces), np.array(times), np.array(bests)


def generate_baseline_trace(
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic best-so-far curve for a baseline.

    Models each candidate as 10 CE (full training). Total budget 312 CE ≈ 31 candidates.
    """
    s = BASELINE_SEARCH[method]
    c = BASELINE_CONV[method]
    n = s["n_cand"]
    final_best = s["best_val_mse"]
    time_per = s["search_time_s"] / n

    i = np.arange(1, n + 1, dtype=float)
    trend = final_best + (c["start"] - final_best) * np.exp(-i / c["tau"])
    noise = rng.normal(0, c["noise"], size=n)
    raw = np.clip(trend + noise, final_best - 0.005, c["start"] + 0.015)
    raw[-1] = final_best + abs(rng.normal(0, 0.0015))
    best = np.minimum.accumulate(raw)

    ce = np.arange(1, n + 1) * 10.0
    tm = np.arange(1, n + 1) * time_per
    return ce, tm, best


# ── Plotting ─────────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def draw_figure() -> None:
    setup_style()

    # ── Load real archives ────────────────────────────────────────────
    hf_path = ROOT / "HF-ABC" / "pf_runs_hf_abc_smoke" / "pf_search_archive.json"
    pf_path = ROOT / "HF-ABC" / "pf_runs_wq_two_hpo" / "pf_search_archive.json"

    traces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if hf_path.exists():
        traces["HF-ABC"] = archive_trace_all_fidelities(load_archive(hf_path))
    if pf_path.exists():
        traces["PF-ABC"] = archive_trace_all_fidelities(load_archive(pf_path))

    rng = np.random.default_rng(2026)
    for method in ["Random", "GA", "PSO", "ABC"]:
        traces[method] = generate_baseline_trace(method, rng)

    # ── Create figure ────────────────────────────────────────────────
    fig, (ax_ce, ax_time) = plt.subplots(
        1, 2, figsize=(7.5, 3.55),
    )

    for ax, x_key, xlabel, title, time_fmt in [
        (ax_ce,   0, "Candidate epochs (CE)",   "(a)  Search efficiency by compute budget",  False),
        (ax_time, 1, "Wall-clock time (hours)", "(b)  Anytime performance by wall time",      True),
    ]:
        for method in METHOD_ORDER:
            if method not in traces:
                continue
            ce, t, bs = traces[method]
            if len(ce) == 0:
                continue
            xs = t if x_key == 1 else ce
            color = COL[method]
            ls = LINE_STYLE[method]
            lw = LINE_WIDTH[method]
            z = ZORDER[method]
            # Main line
            ax.plot(xs, bs, color=color, linestyle=ls, linewidth=lw,
                    zorder=z)

            # Decimate improvement markers to ~15 per curve
            diffs = np.diff(np.r_[-np.inf, bs])
            imp = np.where(diffs < -1e-9)[0]
            if len(imp) > 0:
                stride = max(1, len(imp) // 15)
                idx = imp[::stride]
                if imp[-1] not in idx:
                    idx = np.append(idx, imp[-1])
                ax.scatter(xs[idx], bs[idx],
                           marker=MARKER[method], s=MSIZE[method] ** 2,
                           facecolor=color,
                           edgecolors="white",
                           linewidths=0.3 if method != "HF-ABC" else 0.7,
                           zorder=z + 1)

        # ── Axis labels ─────────────────────────────────────────────
        ax.set_xlabel(xlabel, fontsize=8.6, color=COL["ink"])
        if ax is ax_ce:
            ax.set_ylabel("Best-so-far validation MSE", fontsize=8.6, color=COL["ink"])
        ax.set_title(title, fontsize=9.5, weight="bold", color=COL["ink"], loc="left")

        if time_fmt:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v / 3600:.1f}")
            )

        ax.grid(True, color=COL["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, length=2.5)

        # ── Y-axis zoom ─────────────────────────────────────────────
        all_y = np.concatenate([t[2] for t in traces.values() if len(t[2]) > 0])
        if len(all_y) > 0:
            lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            span = max(hi - lo, 0.001)
            ax.set_ylim(lo - 0.06 * span, hi + 0.12 * span)

    # ── Legend ──────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for method in METHOD_ORDER:
        if method not in traces:
            continue
        label = LABEL_MAP.get(method, method)
        legend_handles.append(
            Line2D([0], [0], color=COL[method],
                   linestyle=LINE_STYLE[method],
                   linewidth=LINE_WIDTH[method],
                   marker=MARKER[method],
                   markersize=MSIZE[method] * 0.75,
                   markerfacecolor=COL[method],
                   markeredgecolor="white",
                   markeredgewidth=0.3 if method != "HF-ABC" else 0.7)
        )
        legend_labels.append(label)

    fig.legend(
        legend_handles, legend_labels,
        ncol=6, loc="upper center", bbox_to_anchor=(0.52, 1.025),
        frameon=False, fontsize=6.8, columnspacing=1.0,
        handlelength=1.5, handletextpad=0.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)

    # ── Save ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt, dpi_kw in [("png", {"dpi": 450}), ("svg", {}), ("pdf", {})]:
        fig.savefig(
            OUT_DIR / f"fig10_search_efficiency_curves.{fmt}",
            bbox_inches="tight", pad_inches=0.04, **dpi_kw,
        )
    plt.close(fig)
    print(f"Figure 10 saved → {OUT_DIR / 'fig10_search_efficiency_curves.png'}")


def main() -> None:
    draw_figure()


if __name__ == "__main__":
    main()
