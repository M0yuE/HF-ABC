from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
HF_ROOT = ROOT / "HF-ABC"
DATA_CSV = SCRIPT_DIR / "fig03_fidelity_rank_correlation_data.csv"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_PNG = OUT_DIR / "fig03_fidelity_rank_correlation.png"
OUT_SVG = OUT_DIR / "fig03_fidelity_rank_correlation.svg"
OUT_PDF = OUT_DIR / "fig03_fidelity_rank_correlation.pdf"

SOURCE_ARCHIVES = {
    "PF backbone": HF_ROOT / "pf_runs_wq_two_hpo" / "search_seed_2026" / "pf_archive.json",
    "HF w/o restart": HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_wo_restart" / "search_seed_2026" / "pf_archive.json",
    "HF local-only": HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_local_only" / "search_seed_2026" / "pf_archive.json",
    "HF struct-only": HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_struct_only" / "search_seed_2026" / "pf_archive.json",
    "HF restart-only": HF_ROOT / "hf-abc-ablation" / "H120" / "hf_abc_restart_only" / "search_seed_2026" / "pf_archive.json",
}

CONFIG_FIELDS = [
    "seq_len",
    "d_model",
    "n_heads",
    "e_layers",
    "d_ff",
    "dropout",
    "learning_rate",
    "batch_size",
    "csg_reduction",
]

PANELS = [
    {
        "key": "screening",
        "low": 3,
        "high": 6,
        "title": "Progressive screening",
        "xlabel": "3-epoch validation MSE",
        "ylabel": "6-epoch validation MSE",
    },
    {
        "key": "verification",
        "low": 10,
        "high": 20,
        "title": "Top-k verification",
        "xlabel": "10-epoch validation MSE",
        "ylabel": "20-epoch verification MSE",
    },
]


def config_key(config: dict[str, object]) -> tuple[object, ...]:
    key = []
    for name in CONFIG_FIELDS:
        value = config[name]
        if name in {"dropout", "learning_rate"}:
            key.append(round(float(value), 6))
        else:
            key.append(int(value))
    return tuple(key)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sx = sum((item - mx) ** 2 for item in x)
    sy = sum((item - my) ** 2 for item in y)
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return float("nan")
    return (concordant - discordant) / denom


def load_archive_pairs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, path in SOURCE_ARCHIVES.items():
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        by_config: dict[tuple[object, ...], dict[int, float]] = defaultdict(dict)
        for record in records:
            if record.get("status") != "ok":
                continue
            fidelity = int(record["fidelity"])
            if fidelity <= 0:
                continue
            key = config_key(record["config"])
            score = float(record["val_mse"])
            previous = by_config[key].get(fidelity)
            if previous is None or score < previous:
                by_config[key][fidelity] = score

        for panel in PANELS:
            low = int(panel["low"])
            high = int(panel["high"])
            for key, values in by_config.items():
                if low not in values or high not in values:
                    continue
                rows.append(
                    {
                        "panel": panel["key"],
                        "source": source,
                        "low_fidelity": low,
                        "high_fidelity": high,
                        "low_mse": values[low],
                        "high_mse": values[high],
                        "config_key": json.dumps(key),
                    }
                )
    return rows


def write_data_csv(rows: list[dict[str, object]]) -> None:
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DATA_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "panel",
                "source",
                "low_fidelity",
                "high_fidelity",
                "low_mse",
                "high_mse",
                "config_key",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def panel_stats(rows: list[dict[str, object]], panel_key: str) -> dict[str, float]:
    points = [row for row in rows if row["panel"] == panel_key]
    x = [float(row["low_mse"]) for row in points]
    y = [float(row["high_mse"]) for row in points]
    return {
        "n": float(len(points)),
        "spearman": spearman(x, y),
        "kendall": kendall_tau_b(x, y),
        "pearson": pearson(x, y),
    }


def plot_panel(ax, rows: list[dict[str, object]], panel: dict[str, object], label: str) -> None:
    points = [row for row in rows if row["panel"] == panel["key"]]
    x = np.array([float(row["low_mse"]) for row in points], dtype=float)
    y = np.array([float(row["high_mse"]) for row in points], dtype=float)
    stats = panel_stats(rows, str(panel["key"]))

    ax.scatter(
        x,
        y,
        s=34,
        facecolor="#2F6F9F",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.82,
        zorder=3,
    )
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#1F3B57", lw=1.45, zorder=4)
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], color="#AAB3BD", lw=0.9, ls=(0, (3, 2)), zorder=1)

    x_pad = max((float(x.max()) - float(x.min())) * 0.12, 0.0008)
    y_pad = max((float(y.max()) - float(y.min())) * 0.12, 0.0008)
    ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
    ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)

    ax.set_title(str(panel["title"]), fontsize=9.8, fontweight="semibold", pad=4)
    ax.set_xlabel(str(panel["xlabel"]), fontsize=8.6)
    ax.set_ylabel(str(panel["ylabel"]), fontsize=8.6)
    ax.grid(color="#E8EDF3", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.2, width=0.7, labelsize=7.4)
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.05,
        f"Spearman $\\rho$ = {stats['spearman']:.3f}\n"
        f"Kendall $\\tau$ = {stats['kendall']:.3f}\n"
        f"n = {int(stats['n'])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D4DCE5",
            "linewidth": 0.7,
            "alpha": 0.92,
        },
    )


def main() -> None:
    rows = load_archive_pairs()
    if not rows:
        raise RuntimeError("No paired fidelity records were found.")
    write_data_csv(rows)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.85,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    for ax, panel, label in zip(axes, PANELS, ["a", "b"]):
        plot_panel(ax, rows, panel, label)
    fig.suptitle(
        "Rank correlation between low-fidelity and high-fidelity evaluations",
        fontsize=10.5,
        fontweight="semibold",
        y=1.015,
    )
    fig.tight_layout(w_pad=1.25)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    for panel in PANELS:
        stats = panel_stats(rows, str(panel["key"]))
        print(
            f"{panel['key']}: n={int(stats['n'])}, "
            f"spearman={stats['spearman']:.6f}, "
            f"kendall={stats['kendall']:.6f}, "
            f"pearson={stats['pearson']:.6f}"
        )
    print(OUT_PNG.resolve())


if __name__ == "__main__":
    main()
