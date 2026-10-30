#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_METRIC = {
    "epoch": "vali_loss",
    "iter": "interval_train_loss",
    "search": "best_score",
}

DEFAULT_X = {
    "epoch": "epoch",
    "iter": "global_step",
    "search": "step",
}


def parse_input(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
    else:
        p = Path(spec)
        label = p.parent.name or p.stem
        path = spec
    if not label:
        raise ValueError(f"Empty label in input: {spec}")
    return label, Path(path)


def read_series(path: Path, x_col: str, y_col: str) -> Tuple[List[float], List[float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    xs: List[float] = []
    ys: List[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [col for col in (x_col, y_col) if col not in reader.fieldnames]
        if missing:
            raise KeyError(f"{path} missing columns: {', '.join(missing)}")
        for row in reader:
            if row.get(x_col, "") == "" or row.get(y_col, "") == "":
                continue
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    if not xs:
        raise ValueError(f"No plottable rows found in {path}")
    return xs, ys


def infer_axis_labels(kind: str, x_col: str, y_col: str) -> Tuple[str, str]:
    x_label = {
        "epoch": "Epoch",
        "global_step": "Training Iteration",
        "step": "Candidate Evaluation",
    }.get(x_col, x_col)
    y_label = {
        "train_loss": "Train Loss",
        "interval_train_loss": "Train Loss",
        "vali_loss": "Validation Loss",
        "test_loss": "Test Loss",
        "best_score": "Best Proxy Validation Loss",
    }.get(y_col, y_col)
    if kind == "search" and y_col == "best_score":
        y_label = "Best Proxy Validation MSE"
    return x_label, y_label


def plot_series(
    series: Dict[str, Tuple[List[float], List[float]]],
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    for label, (xs, ys) in series.items():
        marker = "o" if len(xs) <= 30 else None
        plt.plot(xs, ys, marker=marker, linewidth=1.8, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-run loss/search comparison curves from saved CSV files."
    )
    parser.add_argument(
        "--kind",
        choices=["epoch", "iter", "search"],
        default="epoch",
        help="CSV type: epoch loss_history, iter train_iter_history, or search_history.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='Input CSVs. Use "label=path/to/file.csv" to set legend names.',
    )
    parser.add_argument("--metric", default=None, help="Y column to plot.")
    parser.add_argument("--x_col", default=None, help="X column to plot.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--title", default="", help="Figure title.")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    metric = args.metric or DEFAULT_METRIC[args.kind]
    x_col = args.x_col or DEFAULT_X[args.kind]
    xlabel, ylabel = infer_axis_labels(args.kind, x_col, metric)
    if args.xlabel:
        xlabel = args.xlabel
    if args.ylabel:
        ylabel = args.ylabel

    series: Dict[str, Tuple[List[float], List[float]]] = {}
    for spec in args.inputs:
        label, path = parse_input(spec)
        series[label] = read_series(path, x_col=x_col, y_col=metric)

    plot_series(
        series=series,
        output=Path(args.output),
        title=args.title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=args.dpi,
    )
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
