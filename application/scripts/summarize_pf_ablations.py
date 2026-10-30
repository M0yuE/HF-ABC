#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


METHOD_SPECS = [
    ("PF-ABC", "pf_abc", ["pf_abc", "pf_tahc_abc"]),
    ("w/o Elite", "w_o_elite", ["w_o_elite"]),
    ("w/o Type-aware", "w_o_typeaware", ["w_o_typeaware"]),
    ("w/o Projection", "w_o_projection", ["w_o_projection"]),
    ("w/o Two-stage", "w_o_two_stage", ["w_o_two_stage"]),
]


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "final_summary" in payload:
        return payload["final_summary"]
    return payload


def metric_cell(summary: dict[str, object] | None, mean_key: str, std_key: str) -> str:
    if not summary or mean_key not in summary:
        return "MISSING"
    mean_val = float(summary[mean_key])
    std_val = float(summary.get(std_key, 0.0))
    return f"{mean_val:.4f}+/-{std_val:.4f}"


def resolve_summary_path(
    output_dir: Path,
    pred_len: int,
    model_variant: str,
    method_names: list[str],
) -> tuple[Path, dict[str, object] | None]:
    for method_name in method_names:
        summary_path = output_dir / f"water_quality_{pred_len}_{model_variant}_{method_name}.json"
        summary = load_summary(summary_path)
        if summary is not None:
            return summary_path, summary
    primary_path = output_dir / f"water_quality_{pred_len}_{model_variant}_{method_names[0]}.json"
    return primary_path, None


def build_rows(output_dir: Path, pred_len: int, model_variant: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, method_key, method_names in METHOD_SPECS:
        summary_path, summary = resolve_summary_path(output_dir, pred_len, model_variant, method_names)
        rows.append(
            {
                "pred_len": str(pred_len),
                "method": label,
                "method_key": method_key,
                "source_file": summary_path.name,
                "mse": metric_cell(summary, "mse_mean", "mse_std"),
                "mae": metric_cell(summary, "mae_mean", "mae_std"),
                "key_mse": metric_cell(summary, "key_mse_mean", "key_mse_std"),
                "key_mae": metric_cell(summary, "key_mae_mean", "key_mae_std"),
            }
        )
    return rows


def build_missing_commands(
    rows: list[dict[str, str]],
    *,
    script_path: str,
    manifest: str,
    dataset_name: str,
    model_variant: str,
    output_dir: str,
    gpu: int,
    python_bin: str,
    extra_args: str,
) -> list[str]:
    commands: list[str] = []
    base_parts = [
        python_bin,
        script_path,
        "--manifest",
        manifest,
        "--dataset_name",
        dataset_name,
        "--model_variant",
        model_variant,
        "--output_dir",
        output_dir,
        "--gpu",
        str(gpu),
    ]
    extra_parts = shlex.split(extra_args, posix=False) if extra_args.strip() else []
    for row in rows:
        if row["mse"] != "MISSING":
            continue
        parts = base_parts + ["--pred_len", row["pred_len"], "--method", row["method_key"]] + extra_parts
        commands.append(" ".join(powershell_quote(part) for part in parts))
    return commands


def powershell_quote(part: str) -> str:
    text = str(part)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in '"&()[]{};,\'' ):
        return '"' + text.replace('"', '`"') + '"'
    return text


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| H | Method | MSE | MAE | Key-MSE | Key-MAE | Source |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['pred_len']} | {row['method']} | {row['mse']} | {row['mae']} | {row['key_mse']} | {row['key_mae']} | {row['source_file']} |"
        )
    return "\n".join(lines)


def render_csv(rows: list[dict[str, str]]) -> str:
    header = ["pred_len", "method", "mse", "mae", "key_mse", "key_mae", "source_file"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(row[key] for key in header))
    return "\n".join(lines)


def render_missing_commands(commands: list[str]) -> str:
    if not commands:
        return "# No missing PF ablation runs detected."
    return "\n".join(["# Missing PF ablation commands", *commands])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing PF-ABC ablation result JSON files.")
    parser.add_argument("--output_dir", type=str, default="./pf_runs_wq")
    parser.add_argument("--fallback_output_dir", type=str, default="./egta_runs_wq")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[60, 120, 240])
    parser.add_argument("--model_variant", type=str, default="iTransformer_CSG")
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument("--write", type=str, default=None, help="Optional path to save the rendered summary.")
    parser.add_argument("--commands_write", type=str, default=None, help="Optional path to save commands for missing ablation runs.")
    parser.add_argument("--script_path", type=str, default="scripts/run_water_quality_suite.py")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--manifest", type=str, default="./dataset/aved_n2o/manifest.json")
    parser.add_argument("--dataset_name", type=str, default="WQ_GENERIC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--extra_args", type=str, default="", help="Extra CLI args appended to every generated command.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists() and args.fallback_output_dir:
        fallback = Path(args.fallback_output_dir)
        if fallback.exists():
            output_dir = fallback

    rows: list[dict[str, str]] = []
    for pred_len in args.pred_lens:
        rows.extend(build_rows(output_dir, pred_len, args.model_variant))

    if args.format == "json":
        rendered = json.dumps(rows, ensure_ascii=False, indent=2)
    elif args.format == "csv":
        rendered = render_csv(rows)
    else:
        rendered = render_markdown(rows)

    if args.write:
        Path(args.write).write_text(rendered, encoding="utf-8")

    if args.commands_write:
        commands = build_missing_commands(
            rows,
            script_path=args.script_path,
            manifest=args.manifest,
            dataset_name=args.dataset_name,
            model_variant=args.model_variant,
            output_dir=str(output_dir),
            gpu=args.gpu,
            python_bin=args.python_bin,
            extra_args=args.extra_args,
        )
        Path(args.commands_write).write_text(render_missing_commands(commands), encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
