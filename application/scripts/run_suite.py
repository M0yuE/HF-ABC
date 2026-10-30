#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pf.pf_abc import PF_LEGACY_METHOD, PF_METHOD, normalize_pf_method
from pf.runner import ExperimentRunner
from egta.multifidelity import ASHA_METHOD, HYPERBAND_METHOD


SUITE_METHOD_CHOICES = [
    "fixed",
    "random_search",
    "ga",
    "pso",
    "abc",
    "egta_abc",
    HYPERBAND_METHOD,
    ASHA_METHOD,
    PF_METHOD,
    PF_LEGACY_METHOD,
    "w_o_elite",
    "w_o_typeaware",
    "w_o_projection",
    "w_o_two_stage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PF-ABC experiment suite on iTransformer.")
    parser.add_argument("--output_dir", type=str, default="./pf_runs", help="Directory for logs and summaries.")
    parser.add_argument("--dataset", type=str, required=True, choices=["ETTh1", "ETTm1", "Weather", "Electricity"])
    parser.add_argument("--pred_len", type=int, required=True, choices=[96, 192, 336, 720])
    parser.add_argument("--model_variant", type=str, default="iTransformer", choices=["iTransformer", "iTransformer_CSG"])
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=SUITE_METHOD_CHOICES,
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--search_budget", type=int, default=20)
    parser.add_argument("--colony_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--elite_ratio", type=float, default=0.4)
    parser.add_argument("--search_train_epochs", type=int, default=5)
    parser.add_argument("--search_patience", type=int, default=3)
    parser.add_argument("--final_train_epochs", type=int, default=20)
    parser.add_argument("--final_patience", type=int, default=5)
    parser.add_argument("--stage2_radius", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--search_seeds", type=int, nargs="+", default=None)
    parser.add_argument("--final_seeds", type=int, nargs="+", default=None)
    parser.add_argument("--race_fidelities", type=int, nargs="+", default=[1, 3, 6, 10])
    parser.add_argument("--race_init_size", type=int, default=48)
    parser.add_argument("--race_keep", type=int, nargs="+", default=[16, 8, 4, 4])
    parser.add_argument("--race_challengers", type=int, nargs="+", default=[0, 8, 4, 0])
    parser.add_argument("--verify_top_k", type=int, default=4)
    parser.add_argument("--verify_epochs", type=int, default=20)
    parser.add_argument("--verify_seed", type=int, default=None)
    parser.add_argument("--challenge_struct_prob", type=float, default=0.30)
    parser.add_argument("--challenge_local_prob", type=float, default=0.70)
    parser.add_argument("--hf_abc", action="store_true")
    parser.add_argument("--elite_refine_ratio", type=float, default=0.70)
    parser.add_argument("--elite_struct_ratio", type=float, default=0.20)
    parser.add_argument("--elite_random_ratio", type=float, default=0.10)
    parser.add_argument("--elite_top_m", type=int, default=4)
    parser.add_argument("--elite_lr_sigma", type=float, default=0.20)
    parser.add_argument("--elite_dropout_sigma", type=float, default=0.02)
    parser.add_argument("--elite_max_tries", type=int, default=50)
    parser.add_argument("--max_duplicate_tries", type=int, default=50)
    parser.add_argument("--pf_save_archive", action="store_true")
    parser.add_argument("--pf_no_resume", action="store_true")
    parser.add_argument("--mf_min_epochs", type=int, default=1)
    parser.add_argument("--mf_max_epochs", type=int, default=None)
    parser.add_argument("--mf_eta", type=int, default=3)
    parser.add_argument("--mf_budget_epochs", type=int, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--skip_epoch_val",
        dest="skip_epoch_val",
        action="store_true",
        default=True,
        help="Skip validation after every training epoch; enabled by default for faster HPO experiments.",
    )
    parser.add_argument(
        "--keep_epoch_val",
        dest="skip_epoch_val",
        action="store_false",
        help="Restore the original per-epoch validation and early stopping behavior.",
    )
    args = parser.parse_args()
    public_method = args.method.lower()
    normalized_pf = normalize_pf_method(args.method)
    runner_method = PF_LEGACY_METHOD if normalized_pf is not None else public_method
    if normalized_pf is not None:
        public_method = PF_METHOD

    overrides = {"num_workers": args.num_workers, "use_amp": args.use_amp, "skip_epoch_val": args.skip_epoch_val}
    for key in ["root_path", "data_path", "freq", "enc_in", "dec_in", "c_out"]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        pred_len=args.pred_len,
        model_name=args.model_variant,
        gpu=args.gpu,
        overrides=overrides,
    )

    if runner_method == "fixed":
        result = runner.run_fixed_baseline(args.seeds)
    else:
        result = runner.run_search_method(
            method=runner_method,
            seeds=args.seeds,
            search_seeds=args.search_seeds,
            final_seeds=args.final_seeds,
            search_budget=args.search_budget,
            colony_size=args.colony_size,
            limit=args.limit,
            elite_ratio=args.elite_ratio,
            search_train_epochs=args.search_train_epochs,
            search_patience=args.search_patience,
            final_train_epochs=args.final_train_epochs,
            final_patience=args.final_patience,
            use_two_stage=(runner_method != "w_o_two_stage"),
            stage2_radius=args.stage2_radius,
            race_fidelities=args.race_fidelities,
            race_init_size=args.race_init_size,
            race_keep=args.race_keep,
            race_challengers=args.race_challengers,
            verify_top_k=args.verify_top_k,
            verify_epochs=args.verify_epochs,
            verify_seed=args.verify_seed,
            challenge_struct_prob=args.challenge_struct_prob,
            challenge_local_prob=args.challenge_local_prob,
            hf_abc=args.hf_abc,
            elite_refine_ratio=args.elite_refine_ratio,
            elite_struct_ratio=args.elite_struct_ratio,
            elite_random_ratio=args.elite_random_ratio,
            elite_top_m=args.elite_top_m,
            elite_lr_sigma=args.elite_lr_sigma,
            elite_dropout_sigma=args.elite_dropout_sigma,
            elite_max_tries=args.elite_max_tries,
            max_duplicate_tries=args.max_duplicate_tries,
            pf_save_archive=args.pf_save_archive,
            pf_no_resume=args.pf_no_resume,
            mf_min_epochs=args.mf_min_epochs,
            mf_max_epochs=args.mf_max_epochs,
            mf_eta=args.mf_eta,
            mf_budget_epochs=args.mf_budget_epochs,
        )

    summary_path = Path(args.output_dir) / f"{args.dataset}_{args.pred_len}_{args.model_variant}_{public_method}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
