"""Equal CPU wall-clock supplement on the exact M3 manuscript instances.

Each M3 method is run in independent fixed-schedule restarts until the time
deadline. Only feasible full-fidelity evaluations completed by each checkpoint
enter the incumbent. Every restart uses the unchanged M3 optimizer and ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

MVS_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = MVS_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT / "vendor"))
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(MVS_ROOT))

import benchmark_families as families
import run_m3 as m3


class TimedLedger(m3.Ledger):
    def __init__(self, task, method, seed, started):
        super().__init__(task, method, seed, Path("unused"))
        self.started = started

    def evaluate(self, raw_config, fidelity, **kwargs):
        result = super().evaluate(raw_config, fidelity, **kwargs)
        self.rows[-1]["elapsed_seconds"] = time.perf_counter() - self.started
        return result


def run_one(spec, method, seed, duration, checkpoints, output):
    task = families.instantiate(spec)
    started = time.perf_counter()
    deadline = started + duration
    events = []
    cycles = 0
    while time.perf_counter() < deadline:
        cycle_seed = seed + 1_000_003 * cycles
        ledger = TimedLedger(task, method, cycle_seed, started)
        if method == "hf_no_mf":
            m3.run_full_fidelity(task, cycle_seed, ledger)
        else:
            m3.run_multifidelity(task, method, cycle_seed, ledger)
        events.extend(ledger.rows)
        cycles += 1
    actual = time.perf_counter() - started
    rows = []
    for checkpoint in checkpoints:
        within = [r for r in events if r["elapsed_seconds"] <= checkpoint]
        # The original M3 protocol assigns a penalty score to infeasible
        # candidates; retain that score when no feasible candidate is found.
        full = [r for r in within if r["fidelity"] == task.max_fidelity]
        best = min(full, key=lambda r: r["score"]) if full else None
        rows.append({
            "instance_key": spec.instance_key,
            "split": "manuscript_selected",
            "task_family": spec.task_family,
            "difficulty": spec.difficulty,
            "method": method,
            "search_seed": seed,
            "checkpoint_seconds": checkpoint,
            "best_full_fidelity_score": best["score"] if best else "",
            "full_fidelity_calls": len(full),
            "all_calls": len(within),
            "actual_run_seconds": actual,
            "iterations": cycles,
        })
    trace = output / "traces" / f"{spec.instance_key}__{method}__seed{seed}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(m3.jsonable(event), ensure_ascii=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.instances < 1 or args.seeds < 1 or args.seconds <= 0:
        parser.error("instances, seeds and seconds must be positive")
    selection = json.loads(m3.SELECTION_PATH.read_text(encoding="utf-8"))
    registry = families.load_registry()
    specs_by_key = {s.instance_key: s for s in families.specs_from_registry()}
    selected = selection["selected_instance_keys"]
    specs = [specs_by_key[key] for key in selected[:args.instances]]
    checkpoints = sorted(set(round(args.seconds * x, 6) for x in (0.25, 0.5, 1.0)))
    seeds = list(range(303700, 303700 + args.seeds))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps({
        "split": "manuscript_selected",
        "instances": [spec.instance_key for spec in specs],
        "methods": m3.METHODS,
        "search_seeds": seeds,
        "seconds": args.seconds,
        "checkpoints": checkpoints,
        "registry_sha256": registry["instances_sha256"],
        "selection_decision_sha256": m3.sha256_file(m3.SELECTION_PATH),
        "full_fidelity_only": True,
        "fixed_schedule_methods_restart_on_completion": True,
        "cache_scope": "one M3 search restart",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    total = len(specs) * len(m3.METHODS) * len(seeds)
    for spec in specs:
        for seed in seeds:
            for method in m3.METHODS:
                print(f"[{len(rows) // len(checkpoints) + 1}/{total}] {spec.instance_key} {method} seed={seed}", flush=True)
                rows.extend(run_one(spec, method, seed, args.seconds, checkpoints, args.output))
    with (args.output / "checkpoints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
