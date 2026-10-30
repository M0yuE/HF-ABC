"""Rebuild checkpoint scores from preserved M3 timed traces.

This derives a new CSV while retaining the first checkpoint CSV unchanged.
Original M3 penalized scores count when no feasible full-fidelity candidate is
available. Nonempty scores in the first CSV must match the rebuilt values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.output / "checkpoints.csv"
    with original.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (row["instance_key"], row["method"], row["search_seed"])
        if key not in groups:
            trace = args.output / "traces" / f"{key[0]}__{key[1]}__seed{key[2]}.jsonl"
            with trace.open(encoding="utf-8") as stream:
                groups[key] = [json.loads(line) for line in stream]
        checkpoint = float(row["checkpoint_seconds"])
        full = [
            event for event in groups[key]
            if event["elapsed_seconds"] <= checkpoint and event["fidelity"] == 144
        ]
        if not full:
            raise RuntimeError(f"No full-fidelity event at checkpoint: {key} {checkpoint}")
        best = min(full, key=lambda event: event["score"])
        if row["best_full_fidelity_score"] and abs(
            float(row["best_full_fidelity_score"]) - float(best["score"])
        ) > 1e-12:
            raise RuntimeError(f"Existing score differs from trace: {key} {checkpoint}")
        row["best_full_fidelity_score"] = best["score"]
        row["full_fidelity_calls"] = len(full)
    derived = args.output / "checkpoints_penalized.csv"
    with derived.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rebuilt {len(rows)} rows in {derived}; preserved {original}")


if __name__ == "__main__":
    main()
