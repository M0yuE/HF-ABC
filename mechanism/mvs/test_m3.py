from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MVS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MVS_ROOT))

import run_m3 as m3


class M3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = m3.load_selected_specs()[0]

    def test_multifidelity_requested_cost_is_locked(self) -> None:
        self.assertEqual(m3.multifidelity_requested_cost(), 3744)

    def test_cache_accounting_identity(self) -> None:
        task = m3.families.instantiate(self.spec)
        rng = np.random.default_rng(12)
        raw = task.decode(task.sample_vector(rng))
        with tempfile.TemporaryDirectory() as directory:
            ledger = m3.Ledger(
                task,
                "test",
                1,
                Path(directory) / "trace.jsonl",
            )
            for index in range(2):
                ledger.evaluate(
                    raw,
                    9,
                    use_projection=True,
                    stage=0,
                    source="test",
                    candidate_id=f"c{index}",
                    parent_id=None,
                )
            self.assertEqual(ledger.request_count, 2)
            self.assertEqual(ledger.objective_call_count, 1)
            self.assertEqual(ledger.cache_hit_count, 1)
            self.assertEqual(
                ledger.request_count,
                ledger.objective_call_count + ledger.cache_hit_count,
            )
            self.assertEqual(ledger.requested_cost, 18)
            self.assertEqual(ledger.actual_cost, 9)

    def test_projected_multifidelity_budget_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = m3.run_one(
                self.spec,
                "hf_abc",
                303600,
                Path(directory),
            )
            self.assertEqual(row["request_count"], 88)
            self.assertEqual(row["requested_cost"], 3744)
            self.assertLessEqual(row["actual_cost"], row["requested_cost"])
            self.assertTrue(Path(directory, row["trace_file"]).exists())

    def test_full_fidelity_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = m3.run_one(
                self.spec,
                "hf_no_mf",
                303600,
                Path(directory),
            )
            self.assertEqual(row["request_count"], 26)
            self.assertEqual(row["requested_cost"], 3744)
            self.assertLessEqual(row["actual_cost"], row["requested_cost"])


if __name__ == "__main__":
    unittest.main()
