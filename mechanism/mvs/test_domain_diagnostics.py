from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import numpy as np


MVS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MVS_ROOT))

import run_domain_diagnostics as diagnostics


class DomainDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = diagnostics.families.specs_from_registry()[0]
        cls.task = diagnostics.families.instantiate(cls.spec)

    def test_canonical_key_is_order_independent(self) -> None:
        rng = np.random.default_rng(123)
        config, _ = self.task.project(self.task.decode(self.task.sample_vector(rng)))
        reversed_config = dict(reversed(list(config.items())))
        self.assertEqual(
            diagnostics.canonical_key(self.task, config),
            diagnostics.canonical_key(self.task, reversed_config),
        )

    def test_projection_distance_is_bounded(self) -> None:
        rng = np.random.default_rng(456)
        raw = self.task.decode(self.task.sample_vector(rng))
        projected, _ = self.task.project(raw)
        distance = diagnostics.projection_distance(self.task, raw, projected)
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)

    def test_top_retention_known_case(self) -> None:
        high = np.arange(20, dtype=float)
        low = np.arange(20, dtype=float)
        self.assertEqual(diagnostics.top_retention(low, high), 1.0)
        reversed_low = low[::-1]
        self.assertEqual(diagnostics.top_retention(reversed_low, high), 0.0)

    def test_evaluate_instance_writes_all_candidates(self) -> None:
        stream = io.StringIO()
        row = diagnostics.evaluate_instance(self.spec, 20, stream)
        self.assertEqual(row["n_candidates"], 20)
        self.assertEqual(len(stream.getvalue().strip().splitlines()), 20)
        self.assertGreaterEqual(row["projected_feasible_rate"], 0.0)
        self.assertLessEqual(row["projected_feasible_rate"], 1.0)

    def test_eligibility_does_not_use_optimizer_results(self) -> None:
        row = {
            "projected_feasible_rate": 1.0,
            "projection_trigger_rate": 0.2,
            "objective_nonconstant": True,
            "spearman_low_high": 0.6,
            "top10_in_low25_retention": 0.5,
            "variable_kind_count": 3,
            "has_conditional_or_cross_constraint": True,
        }
        accepted, reasons = diagnostics.eligible(row)
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        row["top10_in_low25_retention"] = 0.0
        accepted, reasons = diagnostics.eligible(row)
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        row["spearman_low_high"] = 0.99
        accepted, reasons = diagnostics.eligible(row)
        self.assertFalse(accepted)
        self.assertIn("spearman_low_high", reasons)


if __name__ == "__main__":
    unittest.main()
