import unittest

from silo_amp.legacy_runtime import (
    LegacyRuntimeError,
    candidate_records_from_metrics,
    resolve_policy_weights,
)


class LegacyRuntimeTests(unittest.TestCase):
    def test_prefers_historical_best_weights(self):
        self.assertEqual(
            resolve_policy_weights({"best_model_weights": {"best": 1}, "model_weights": {"latest": 2}}),
            {"best": 1},
        )

    def test_rejects_checkpoint_without_policy_weights(self):
        with self.assertRaises(LegacyRuntimeError):
            resolve_policy_weights({"model_weights": None})

    def test_normalizes_legacy_metric_names_and_missing_checks_fail_closed(self):
        rows = candidate_records_from_metrics([{
            "id": "pep-1",
            "sequence": "AKLMAKLM",
            "apex_mean_mic": 3.5,
            "passes_marlys_80": float("nan"),
            "charge": 4.0,
            "hydrophobicity": 0.1,
        }])
        self.assertEqual(rows[0]["mean_predicted_mic"], 3.5)
        self.assertFalse(rows[0]["marlys_identity_pass"])
        self.assertEqual(rows[0]["max_hydrophobic_run"], 3)


if __name__ == "__main__":
    unittest.main()
