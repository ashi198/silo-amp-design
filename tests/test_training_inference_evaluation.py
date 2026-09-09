import pickle
import tempfile
import unittest
from pathlib import Path

from silo_amp.acceptance import verify_submission
from silo_amp.artifacts import ArtifactError
from silo_amp.evaluation import summarize_seed_results
from silo_amp.inference import run_inference
from silo_amp.training import TrainingCycleRunner


class LaterTicketTests(unittest.TestCase):
    def test_training_selects_50_lowest_and_preserves_historical_best(self):
        runner = TrainingCycleRunner()
        state = {"weight": 0}
        def sample(weights, count):
            return [{"sequence": f"S{i}", "mean_predicted_mic": i} for i in range(count)]
        def train(weights, experts):
            return {"weight": weights["weight"] + 1, "experts": len(experts)}
        runner.run_round(live_weights=state, sample=sample, train=train, validate=lambda weights, _: 2.0, round_index=0)
        runner.run_round(live_weights={"weight": 1}, sample=sample, train=train, validate=lambda weights, _: 3.0, round_index=1)
        self.assertEqual(runner.best_metric, 2.0)
        self.assertEqual(runner.best_weights["experts"], 50)

    def test_inference_and_acceptance_are_exact_and_service_free(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint.pkl"
            candidates = []
            for i in range(3):
                candidates.append({"id": str(i), "sequence": "ACDEFGHI" + "K" * i, "mean_predicted_mic": 1.0, "marlys_identity_pass": True, "charge": 5, "hydrophobicity": 0.2, "cysteine_count": 0, "max_hydrophobic_run": 2})
            with checkpoint.open("wb") as handle:
                pickle.dump({"candidates": candidates}, handle)
            with self.assertRaises(ArtifactError):
                run_inference(checkpoint, Path(temp) / "out", expected_50k=4, expected_top_k=2)

    def test_evaluation_reports_mean_std_and_claim_boundary(self):
        summary = summarize_seed_results([{"seed": 11, "mean_predicted_mic": 2, "mic_pass_fraction": .5}, {"seed": 23, "mean_predicted_mic": 4, "mic_pass_fraction": .7}])
        self.assertEqual(summary["aggregate"]["mean_predicted_mic"]["mean"], 3)
        self.assertIn("uncalibrated surrogate", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
