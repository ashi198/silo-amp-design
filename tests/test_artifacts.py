import json
import tempfile
import unittest
from pathlib import Path

from silo_amp.artifacts import ArtifactError, write_submission_artifacts


def row(identifier, sequence):
    return {"id": identifier, "sequence": sequence, "mean_predicted_mic": 1.0}


class ArtifactTests(unittest.TestCase):
    def test_complete_artifacts_have_stable_manifest_and_csv_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = write_submission_artifacts(
                temp,
                [row("b", "ACDEFGHI"), row("a", "KLMNPQRS")],
                [row("a", "KLMNPQRS")],
                metadata={"seed": 42, "checkpoint_sha256": "abc"},
                expected_50k=2,
                expected_top_k=1,
            )
            self.assertEqual(manifest["counts"], {"library_50k": 2, "top_100": 1})
            self.assertEqual(Path(temp, "library_50k.fasta").read_text().splitlines()[0], ">b")
            self.assertEqual(Path(temp, "library_50k.csv").read_text().count("\n"), 3)
            self.assertEqual(json.loads(Path(temp, "manifest.json").read_text())["metadata"]["seed"], 42)

    def test_shortfall_writes_diagnostic_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ArtifactError):
                write_submission_artifacts(temp, [row("a", "ACDEFGHI")], [], metadata={}, expected_50k=2, expected_top_k=1)
            report = json.loads(Path(temp, "diagnostic_error.json").read_text())
            self.assertIn("50k_count:1 != 2", report["reasons"])


if __name__ == "__main__":
    unittest.main()
