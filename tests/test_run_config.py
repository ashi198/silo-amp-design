import tempfile
import unittest
from pathlib import Path

from silo_amp.run_config import RunConfig, RunConfigurationError, config_from_args


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "policy.pt"
        self.checkpoint.write_bytes(b"fixture")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalizes_paths_and_parses_explicit_values(self):
        config = config_from_args(
            [
                "--checkpoint",
                str(self.checkpoint),
                "--output-dir",
                str(self.root / "output"),
                "--seed",
                "7",
                "--device",
                "cuda:0",
                "--mode",
                "training",
            ]
        )

        self.assertEqual(config.seed, 7)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.mode, "training")
        self.assertTrue(config.checkpoint.is_absolute())

    def test_rejects_missing_checkpoint(self):
        with self.assertRaisesRegex(RunConfigurationError, "checkpoint does not exist"):
            RunConfig(self.root / "missing.pt", self.root / "output")

    def test_rejects_invalid_threshold_and_budget(self):
        with self.assertRaisesRegex(RunConfigurationError, "diversity_threshold"):
            RunConfig(self.checkpoint, self.root / "output", diversity_threshold=1.1)
        with self.assertRaisesRegex(RunConfigurationError, "top_k_peptides"):
            RunConfig(self.checkpoint, self.root / "output", top_k_peptides=100_001)

    def test_rejects_unsupported_device(self):
        with self.assertRaisesRegex(RunConfigurationError, "device must be"):
            RunConfig(self.checkpoint, self.root / "output", device="cuda")


if __name__ == "__main__":
    unittest.main()
