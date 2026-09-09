import json
import tempfile
import unittest
from pathlib import Path

from split_manifest import build_split_manifest, sha256_file, verify_split_manifest


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records), encoding="utf-8")


class SplitManifestTests(unittest.TestCase):
    def test_manifest_is_reproducible_and_records_hashes_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fasta"
            antibacterial = root / "antibacterial.fasta"
            marlys = root / "marlys.fasta"
            manifest_a = root / "a.json"
            manifest_b = root / "b.json"
            records = [(f"id-{i}", "ACDEFGHI" + "K" * i) for i in range(9)]
            write_fasta(source, records + [("duplicate", records[0][1]), ("bad", "ACDXYZ")])
            write_fasta(antibacterial, [("reference-a", records[1][1])])
            write_fasta(marlys, [("reference-b", records[2][1])])

            references = [antibacterial, marlys]
            first = build_split_manifest(source, manifest_a, reference_fastas=references, seed=7)
            second = build_split_manifest(source, manifest_b, reference_fastas=references, seed=7)

            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["sources"]["source_fasta"]["sha256"], sha256_file(source))
            self.assertIn("first-seen FASTA order", first["source_order"])
            self.assertEqual(
                [record["order"] for record in first["splits"]["train"]["records"]],
                [record["order"] for record in second["splits"]["train"]["records"]],
            )
            self.assertEqual(first["curation_policy"]["counts"]["removed_reference_match"], 2)
            self.assertEqual(first["curation_policy"]["counts"]["removed_duplicate"], 1)
            self.assertEqual(first["curation_policy"]["counts"]["removed_empty_or_noncanonical"], 1)
            verify_split_manifest(first)
            self.assertEqual(json.loads(manifest_a.read_text()), json.loads(manifest_b.read_text()))

    def test_training_exclusion_applies_to_all_splits_and_writes_fastas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fasta"
            training = root / "training.fasta"
            manifest_path = root / "manifest.json"
            sequences = ["ACDEFGHI", "CDEFGHIK", "DEFGHIKL", "EFGHIKLM", "FGHIKLMN", "GHIKLMNP"]
            write_fasta(source, [(str(i), sequence) for i, sequence in enumerate(sequences)])
            write_fasta(training, [("already-trained", sequences[0])])

            manifest = build_split_manifest(
                source,
                manifest_path,
                training_fasta=training,
                split_output_dir=root / "splits",
                validation_fraction=0.2,
                test_fraction=0.2,
            )
            verify_split_manifest(manifest)
            all_sequences = {
                record["sequence"]
                for split in manifest["splits"].values()
                for record in split["records"]
            }
            self.assertNotIn(sequences[0], all_sequences)
            self.assertEqual(
                {path.name for path in (root / "splits").glob("*.fasta")},
                {"train.fasta", "validation.fasta", "test.fasta"},
            )

    def test_invalid_small_or_overlapping_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fasta"
            write_fasta(source, [(str(i), "ACDEFGHI") for i in range(2)])
            with self.assertRaises(ValueError):
                build_split_manifest(source, root / "manifest.json")

            write_fasta(source, [(str(i), "ACDEFGHI") for i in range(3)])
            with self.assertRaises(ValueError):
                build_split_manifest(
                    source,
                    root / "manifest.json",
                    validation_fraction=0.5,
                    test_fraction=0.5,
                )

            manifest = {
                "manifest_version": 1,
                "splits": {
                    "train": {"count": 1, "records": [{"sequence": "ACDEFGHI"}]},
                    "validation": {"count": 1, "records": [{"sequence": "ACDEFGHI"}]},
                    "test": {"count": 0, "records": []},
                },
            }
            with self.assertRaises(ValueError):
                verify_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
