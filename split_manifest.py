"""Create and verify reproducible SILO AMP data split manifests.

The manifest is the source of truth for the starting states used by training,
validation, and test runs.  This module deliberately has no model or runtime
dependencies so that split construction can be audited in a clean Python
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class FastaRecord:
    """A FASTA record retaining its source order and header."""

    header: str
    sequence: str
    source_ordinal: int


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read FASTA records, preserving record order and headers."""
    path = Path(path)
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def finish() -> None:
        if header is not None:
            records.append(
                FastaRecord(header, "".join(sequence_parts).upper(), len(records))
            )

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header in {path} at line {line_number}")
                sequence_parts = []
            elif header is None:
                raise ValueError(f"Sequence appears before a FASTA header in {path} at line {line_number}")
            else:
                sequence_parts.append(line)
    finish()
    return records


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a source file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(sequences: Iterable[str]) -> str:
    """Hash canonical sequence order with unambiguous line framing."""
    digest = hashlib.sha256()
    for sequence in sequences:
        encoded = sequence.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _source_metadata(path: Path, records: Sequence[FastaRecord]) -> dict[str, object]:
    sequences = [record.sequence for record in records]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_count": len(records),
        "unique_sequence_count": len(set(sequences)),
        "sequence_sha256": sequence_digest(sequences),
    }


def _allocate_counts(total: int, validation_fraction: float, test_fraction: float) -> dict[str, int]:
    if total < 3:
        raise ValueError("At least three curated sequences are required for train/validation/test")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be greater than zero")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    raw = {
        "train": total * (1 - validation_fraction - test_fraction),
        "validation": total * validation_fraction,
        "test": total * test_fraction,
    }
    counts = {name: int(value) for name, value in raw.items()}
    for name in ("validation", "test"):
        counts[name] = max(1, counts[name])
    while sum(counts.values()) > total:
        reducible = max((name for name in counts if counts[name] > 1), key=lambda name: raw[name])
        counts[reducible] -= 1
    while sum(counts.values()) < total:
        name = max(counts, key=lambda candidate: raw[candidate] - counts[candidate])
        counts[name] += 1
    return counts


def curate_records(
    source_records: Sequence[FastaRecord],
    reference_sequences: set[str],
    *,
    min_length: int = 8,
    max_length: int = 50,
) -> tuple[list[FastaRecord], dict[str, int]]:
    """Apply the documented curation policy in a stable, first-seen order."""
    if min_length < 1 or max_length < min_length:
        raise ValueError("Invalid sequence length bounds")
    counts = {
        "source_records": len(source_records),
        "removed_empty_or_noncanonical": 0,
        "removed_length": 0,
        "removed_reference_match": 0,
        "removed_duplicate": 0,
        "curated_records": 0,
    }
    seen: set[str] = set()
    curated: list[FastaRecord] = []
    for record in source_records:
        sequence = record.sequence.upper()
        if not sequence or not set(sequence).issubset(CANONICAL_AA):
            counts["removed_empty_or_noncanonical"] += 1
        elif not min_length <= len(sequence) <= max_length:
            counts["removed_length"] += 1
        elif sequence in reference_sequences:
            counts["removed_reference_match"] += 1
        elif sequence in seen:
            counts["removed_duplicate"] += 1
        else:
            seen.add(sequence)
            curated.append(FastaRecord(record.header, sequence, len(curated)))
    counts["curated_records"] = len(curated)
    return curated, counts


def _split_records(records: Sequence[FastaRecord], seed: int, counts: Mapping[str, int]) -> dict[str, list[FastaRecord]]:
    order = list(records)
    random.Random(seed).shuffle(order)
    result: dict[str, list[FastaRecord]] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        size = counts[split]
        result[split] = [
            FastaRecord(record.header, record.sequence, record.source_ordinal)
            for record in order[cursor : cursor + size]
        ]
        cursor += size
    return result


def _record_dict(record: FastaRecord) -> dict[str, object]:
    return {"header": record.header, "sequence": record.sequence, "order": record.source_ordinal}


def _write_fasta(records: Sequence[FastaRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.header}\n{record.sequence}\n")


def build_split_manifest(
    source_fasta: str | Path,
    output_manifest: str | Path,
    *,
    training_fasta: str | Path | None = None,
    reference_fastas: Sequence[str | Path] = (),
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    min_length: int = 8,
    max_length: int = 50,
    split_output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Curate a source FASTA, split it, and write an auditable JSON manifest.

    ``training_fasta`` is an optional independent exclusion set.  It is useful
    when a held-out source is assembled from a broader corpus: no validation or
    test sequence may occur in it.  The generated ``train`` split is always
    disjoint from validation and test by construction.
    """
    source_path = Path(source_fasta)
    manifest_path = Path(output_manifest)
    source_records = read_fasta(source_path)
    reference_paths = [Path(path) for path in reference_fastas]
    training_records = read_fasta(training_fasta) if training_fasta else []
    reference_records = {str(path): read_fasta(path) for path in reference_paths}
    reference_sequences = {
        record.sequence for records in reference_records.values() for record in records
    }
    if training_fasta:
        reference_sequences.update(record.sequence for record in training_records)

    curated, curation_counts = curate_records(
        source_records,
        reference_sequences,
        min_length=min_length,
        max_length=max_length,
    )
    counts = _allocate_counts(len(curated), validation_fraction, test_fraction)
    splits = _split_records(curated, seed, counts)
    split_sequences = {name: [record.sequence for record in records] for name, records in splits.items()}
    if set(split_sequences["validation"]) & set(split_sequences["train"]):
        raise AssertionError("Validation split overlaps training split")
    if set(split_sequences["test"]) & set(split_sequences["train"]):
        raise AssertionError("Test split overlaps training split")
    if set(split_sequences["validation"]) & set(split_sequences["test"]):
        raise AssertionError("Validation split overlaps test split")
    if any(set(sequences) & reference_sequences for sequences in split_sequences.values()):
        raise AssertionError("A generated split overlaps an exclusion FASTA")

    split_metadata = {}
    for name, records in splits.items():
        metadata: dict[str, object] = {
            "count": len(records),
            "sequence_sha256": sequence_digest(record.sequence for record in records),
            "records": [_record_dict(record) for record in records],
        }
        if split_output_dir is not None:
            split_path = Path(split_output_dir) / f"{name}.fasta"
            _write_fasta(records, split_path)
            metadata["fasta"] = str(split_path)
        split_metadata[name] = metadata

    manifest: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "silo_amp_starting_state_split",
        "seed": seed,
        "split_fractions": {"validation": validation_fraction, "test": test_fraction},
        "source_order": "first-seen FASTA order after curation; split order is seeded shuffle order",
        "curation_policy": {
            "uppercase": True,
            "allowed_alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "min_length_inclusive": min_length,
            "max_length_inclusive": max_length,
            "deduplicate": "exact sequence, keep first occurrence",
            "exclude_exact_matches_to": "training_fasta and reference_fastas",
            "counts": curation_counts,
        },
        "sources": {
            "source_fasta": _source_metadata(source_path, source_records),
            "training_fasta_exclusion": _source_metadata(Path(training_fasta), training_records)
            if training_fasta
            else None,
            "reference_fastas": {
                str(path): _source_metadata(path, reference_records[str(path)])
                for path in reference_paths
            },
        },
        "exclusion_sequence_count": len(reference_sequences),
        "exclusion_sequence_sha256": sequence_digest(sorted(reference_sequences)),
        "splits": split_metadata,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def verify_split_manifest(manifest: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if manifest split invariants are violated."""
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported split manifest version")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Manifest is missing splits")
    sequences: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        split = splits.get(name)
        if not isinstance(split, Mapping) or not isinstance(split.get("records"), list):
            raise ValueError(f"Manifest is missing {name} records")
        records = split["records"]
        values = [record.get("sequence") for record in records if isinstance(record, Mapping)]
        if len(values) != len(records) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Malformed records in {name} split")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate sequence in {name} split")
        if split.get("count") != len(values):
            raise ValueError(f"Incorrect count in {name} split")
        sequences[name] = set(values)
    if sequences["train"] & (sequences["validation"] | sequences["test"]):
        raise ValueError("Training overlaps a held-out split")
    if sequences["validation"] & sequences["test"]:
        raise ValueError("Validation overlaps test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source FASTA for the split")
    parser.add_argument("--manifest", required=True, help="Output JSON manifest")
    parser.add_argument("--training-exclusion", help="Optional FASTA whose exact sequences are excluded")
    parser.add_argument("--reference", action="append", default=[], help="Reference FASTA to exclude; repeatable")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-output-dir", help="Optional directory for train/validation/test FASTAs")
    args = parser.parse_args()
    manifest = build_split_manifest(
        args.source,
        args.manifest,
        training_fasta=args.training_exclusion,
        reference_fastas=args.reference,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_output_dir=args.split_output_dir,
    )
    verify_split_manifest(manifest)
    print(json.dumps({name: split["count"] for name, split in manifest["splits"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
