from __future__ import annotations
import argparse
import random
from pathlib import Path


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """
    Read FASTA records.

    Returns
    -------
    list of (header, sequence)
    """
    path = Path(path)

    records = []
    header = None
    seq_parts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(seq_parts).upper()
                    records.append((header, sequence))

                header = line[1:].strip()
                seq_parts = []

            else:
                seq_parts.append(line)

    if header is not None:
        sequence = "".join(seq_parts).upper()
        records.append((header, sequence))

    return records


def write_fasta(
    records: list[tuple[str, str]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for header, sequence in records:
            f.write(f">{header}\n")
            f.write(f"{sequence}\n")


def is_canonical(sequence: str) -> bool:
    """
    True if sequence contains only the 20 canonical amino acids.
    """
    return (
        len(sequence) > 0
        and set(sequence).issubset(CANONICAL_AA)
    )


def prepare_peptide_dataset(
    input_fasta: str | Path,
    train_fasta: str | Path,
    validation_fasta: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 42,

) -> None:
    """
    Read, filter, deduplicate, shuffle, and split peptide sequences.
    """

    records = read_fasta(input_fasta)
    print(f"Loaded records: {len(records)}")
    canonical_records = [(header, seq) for header, seq in records if is_canonical(seq)]

    print(f"After canonical-AA filtering: " f"{len(canonical_records)}")
    print(f"Removed noncanonical/empty: " f"{len(records) - len(canonical_records)}")

    # 2. Exact deduplication

    unique_by_sequence = {}

    for header, sequence in canonical_records:
        if sequence not in unique_by_sequence:
            unique_by_sequence[sequence] = header

    unique_records = [(header, sequence) for sequence, header in unique_by_sequence.items()]

    print(f"After exact deduplication: " f"{len(unique_records)}")
    print(f"Exact duplicates removed: " f"{len(canonical_records) - len(unique_records)}")

    rng = random.Random(seed)
    rng.shuffle(unique_records)

    # -----------------------------
    # 4. Train/validation split
    # -----------------------------
    num_validation = max(1, round(len(unique_records) * validation_fraction))
    validation_records = unique_records[:num_validation]
    train_records = unique_records[num_validation:]

    # -----------------------------
    # 5. Write FASTA files
    # -----------------------------
    write_fasta(train_records, train_fasta)
    write_fasta(validation_records, validation_fasta)

    print(f"Train sequences:      {len(train_records)}")
    print(f"Validation sequences: {len(validation_records)}")
    print(f"Train FASTA:          {train_fasta}")
    print(f"Validation FASTA:     {validation_fasta}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a peptide FASTA dataset by filtering to "
            "canonical amino acids, removing exact duplicates, "
            "and randomly splitting into train and validation sets."
        )
    )

    parser.add_argument("--input", default='./training/training_data/final_SILO_training_dataset.fasta',  help="Input FASTA file.")
    parser.add_argument("--train-output", default="./training/training_data/final_train.fasta", help="Output training FASTA.")
    parser.add_argument("--validation-output", default="./training/training_data/final_validation.fasta", help="Output validation FASTA.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Fraction of data used for validation. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for splitting. Default: 42")
    args = parser.parse_args()

    prepare_peptide_dataset(input_fasta=args.input, train_fasta=args.train_output, validation_fasta=args.validation_output,
        validation_fraction=args.validation_fraction, seed=args.seed)


if __name__ == "__main__":
    main()