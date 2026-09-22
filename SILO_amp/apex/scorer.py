"""Isolated APEX ensemble scoring contract.

APEX models are trained to predict ``-log10(MIC / 1,000,000)``.  This module
owns the conversion back to MIC (in uM), the eight-model ensemble contract,
and the lower-is-better optimization score.  It deliberately does not own
candidate filtering or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

try:  # Torch is needed for real APEX models, but not for the test double.
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    torch = None  # type: ignore[assignment]

PATHOGENS: tuple[str, ...] = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)
EXPECTED_MODEL_COUNT = 8
MAX_SEQUENCE_LENGTH = 50
MODEL_OUTPUT_TRANSFORM = 6.0
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _make_vocab() -> dict[str, int]:
    """Return the APEX token vocabulary without importing training utilities."""
    return {
        token: index
        for index, token in enumerate(("0", "1", "2", *"ACDEFGHIKLMNPQRSTVWY"))
    }


def _onehot_encoding(sequences: Sequence[str], max_len: int) -> np.ndarray:
    """Encode APEX sequences as padded integer token arrays."""
    vocabulary = _make_vocab()
    encoded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        tokens = [vocabulary["1"]]
        tokens.extend(vocabulary[residue] for residue in sequence)
        tokens.append(vocabulary["2"])
        if len(tokens) > max_len:
            raise APEXInputError(f"Sequence {row} exceeds the APEX encoding length.")
        encoded[row, : len(tokens)] = tokens
    return encoded


class APEXScoringError(RuntimeError):
    """Base error for invalid APEX scorer configuration or model output."""


class APEXModelError(APEXScoringError):
    """Raised when the required ensemble models are absent or incomplete."""


class APEXInputError(ValueError, APEXScoringError):
    """Raised when sequences cannot be scored by the APEX input contract."""


class APEXModel(Protocol):
    """Minimum callable interface required from an APEX model."""

    def __call__(self, encoded_sequences: object) -> object:
        ...


@dataclass(frozen=True)
class APEXScoreBatch:
    """Complete score output, retaining model and pathogen diagnostics."""

    sequences: tuple[str, ...]
    pathogen_mic: np.ndarray
    model_pathogen_mic: np.ndarray
    mean_mic: np.ndarray

    @property
    def pathogen_names(self) -> tuple[str, ...]:
        return PATHOGENS

    @property
    def optimization_score(self) -> np.ndarray:
        """The objective used by SILO: lower predicted mean MIC is better."""
        return self.mean_mic


class APEXEnsembleScorer:
    """Batch scorer for exactly eight APEX models and eleven pathogens.

    Models are injected to keep this contract independent of model discovery,
    CUDA, and filesystem state.  ``from_directory`` is the explicit real-model
    construction path; it fails loudly unless exactly eight model files exist.
    """

    def __init__(
        self,
        models: Sequence[APEXModel],
        *,
        device: str = "cpu",
        batch_size: int = 1000,
        encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> None:
        if len(models) != EXPECTED_MODEL_COUNT:
            raise APEXModelError(
                f"APEX requires exactly {EXPECTED_MODEL_COUNT} models; found {len(models)}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if device != "cpu" and torch is None:
            raise APEXModelError("A non-CPU APEX device requires PyTorch.")
        self.models = tuple(models)
        self.device = device
        self.batch_size = batch_size
        self._encoder = encoder or self._encode

    @classmethod
    def from_directory(
        cls,
        model_directory: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 1000,
        loader: Callable[[Path, str], APEXModel] | None = None,
    ) -> "APEXEnsembleScorer":
        """Load and validate the complete eight-model APEX ensemble."""
        directory = Path(model_directory)
        if not directory.is_dir():
            raise APEXModelError(f"APEX model directory does not exist: {directory}")
        paths = tuple(sorted(path for path in directory.glob("APEX_*") if path.is_file()))
        if len(paths) != EXPECTED_MODEL_COUNT:
            raise APEXModelError(
                f"APEX requires exactly {EXPECTED_MODEL_COUNT} model files in {directory}; "
                f"found {len(paths)}."
            )
        if torch is None:
            raise APEXModelError("Loading APEX models requires PyTorch.")
        model_loader = loader or (lambda path, target: torch.load(path, map_location=target, weights_only=False))
        models = []
        for path in paths:
            try:
                model = model_loader(path, device)
                if not callable(model):
                    raise TypeError("loaded object is not callable")
                if hasattr(model, "eval"):
                    model.eval()
                models.append(model)
            except Exception as exc:
                raise APEXModelError(f"Could not load complete APEX model {path}: {exc}") from exc
        return cls(models, device=device, batch_size=batch_size)

    def score(self, sequences: Iterable[str]) -> APEXScoreBatch:
        """Return per-model, per-pathogen, and aggregate MIC diagnostics."""
        normalized = self._validate_sequences(sequences)
        if not normalized:
            empty = np.empty((0, len(PATHOGENS)), dtype=np.float64)
            return APEXScoreBatch((), empty, np.empty((EXPECTED_MODEL_COUNT, 0, len(PATHOGENS))), np.empty(0))

        model_scores = []
        for model in self.models:
            batches = []
            for start in range(0, len(normalized), self.batch_size):
                encoded = self._encoder(normalized[start : start + self.batch_size])
                raw = self._model_output(model, encoded)
                batches.append(self._raw_to_mic(raw))
            predictions = np.concatenate(batches, axis=0)
            if predictions.shape != (len(normalized), len(PATHOGENS)):
                raise APEXModelError(
                    "APEX model output changed shape after batching: "
                    f"expected {(len(normalized), len(PATHOGENS))}, got {predictions.shape}."
                )
            model_scores.append(predictions)

        per_model = np.stack(model_scores, axis=0)
        pathogen_mic = per_model.mean(axis=0)
        mean_mic = pathogen_mic.mean(axis=1)
        return APEXScoreBatch(tuple(normalized), pathogen_mic, per_model, mean_mic)

    def _encode(self, sequences: Sequence[str]) -> np.ndarray:
        return _onehot_encoding(sequences, MAX_SEQUENCE_LENGTH + 2)

    def _model_output(self, model: APEXModel, encoded: np.ndarray) -> np.ndarray:
        try:
            if torch is not None:
                tensor = torch.as_tensor(encoded, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    output = model(tensor)
            else:  # pragma: no cover
                output = model(encoded)
        except Exception as exc:
            raise APEXModelError(f"APEX model prediction failed: {exc}") from exc
        if torch is not None and isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        return np.asarray(output, dtype=np.float64)

    @staticmethod
    def _raw_to_mic(raw_output: np.ndarray) -> np.ndarray:
        values = np.squeeze(raw_output)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(PATHOGENS):
            raise APEXModelError(
                f"APEX models must return [batch, {len(PATHOGENS)}] values; got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise APEXModelError("APEX model output contains non-finite values.")
        return np.power(10.0, MODEL_OUTPUT_TRANSFORM - values)

    @staticmethod
    def _validate_sequences(sequences: Iterable[str]) -> list[str]:
        normalized = list(sequences)
        for index, sequence in enumerate(normalized):
            if not isinstance(sequence, str) or not sequence:
                raise APEXInputError(f"Sequence {index} must be a non-empty string.")
            sequence = sequence.upper()
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                raise APEXInputError(f"Sequence {index} exceeds {MAX_SEQUENCE_LENGTH} residues.")
            invalid = sorted(set(sequence) - AMINO_ACIDS)
            if invalid:
                raise APEXInputError(f"Sequence {index} contains unsupported residues: {invalid}.")
            normalized[index] = sequence
        return normalized


class DeterministicAPEXScorer(APEXEnsembleScorer):
    """Dependency-free deterministic test double with the same score contract."""

    def __init__(self, *, pathogen_weights: Sequence[float] | None = None) -> None:
        weights = np.asarray(pathogen_weights if pathogen_weights is not None else np.arange(11), dtype=float)
        if weights.shape != (len(PATHOGENS),) or not np.isfinite(weights).all():
            raise ValueError(f"pathogen_weights must contain {len(PATHOGENS)} finite values.")
        self._weights = weights

    def score(self, sequences: Iterable[str]) -> APEXScoreBatch:
        normalized = self._validate_sequences(sequences)
        pathogen = np.asarray(
            [[float(len(sequence)) + weight for weight in self._weights] for sequence in normalized],
            dtype=np.float64,
        ).reshape(len(normalized), len(PATHOGENS))
        per_model = np.repeat(pathogen[None, :, :], EXPECTED_MODEL_COUNT, axis=0)
        return APEXScoreBatch(tuple(normalized), pathogen, per_model, pathogen.mean(axis=1))
