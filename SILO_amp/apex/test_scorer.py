import numpy as np
import unittest

from apex.scorer import (
    APEXEnsembleScorer,
    APEXModelError,
    APEXInputError,
    DeterministicAPEXScorer,
    PATHOGENS,
)


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, encoded):
        return np.full((len(encoded), len(PATHOGENS)), self.value)


class ScorerTests(unittest.TestCase):
    def test_ensemble_converts_and_averages_eight_models(self):
        scorer = APEXEnsembleScorer([ConstantModel(6.0), *[ConstantModel(5.0)] * 7])
        result = scorer.score(["ACDE", "ACDEF"])

        self.assertEqual(result.pathogen_mic.shape, (2, 11))
        self.assertEqual(result.model_pathogen_mic.shape, (8, 2, 11))
        # Each model is converted to MIC before the eight-model arithmetic mean.
        np.testing.assert_allclose(result.mean_mic, np.full(2, (1 + 10 * 7) / 8))
        np.testing.assert_allclose(result.optimization_score, result.mean_mic)


    def test_batching_preserves_sequence_order(self):
        scorer = APEXEnsembleScorer([ConstantModel(6.0)] * 8, batch_size=1)
        result = scorer.score(["ACDE", "ACDEF"])
        np.testing.assert_allclose(result.mean_mic, [1.0, 1.0])
        self.assertEqual(result.sequences, ("ACDE", "ACDEF"))


    def test_lower_predicted_mic_is_better(self):
        scorer = DeterministicAPEXScorer()
        result = scorer.score(["A" * 8, "A" * 12])
        self.assertLess(result.mean_mic[0], result.mean_mic[1])


    def test_input_contract_rejects_unknown_and_overlong_sequences(self):
        scorer = DeterministicAPEXScorer()
        with self.assertRaisesRegex(APEXInputError, "unsupported"):
            scorer.score(["ACX"])
        with self.assertRaisesRegex(APEXInputError, "exceeds"):
            scorer.score(["A" * 51])


    def test_real_ensemble_requires_exactly_eight_models(self):
        with self.assertRaisesRegex(APEXModelError, "exactly 8"):
            APEXEnsembleScorer([ConstantModel(6.0)] * 7)


    def test_incomplete_model_output_fails_loudly(self):
        class IncompleteModel:
            def __call__(self, encoded):
                return np.zeros((len(encoded), 10))

        scorer = APEXEnsembleScorer([IncompleteModel()] * 8)
        with self.assertRaisesRegex(APEXModelError, "11"):
            scorer.score(["ACDE"])


if __name__ == "__main__":
    unittest.main()
