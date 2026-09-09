import unittest

from silo_amp.candidate_selection import SelectionPolicy, local_similarity, select_candidates


def candidate(identifier, sequence, mic=10.0, **metrics):
    return {
        "id": identifier,
        "sequence": sequence,
        "mean_predicted_mic": mic,
        "marlys_identity_pass": True,
        "charge": 5.0,
        "hydrophobicity": 0.2,
        "cysteine_count": 0,
        "max_hydrophobic_run": 2,
        **metrics,
    }


class CandidateSelectionTests(unittest.TestCase):
    def test_basic_filters_are_ordered_and_counted(self):
        result = select_candidates(
            [
                candidate("bad-alphabet", "A" * 7 + "Z"),
                candidate("reference", "ACDEFGHI", mic=1),
                candidate("first", "ACDEFGHIK", mic=3),
                candidate("duplicate", "ACDEFGHIK", mic=2),
            ],
            references={"ACDEFGHI"},
            top_k=1,
        )
        self.assertEqual([item["id"] for item in result.selected], ["first"])
        self.assertEqual(result.rejection_counts["alphabet"], 1)
        self.assertEqual(result.rejection_counts["reference_exact_match"], 1)

    def test_thresholds_are_inclusive_except_diversity(self):
        result = select_candidates(
            [
                candidate("boundary", "ACDEFGHI", mic=64, charge=2, hydrophobicity=-0.5),
                candidate("too-similar", "ACDEFGHV", mic=65),
            ],
            top_k=2,
        )
        self.assertEqual([item["id"] for item in result.selected], ["boundary"])
        self.assertEqual(result.rejection_counts["mic_threshold"], 1)
        self.assertEqual(local_similarity("ACDEFGHI", "ACDEFGHV"), 7 / 8)

    def test_mic_order_and_diversity_keep_lower_mic_candidate(self):
        result = select_candidates(
            [candidate("later", "ACDEFGHV", mic=2), candidate("winner", "ACDEFGHI", mic=1)],
            top_k=2,
        )
        self.assertEqual([item["id"] for item in result.selected], ["winner"])
        self.assertEqual(result.rejection_counts["diversity"], 1)


if __name__ == "__main__":
    unittest.main()
