import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light checkouts
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if torch is not None:
    from model.transformer_architecture import SequenceTransformer
    from sequence_design import SequenceDesign


def make_config(min_length=2, max_length=4):
    return SimpleNamespace(
        residue_vocabulary={"A": {"allowed": True}, "B": {"allowed": True}},
        min_max_seq_length=[min_length, max_length],
        training_device="cpu",
        latent_dimension=2,
        num_heads=1,
        num_transformer_blocks=0,
        dropout=0.0,
    )


@unittest.skipIf(torch is None, "PyTorch is required for sequence-state tests")
class SequenceStateTests(unittest.TestCase):
    def test_action_mask_termination_bounds_and_feasible_count(self):
        config = make_config(min_length=2, max_length=3)
        sequence = SequenceDesign(config, 0)

        self.assertEqual(sequence.current_action_mask.tolist(), [0, 1, 1])
        self.assertEqual(sequence.num_actions(), 2)
        with self.assertRaises(ValueError):
            sequence.take_action(0)

        sequence.take_action(1)
        self.assertEqual(sequence.current_action_mask.tolist(), [1, 1, 1])
        self.assertEqual(sequence.num_actions(), 3)
        sequence.take_action(0)
        self.assertEqual(sequence.seq_string, "AA")
        self.assertEqual(sequence.history, [1, 1, 0])
        self.assertEqual(sequence.identifier, "58bb119c3551")

        with self.assertRaises(ValueError):
            sequence.take_action(1)

    def test_disallowed_residues_are_masked(self):
        config = make_config()
        config.residue_vocabulary["B"]["allowed"] = False
        sequence = SequenceDesign(config, 0)
        self.assertEqual(sequence.current_action_mask.tolist(), [0, 1, 0])
        self.assertEqual(sequence.num_actions(), 1)
        with self.assertRaises(ValueError):
            sequence.take_action(2)

    def test_reconstruction_is_deterministic_and_validates_length(self):
        config = make_config(min_length=2, max_length=4)
        first = SequenceDesign.make_instance_sequence(config, "AB")
        second = SequenceDesign.make_instance_sequence(config, "AB")
        self.assertEqual(first.seq_string, "AB")
        self.assertEqual(first.history, [1, 2, 0])
        self.assertEqual(first.identifier, second.identifier)

        with self.assertRaises(ValueError):
            SequenceDesign.make_instance_sequence(config, "A")

    def test_batch_feasibility_mask_matches_each_sequence(self):
        config = make_config(min_length=2, max_length=3)
        short = SequenceDesign(config, 0)
        long = SequenceDesign(config, 0)
        long.take_action(1)
        batch = SequenceDesign.list_to_batch(
            [short, long], include_feasibility_masks=True, device=torch.device("cpu"), config=config
        )
        self.assertEqual(batch["seq_tokens"].tolist(), [[0, 21], [0, 0]])
        self.assertEqual(batch["valid_positions"].tolist(), [[True, False], [True, True]])
        self.assertEqual(batch["feasibility_mask"].tolist(), [[True, False, False], [False, False, False]])

    def test_transformer_reads_each_row_final_valid_position(self):
        config = make_config(min_length=1, max_length=4)
        short = SequenceDesign(config, 0)
        long = SequenceDesign(config, 0)
        long.take_action(2)
        network = SequenceTransformer(config)
        with torch.no_grad():
            network.token_embedding.weight.zero_()
            network.token_embedding.weight[0].fill_(1.0)
            network.token_embedding.weight[1].fill_(2.0)
            network.position_embedding.weight.zero_()
            network.action_head.weight.zero_()
            network.action_head.bias.zero_()
            network.action_head.weight[0, 0] = 1.0

        batch = SequenceDesign.list_to_batch([short, long], device=torch.device("cpu"), config=config)
        logits = network(batch)
        self.assertEqual(logits[:, 0].tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
