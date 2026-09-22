from typing import Optional, Tuple, List, Dict

import torch
import pickle
import random
from torch.utils.data import Dataset
import copy
from .config import SequenceConfig
from .sequence_design import SequenceDesign
from tqdm import tqdm
from pathlib import Path

def _clone_sequence(seq: SequenceDesign) -> SequenceDesign:
    seq_copy = copy.deepcopy(seq)
    return seq_copy

def precompute_flat_dataset(instances, config, if_pretrain: None):

    """
    Computes flattened training datapoints from instances and saves all partial states. 
    We want to uniformly sample from partial sequences. So for each instance, check how many partial mutated sequences
    there are, and create a list of them where each entry is a tuple (int, int), where first entry is index of
    the instance, and second entry is the index in the action sequence which is the training target.

    """

    flat_sequences = [] # state BEFORE taking the target action
    flat_targets = [] # the target action (int)
    tuple_to_flat : Dict[Tuple[int, int], int] = {}
    targets_to_sample : List[Tuple[int, int]] = [] # (instance_idx, target_idx)


    if if_pretrain:
        iterable = instances.values()
    else:
        iterable = instances

    for i, instance in tqdm(enumerate(iterable), total=len(iterable), desc="Computing dataset"):
        seq = SequenceDesign(config, inital_res=instance["residues"][:1])
        sequence_of_actions_idx = list(range(len(instance["action_seq"][1:])))
        targets_to_sample.extend([(i, j) for j in sequence_of_actions_idx])

        for j, action in enumerate(instance["action_seq"][2:]):
            # store state before taking this action
            flat_idx = len(flat_targets)
            tuple_to_flat[(i, j)] = flat_idx

            seq_copy = _clone_sequence(seq)
            flat_sequences.append(seq_copy)
            flat_targets.append(action)

            seq.take_action(action)

    return {
        "flat_sequences": flat_sequences,
        "flat_targets": flat_targets,
        "tuple_to_flat": tuple_to_flat,
        "targets_to_sample": targets_to_sample,
    }



class PolicyTrainingDataset(Dataset):

    """

    Dataset for supervised training of the protein sequence design given as a list pseudo-expert sequence.
    Each sequence is given as a dictionary with the following keys and values
          "start_residue": [int] the int representing the residue from which to start
          "action_seq": List[List[int]] Actions which need to be taken on each index to create the sequence
          "peptide": [str] Corresponding peptide string
          "obj": [float] Objective function evaluation

    Each datapoint in this dataset is a partial sequence: We sample an instance, randomly choose an index up to which
    all actions will be performed. Then, ending up at action index 0, we take the next item in the action seq
    (which corresponds to a list all actions that need to be taken from index to index) as training target.

    """
    def __init__(self, config: SequenceConfig, path_to_pickle: str, batch_size: int, custom_num_batches: Optional[int],
                 no_random: bool = False, if_pretrain: bool =False, if_train: bool =False):
        self.config = config
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.path_to_pickle = path_to_pickle
        self.if_pretrain = if_pretrain
        self.if_train = if_train
        
        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f)  # list of dictionaries

        if config.if_pretrain: 
            if self.if_train:
                cached_path = Path(f"./pretrain/full_training_dataset.pickle")
            else:
                cached_path = Path(f"./pretrain/full_val_dataset.pickle")

            if cached_path.exists():
                print(f"Loading precomputed dataset from {cached_path}")
                with open(cached_path, "rb") as f:
                    cached = pickle.load(f)
            else:
                print(f"Precomputing dataset (first time). This may take a while...")
                cached = precompute_flat_dataset(self.instances, config, if_pretrain=True)
                cached_path.parent.mkdir(parents=True,exist_ok=True)
                with open(cached_path, "wb") as f:
                    pickle.dump(cached, f)
                print(f"Saved precomputed dataset to {cached_path}")
        else:
            cached = precompute_flat_dataset(instances=self.instances, config=config, if_pretrain=self.if_pretrain)

        self.targets_to_sample = cached["targets_to_sample"]
        self._flat_sequences = cached["flat_sequences"]
        self._flat_targets = cached["flat_targets"]
        self._tuple_to_flat = cached["tuple_to_flat"]

        if custom_num_batches is None:
            self.length = len(self.targets_to_sample) // self.batch_size 
        else:
            self.length = custom_num_batches

        self.no_random = no_random

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        """
        :param idx: is not used, as we directly randomly sample a full batch from the datapoints here.

        Returns: Dictionary with keys:

        """
        partial_sequences: List[SequenceDesign] = []   # partial sequences which will become the batch
        instance_targets: List[List[int]] = []  # corresponding targets taken from the instances

        if self.no_random:
            batch_to_pick = self.targets_to_sample[idx * self.batch_size: (idx+1) * self.batch_size]
        else:
            batch_to_pick = random.choices(self.targets_to_sample, k=self.batch_size)  # with replacement

        # Map each (instance_idx, target_idx) to the precomputed flat index
        flat_indices = [self._tuple_to_flat[tup] for tup in batch_to_pick]

        # Gather precomputed partial sequences and their targets/levels
        partial_sequences = [self._flat_sequences[k] for k in flat_indices]
        instance_targets  = [self._flat_targets[k]   for k in flat_indices]

        # Create the input batch from the partial sequences.
        batch_input = SequenceDesign.list_to_batch(sequences=partial_sequences, device=torch.device("cpu"), include_feasibility_masks=True, 
                                                   config=self.config)

        # We now create the targets
        targets = torch.tensor(instance_targets, dtype=torch.long)    

        return dict(
            input=batch_input,
            targets=targets
        )
