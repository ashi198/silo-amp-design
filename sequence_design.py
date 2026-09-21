import copy
import hashlib
import numpy as np
import torch
from torch import nn
from config import SequenceConfig
from core.abstract import BaseTrajectory
from core.utils import softmax
from typing import Optional, List, Tuple

class SequenceDesign(BaseTrajectory):

    """Left-to-right autoregressive protein sequence design environment.

    There is a fixed action space:
      * 0: terminate
      * 1..20: append the corresponding amino acid

    """


    def __init__(self, config: SequenceConfig, inital_res: int):
        """
        Parameters:
            config [SequenceConfig]: Config
            initial_residue [int] or [List]: We always start with already one residue in the sequence to be able to diversify
                the starting point for the network.
        """

        self.config = config
        self.vocabulary_residue_idcs = list(range(0, len(self.config.residue_vocabulary))) 
        self.vocabulary_residue_names = list(self.config.residue_vocabulary.keys())
        self.residue_feasibility_mask = np.asarray(
            [not self.config.residue_vocabulary[x]["allowed"] for x in self.vocabulary_residue_names],
            dtype=bool,
        )
        self.residue_to_idx = {aa: i for i, aa in enumerate(self.config.residue_vocabulary.keys())}

        if inital_res not in self.vocabulary_residue_idcs:
            raise ValueError(f"Initial residue index {inital_res} is outside the vocabulary")
        if self.residue_feasibility_mask[inital_res]:
            raise ValueError(f"Initial residue {inital_res} is not allowed by the vocabulary")

        self.residues = []
        self.seq_string: str = None

        self.seq_list= [] # to store the actual sequence string 
        self.identifier = None
        self.TERMINATE_ACTION = 0
        self.residues.append(inital_res)

        # The action mask indicates before each action what is feasible at the current level.
        # 0: the action is masked. 1 = action is allowed.
        self.current_action_mask: Optional[np.array] = None

        # History is a list of `actions_taken` above, indicating how you get from the initial residue to the current sequence.
        self.history: List[int] = []
        self.history.append(inital_res + 1)
        self.design_done: bool = False

        # Keep track of all objectives 
        self.objective: Optional[float] = None
        self.omegAMP_prob = None
        self.apex_mean_score: Optional[float] = None
        self.apex_dict = {
            "A_baumannii": None,
            "E_coli_11775": None,
            "E_coli_AIC221": None,
            "E_coli_AIC222": None,
            "K_pneumoniae": None,
            "P_aeruginosa_PAO1": None,
            "P_aeruginosa_PA14": None,
            "S_aureus": None,
            "MRSA": None,
            "VRE_faecalis": None,
            "VRE_faecium": None,
            "apex_mic50": None, 
            "apex_mic90": None, 
            "apex_gram_negative_mean": None, 
            "apex_gram_positive_mean": None, 
            "gram_negative_selectivity": None, 
            "gram_positive_selectivity": None
        }  

        # Set this to True if anything goes wrong and the sequence will always evaluate to objective -inf
        self.infeasibility_flag: bool = False

        self.update_action_mask()
        self.update_design_sequence(new_residue=self.residues)


    def update_action_mask(self):

        """
        Creates the action mask.
        1: this action is allowed, 0: this action is not allowed.

        """

        mask = np.zeros(1 + len(self.vocabulary_residue_names), dtype=np.int64)

        # Action mask convention: 1 means feasible, 0 means masked.
        mask[1:] = (~self.residue_feasibility_mask).astype(np.int64)

        if self.optional_termination():
            mask[0] = 1
        if self.forced_termination():
            mask[1:] = 0
            mask[0] = 1

        self.current_action_mask = mask

    def update_design_sequence(self, new_residue: List):
        
        """
            Update the sequence by adding new residues.

            new_residue : List[Optional[int]]
                A list of residue indices (ints) or None values.
        """
        for residue_idx in new_residue:
            if residue_idx is None:
                self.seq_list.append(None)
            else:
                residue_alphabet = self.vocabulary_residue_names[residue_idx]
                self.seq_list.append(residue_alphabet)

    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        
        """
        Apply current_action_mask to logits and return normalized log-probs.
        
        """
        mask = self.current_action_mask.astype(bool)
        logits = logits.copy()
        logits[~mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs
    
    def make_residue_list_from_string(self, sequence: str) -> List:

        """
        Makes a list of residues idcs with virtual residue index for a gvien sequence string.

        Parameters:
            sequence: a string of protein with ther permissible 20 amino acids 

        Returns:
            residues: a list of residue idcs corresponding to the amino acids string 
        """
        residue_to_idx = {res: idx for res, idx in zip(self.vocabulary_residue_names, self.vocabulary_residue_idcs)}
        
        residues = []

        for s in sequence:
            if s in residue_to_idx.keys():
                residues.append(residue_to_idx[s])
            else:
                raise KeyError(f"Residue '{s}' not found in residue vocabulary: {list(residue_to_idx.keys())}")
        
        return residues
    
    def take_action(self, action: int):

        """
        Takes an action on the current action level and updates everything accordingly (see inline comments).
        Note that the updates are performed in-place

        Action space layout: [0 = Terminate, 1...20: residues]

        """
        if self.design_done:
            raise ValueError("Taking action on an already terminated design is invalid")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(self.current_action_mask):
            raise ValueError(f"Action {action} is outside the action space")
        action = int(action)
        
        if self.current_action_mask[action] != 1:
            raise ValueError(f"Action {action} is infeasible at sequence length {len(self.residues)}")
        
        if action == self.TERMINATE_ACTION:
            self.design_done = True
            self.seq_string = ''.join(self.seq_list)
            self.identifier = self.generate_custom_sequence_id()
        else:
            residue_idx = action - 1
            self.residues.append(residue_idx)
            self.seq_list.append(self.vocabulary_residue_names[residue_idx])

        self.history.append(int(action))
        self.update_action_mask()

    # ---- Implementation of abstract methods from `BaseTrajectory`

    @staticmethod
    def log_probability_fn(trajectories: List['SequenceDesign'], network: nn.Module, config: SequenceConfig, device: torch.device) -> List[np.array]:
        
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.

        """
        log_probs_to_return: List[np.array] = []
        device = torch.device("cpu") if device is None else device
        network.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=config.training_device, dtype=torch.bfloat16):
                batch = SequenceDesign.list_to_batch(sequences=trajectories, device=network.device, config=config)
                batch_logits = list(network(batch))
                for i, seq in enumerate(trajectories):
                    logits = batch_logits[i]
                    logits = logits.to(torch.float32).cpu().numpy()
                    log_probs_to_return.append(seq.masked_log_probs_for_current_action_level(logits))

        return log_probs_to_return
    

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_sequence= copy.deepcopy(self)
        copied_sequence.take_action(action)
        return copied_sequence, copied_sequence.design_done

    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            # assign highest possible value
            self.objective = 1000.0
            #raise ValueError("Objective is `None`. Evaluate Sequence with `SequenceObjectiveEvaluator` first.")
        return self.objective
    
    def forced_termination(self):
        return len(self.residues) >= (self.config.min_max_seq_length[1])
    
    def optional_termination(self) -> bool:
        return len(self.residues) >= self.config.min_max_seq_length[0]
    
    def is_terminable(self):
        return self.forced_termination() or self.optional_termination()

    def num_actions(self) -> int:
        
        """
        Returns number of current _feasible_ actions.
        """
        return int(self.current_action_mask.sum())

    def generate_custom_sequence_id(self):

        """
            Generates a custom identifier for a generated finished design 

        """
        if self.seq_string is None:
            raise ValueError("A sequence identifier can only be generated for a finished sequence")
        return hashlib.sha256(self.seq_string.encode("ascii")).hexdigest()[:12]
    
    @staticmethod
    def make_instance_sequence(config, sequence):

        sequence_list = list(sequence)
        min_length, max_length = config.min_max_seq_length
        if not min_length <= len(sequence_list) <= max_length:
            raise ValueError(
                f"Sequence length {len(sequence_list)} is outside the inclusive bounds "
                f"[{min_length}, {max_length}]"
            )
        if not sequence_list:
            raise ValueError("Cannot create a sequence instance from an empty sequence")
        residue_to_idx = {aa: i for i, aa in enumerate(config.residue_vocabulary.keys())}
        idx_to_residue = {i: aa for aa, i in residue_to_idx.items()}


        residue_list = [residue_to_idx[r] for r in sequence_list]
        if any(not config.residue_vocabulary[aa]["allowed"] for aa in sequence_list):
            raise ValueError("Sequence contains a residue disallowed by the vocabulary")
        seq_from_residue = [idx_to_residue[r] for r in residue_list]

        instance = SequenceDesign(config=config, inital_res = residue_list[0])
        instance.residues = residue_list
        instance.seq_list = seq_from_residue
        instance.seq_string = (''.join(instance.seq_list))
        instance.history = [r + 1 for r in residue_list]
        instance.history.append(0) #for termination
        instance.design_done = True
        instance.update_action_mask()
        instance.identifier = instance.generate_custom_sequence_id()
        
        # -------------------------
        # Verify history conversion
        # -------------------------

        history_without_eos = instance.history[:-1]

        decoded_history = [idx_to_residue[action_idx - 1] for action_idx in history_without_eos]
        history_sequence = "".join(decoded_history)

        if sequence != history_sequence:
                raise ValueError(f"Sequence conversion failed: " f"expected '{instance.seq_string}', got '{sequence}'")
        
        if instance.history[-1] != 0:
            raise ValueError("History does not terminate with EOS action 0.")

        return instance
 
    @staticmethod
    def list_to_batch(sequences: List['SequenceDesign'], include_feasibility_masks: bool = False, device: torch.device = None, 
                      config = None) -> dict:
        """
        Given a list of sequence designs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        
        * "tokens_np": direct embeddings from ESM model
        * "valid_positions": a mask tensor specifying valid positions

        if `include_feasibility_masks` is set to True, we also return
        """

        assert len(sequences) > 0, "Empty batch of sequences" 

        max_aa_per_seq = max(len(seq.residues) for seq in sequences)
        PAD_index = 21
        target_device = torch.device("cpu") if device is None else torch.device(device)
        valid_positions = torch.zeros(len(sequences), max_aa_per_seq, dtype=torch.bool, device=target_device)
        seq_tokens = torch.full((len(sequences), max_aa_per_seq), fill_value=PAD_index, device=target_device, dtype=torch.long)

        for i, seq in enumerate(sequences): 
            if seq.residues:
                valid_positions[i, 0:len(seq.residues)] = True
                residue_ids = torch.tensor(seq.residues, dtype=torch.long, device=target_device)
                seq_tokens[i, 0:len(seq.residues)] = residue_ids
        return_dict = dict(
            seq_tokens = seq_tokens,   # (B, L)
            valid_positions = valid_positions  # (B, L),

        )

        if include_feasibility_masks:
            '''# Build per-level feasibility masks.
            feasibility_mask = []
            numpy_mask = torch.from_numpy(np.stack([~seq.current_action_mask.astype(bool) for seq in sequences])).to(device)
            feasibility_mask = numpy_mask'''

            feasibility_mask = torch.from_numpy(
                np.stack([seq.current_action_mask == 0 for seq in sequences])
            ).to(target_device)

            
            # Add to return_dict
            return_dict["feasibility_mask"] = feasibility_mask 

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
            Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
        
            
    @staticmethod
    def design_sequences(config: SequenceConfig, initial_seed= None) -> List['SequenceDesign']:

        """
            Returns list of designs based on the starting point and mode.
        """
        instance = SequenceDesign(config=config, inital_res = initial_seed)
        
        return instance







