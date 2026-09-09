from typing import List, Union
import numpy as np
from config import SequenceConfig
import os, ray, torch
from sequence_design import SequenceDesign
from evaluation_metrics.utils import APEXEnsemble
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from Bio import Align
from evaluation_metrics.utils import read_fasta_sequences, check_amp_synthesizability, check_sequence_for_hydrophobic_clusters
from evaluation_metrics.metrics_utils import local_similarity
STANDARD_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

@ray.remote
class PredictorWorker:
    def __init__(self, config: SequenceConfig, device: torch.device):

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        self.device = device
        self.config = config

class SequenceEvaluator:
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)] 
        self.apex_ensemble = APEXEnsemble(self.config, self.device)
        self.peptide_checks = PeptideChecks(self.config)

    def calculate_apex_scores(self, sequences:List[Union[SequenceDesign, str]]):
        
        mic_scores = self.apex_ensemble.calculate_mic_scores(sequences)

        for i, (seq, scores) in enumerate(zip(sequences, mic_scores)):
            seq.apex_dict["A_baumannii"] = scores[0]
            seq.apex_dict["E_coli_11775"] = scores[1]
            seq.apex_dict["E_coli_AIC221"] = scores[2]
            seq.apex_dict["E_coli_AIC222"] = scores[3]
            seq.apex_dict["K_pneumoniae"] = scores[4]
            seq.apex_dict["P_aeruginosa_PAO1"] = scores[5]
            seq.apex_dict["P_aeruginosa_PA14"] = scores[6]
            seq.apex_dict["S_aureus"] = scores[7]
            seq.apex_dict["MRSA"] = scores[8]
            seq.apex_dict["VRE_faecalis"] = scores[9]
            seq.apex_dict["VRE_faecium"] = scores[10]
            seq.apex_mean_score = float(np.mean(scores))
            seq.apex_dict["apex_mic50"] = float(np.median(scores))
            seq.apex_dict["apex_mic90"] = float(np.quantile(scores, 0.90))

        return np.mean(mic_scores, axis=1)
    
class PeptideChecks:
    def __init__(self, config):
        self.config = config
        self.reference_fasta_path = './objectives/reference_data/antibacterial.fasta'
        self.reference_sequences = read_fasta_sequences(self.reference_fasta_path)
        self.STANDARD_AMINO_ACIDS = STANDARD_ALPHABET
           
    def basic_validity_mask(self, 
                            candidates,
                            seen):
    
        mask: list[bool] = []
        min_length = self.config.min_max_seq_length[0]
        max_length = self.config.min_max_seq_length[1]
        for seq in candidates:
            valid = (
                min_length <= len(seq['peptide']) <= max_length
                and set(seq['peptide']).issubset(self.STANDARD_AMINO_ACIDS)
                and seq['peptide'] not in seen
                and seq['peptide'] not in self.reference_sequences)
            
            mask.append(valid)

        return mask
    
    def synthesis_based_masking(self, candidates):
        mask: list[bool] = []
        for seq in candidates:
            valid = (check_sequence_for_hydrophobic_clusters(seq['peptide']) 
                     and check_amp_synthesizability(sequence=seq['peptide']))
            mask.append(valid)

        valid_sequences = [sequence for sequence, valid in zip(candidates, mask) if valid]
        return valid_sequences
    
@dataclass(frozen=True)
class SelectionPolicy:
    min_length: int = 8
    max_length: int = 50
    mic_threshold: float = 64.0
    marlys_identity_limit: float = 0.80
    charge_range: tuple[float, float] = (2.0, 10.0)
    hydrophobicity_range: tuple[float, float] = (-0.5, 0.8)
    hydrophobic_moment_cutoff: float = (0.2, 0.6), 
    max_cysteines: int = 1
    max_hydrophobic_run: int = 3
    diversity_similarity_limit: float = 0.40


@dataclass
class SelectionResult:
    selected: list[dict[str, Any]]
    valid_50k: list[dict[str, Any]]
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, list[str]] = field(default_factory=dict)

def _inc(result: SelectionResult, candidate_id: str, reason: str) -> None:
    result.rejection_counts[reason] = result.rejection_counts.get(reason, 0) + 1
    result.rejection_reasons.setdefault(candidate_id, []).append(reason)


def _valid_top(candidate: Mapping[str, Any], policy: SelectionPolicy) -> str | None:
    try:
        mic = float(candidate["mean_predicted_mic"])
    except (KeyError, TypeError, ValueError):
        return "missing_mic"
    if not bool(candidate.get("marlys_identity_pass", False)):
        return "marlys identity over 80%"
    if not bool(candidate.get("passes_levenshtein_check", False)):
        return "levenshtein identity over 80%"
    charge = candidate.get("charge")
    hydrophobicity = candidate.get("hydrophobicity")
    hydrophobic_moment = candidate.get("amphipathicity")
    if charge is None or not policy.charge_range[0] <= float(charge) <= policy.charge_range[1]:
        return "charge"
    if hydrophobicity is None or not policy.hydrophobicity_range[0] <= float(hydrophobicity) <= policy.hydrophobicity_range[1]:
        return "hydrophobicity"
    if hydrophobic_moment is None or not policy.hydrophobic_moment_cutoff[0]  <= float(hydrophobic_moment) <= policy.hydrophobic_moment_cutoff[1]:
        return "amphipathicity"
    if int(candidate.get("cysteine_count", -1)) > policy.max_cysteines:
        return "cysteine_count"
    if int(candidate.get("max_hydrophobic_run", policy.max_hydrophobic_run + 1)) > policy.max_hydrophobic_run:
        return "hydrophobic_run"
    return None

def select_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    references: Iterable[str] = (),
    marlys_references: Iterable[str] = (),
    policy: SelectionPolicy = SelectionPolicy(),
    top_k: int = 100,
) -> SelectionResult:
    """Validate a candidate population and greedily select a diverse top-K.

    Valid candidates are ranked by lower MIC, then
    sequence, then identifier.  Duplicate sequences are rejected before scoring.
    Diversity is checked against already-selected candidates and a candidate is
    rejected only when local similarity is strictly greater than the limit.
    
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reference_set = set(references)
    marlys_set = set(marlys_references)
    result = SelectionResult(selected=[], valid_50k=[])
    seen: set[str] = set()

    for raw in candidates:
        candidate = dict(raw)
        candidate_id = str(candidate.get("id", "<missing-id>"))
        sequence = candidate.get("sequence")
        basic_reason = _valid_basic(candidate, reference_set, marlys_set, policy)
        if basic_reason:
            _inc(result, candidate_id, basic_reason)
            continue
        if sequence in seen:
            _inc(result, candidate_id, "duplicate_sequence")
            continue
        seen.add(sequence)
        result.valid_50k.append(candidate)

    ranked = sorted(
        result.valid_50k,
        key=lambda item: (
            float(item.get("mean_predicted_mic", float("inf"))),
            str(item["sequence"]),
            str(item.get("id", "")),
        ),
    )
    eligible: list[dict[str, Any]] = []
    for candidate in ranked:
        candidate_id = str(candidate.get("id", "<missing-id>"))
        top_reason = _valid_top(candidate, policy)
        if top_reason:
            _inc(result, candidate_id, top_reason)
            continue
        if any(
            local_similarity(candidate["sequence"], chosen["sequence"])
            > policy.diversity_similarity_limit
            for chosen in eligible
        ):
            _inc(result, candidate_id, "diversity")
            continue
        eligible.append(candidate)
        if len(eligible) == top_k:
            break

    result.selected = [dict(item, rank=rank) for rank, item in enumerate(eligible, 1)]
    return result

def _valid_basic(candidate: Mapping[str, Any], references: set[str], marlys_set: set[str], policy: SelectionPolicy) -> str | None:
    sequence = candidate.get("sequence")
    if not isinstance(sequence, str):
        return "invalid_sequence"
    if not policy.min_length <= len(sequence) <= policy.max_length:
        return "length"
    if not set(sequence).issubset(STANDARD_ALPHABET):
        return "alphabet"
    if sequence in references:
        return "antibacterial.fasta_exact_match"
    if sequence in marlys_set:
        return "marlys_database_exact_match"
    return None





    








    
