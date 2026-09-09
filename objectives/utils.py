import torch, os
import numpy as np 
import numpy as np
from apex import APEX_models
from apex.utils import onehot_encoding, make_vocab
import glob, math
from pathlib import Path
import sys
sys.modules["APEX_models"] = APEX_models
from Bio import Align
import Levenshtein
from evaluation_metrics.metrics_utils import calculate_hydrophobicity, calculate_hydrophobicmoment, calculate_charge

MIC_THRESHOLD = 64.0  # (1) activity: keep APEX mean MIC <= 64 uM
NOVELTY_SIMILARITY = 0.60  # (2) exclude local-alignment similarity > 0.60 to known AMPs
DIVERSITY_SIMILARITY = 0.40  # (3) among survivors, > 0.40 pair -> keep the lower-MIC peptide
SIMILARITY_THRESHOLD = 0.80 # A final guard enforces the challenge's own < 80% Levenshtein rule vs data/antibacterial.fasta


def select_top_candidates(
    config,
    generated_peptide_df: pd.DataFrame,
    output_fasta: str,
    output_csv: str | None = None,
    top_k: int = 100,
    mic_threshold: float = 64.0,
    diversity_threshold: float = 0.40,
    evalutor= None) -> pd.DataFrame:

    """
    Select top AMP candidates from a precomputed metrics dataframe.
    Based on Select the top-``top_k`` via the paper's computational filtering (Torres et al. 2025)

    Expected columns:
        - id
        - sequence
        - apex_mean_mic
        - passes_marlys_80
        - passes_local_similarity_check
        - passes_levenshtein_check

    Selection:
        1. Rank by ascending APEX mean MIC. Prefer peptides with APEX mean MIC <= ``mic_threshold`` (64 uM),
        2. Keep only peptides passing:
            - MarLys <=80% MMSeq2 identity
            - local-similarity novelty check
            - Levenshtein novelty check from antibacterial.fasta file
        3. Greedily enforce internal diversity among selected peptides.
        4. Save selected candidates as CSV and FASTA.
    """

    required_columns = {"id", "sequence", "apex_mean_mic", "passes_marlys_80", "passes_local_similarity_check", "passes_levenshtein_check"}
    missing = required_columns - set(generated_peptide_df.columns)
    top_k = config.top_k_peptides

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = generated_peptide_df.copy()

    # Sanity checks
    if not df["id"].is_unique:
        raise ValueError("Sequence IDs must be unique.")

    if df["sequence"].duplicated().any():
        raise ValueError("Duplicate sequences found.")

    # Remove missing MIC predictions
    df = df.dropna(subset=["apex_mean_mic"])

    #First do synthesis related checkes
    synthesis_results = evalutor.peptide_checks.synthesis_based_masking(df["sequence"].tolist())
    synthesis_passed_sequences = {candidate["peptide"] for candidate in synthesis_results}
    df["passes_synthesis_check"] = df["sequence"].isin(synthesis_passed_sequences)

    
    filter_mask = (df["passes_marlys_80"].fillna(False) & df["passes_local_similarity_check"].fillna(False) & df["passes_levenshtein_check"].fillna(False)
                   & df["passes_synthesis_check"].fillna(False))
    filtered_df = df[filter_mask].copy()

    # Rank by predicted MIC
    filtered_df = filtered_df.sort_values("apex_mean_mic", ascending=True).reset_index(drop=True)

    # Greedy diversity filtering
    selected_indices = []
    selected_sequences = []

    n_failed_diversity = 0

    for idx, row in filtered_df.iterrows():

        seq = row["sequence"]

        if any(row["max_train_reference_similarity"] > diversity_threshold for selected_seq in selected_sequences):
            n_failed_diversity += 1
            continue

        selected_indices.append(idx)
        selected_sequences.append(seq)

        if len(selected_indices) == top_k:
            break

    if len(selected_indices) < top_k:
        raise RuntimeError(f"Only {len(selected_indices)} peptides passed all filters " f"and diversity constraints; requested {top_k}.")

    # Preserve all original dataframe columns
    top_df = (filtered_df.loc[selected_indices].copy().reset_index(drop=True))
    top_df.insert(0,"rank", range(1, len(top_df) + 1))

    # Save FASTA
    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    with open(output_fasta, "w") as f:
        for _, row in top_df.iterrows():
            f.write(f">{row['id']}" f"|rank={row['rank']}" f"|apex_mean_mic={row['apex_mean_mic']:.4f}\n")
            f.write(f"{row['sequence']}\n")

    # Save CSV

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True,
        )
        top_df.to_csv(output_csv, index=False)

    return top_df


def candidate_selection_for_SILO_training(trajectories, 
                          seen_protein_smiles, config) -> np.array:
    

    """
    Candidate selection for choosing top trajectories to train from. The following requirements should be met:
        -- Use only the 20 standard proteinogenic amino acids (checked in basic_validity_mask function)
        -- the length of the peptide must be better a permissible range (8 to 50) (checked in basic_validity_mask function)
        -- Be unique (no duplicates) (checked in basic_validity_mask function)
        -- Must be unique from the known antibacterial peptides in antibacterial.fasta file. 
    """

    local_seen = set()
    new_unique = []

    if isinstance(trajectories, dict):
        iterator = [traj for _, traj in trajectories.items()]
    elif isinstance(trajectories, list):
        iterator = [traj for traj in trajectories] 

    # Identify which trajectories are new unique SMILES
    for traj in iterator:
        peptide = traj['peptide']

        # Only keep peptides between 8 and 50 amino acids
        if not config.min_max_seq_length[0] <= len(peptide) <= config.min_max_seq_length[1]:
            continue

        # duplicate within this batch
        if peptide in local_seen:
            continue
        local_seen.add(peptide)

        # already evaluated previously (global cache, which includes all generated sequences as well as sequences from antibacterial.fasta)
        if peptide not in seen_protein_smiles:
            new_unique.append(traj)

    final_trajs = sorted(new_unique, key=lambda x: x['objective'], reverse=config.max_objective)
    return final_trajs


def normalize_sequence(sequence: str) -> str:
    """
    Normalize a peptide sequence.

    Whitespace is removed and letters are converted to uppercase.
    """
    return "".join(str(sequence).split()).upper()


def read_fasta_sequences(path: str | Path) -> list[str]:
    """

    Read sequences from a FASTA file.

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    sequences: list[str] = []
    sequence_parts: list[str] = []

    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if sequence_parts:
                    sequences.append(
                        normalize_sequence("".join(sequence_parts))
                    )
                    sequence_parts = []
            else:
                sequence_parts.append(line)

    if sequence_parts:
        sequences.append(
            normalize_sequence("".join(sequence_parts))
        )

    return sequences

def check_amp_synthesizability(
    sequence: str,
    charge_range: tuple[float, float] = (2.0, 10.0),
    hydrophobicity_range: tuple[float, float] = (-0.5, 0.8),
    hydrophobic_moment_cutoff: float = (0.2, 0.7),
    max_cysteines: int = 1,
) -> dict:
    
    """
    Apply physicochemical/developability filters to an AMP candidate.

    Returns True if all filters are passed 

    # Taken from OmegaAMP https://arxiv.org/html/2504.17247

    """


    # Synthesizability criteria:
    # Charge (2-10) # AMPs need net positive charge at pH 7.4: +2 to +9 is the validated AMP activity range (APD3 database)
    # Hydrophobicity (-0.5-0.8)
    # amphipathicity: Hydrophobic moment HM (0.2-0.6): distinct hydrophilic face keeps the peptide solvated post-synthesis and predicts good AMP activity.
    # Not more than 3 consecutive hydrophobic residues 
    # Not more than 1 Cys disulfide scrambling during/after SPPS

    # Taken from OmegaAMP https://arxiv.org/html/2504.17247


    hydrophobic_mom = calculate_hydrophobicmoment([sequence])
    hydrophobicity = calculate_hydrophobicity([sequence])
    charge = calculate_charge([sequence])
    n_cys = sequence.count("C")

    if not charge_range[0] <= charge <= charge_range[1]:
        return False
    
    if not hydrophobicity_range[0] <= hydrophobicity <= hydrophobicity_range[1]:
        return False

    if not hydrophobic_moment_cutoff[0] <= hydrophobic_mom <= hydrophobic_moment_cutoff[1]:
        return False

    if n_cys > max_cysteines:
        return False
    
    return True

def check_sequence_for_hydrophobic_clusters(sequence: str, max_run: int = 5) -> bool:

    # filter out of three hydrophobic residues consecutively
    
    HYDROPHOBIC_AA = set("FILVWMA")

    run = 0

    for aa in sequence:
        if aa in HYDROPHOBIC_AA:
            run += 1

            if run > max_run:
                return False
        else:
            run = 0

    return True
    

class PeptideChecks:
    def __init__(self, config):
        self.config = config
        self.reference_fasta_path = './objectives/reference_data/antibacterial.fasta'
        self.reference_sequences = read_fasta_sequences(self.reference_fasta_path)
        self.STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
           
    def basic_validity_mask(self, 
                            candidates,
                            seen, 
                            canonical_only=True):
    
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
    
    def filter_valid_sequences(self, candidates, config):
        valid_mask = self.basic_validity_mask(candidates, reference_sequences=self.reference_sequences)
        valid_sequences = [sequence for sequence, valid in zip(candidates, valid_mask) if valid]
        return valid_sequences
    
    def synthesis_based_masking(self, candidates):
        
        mask: list[bool] = []
        for seq in candidates:
            valid = (check_sequence_for_hydrophobic_clusters(seq['peptide']) 
                     and check_amp_synthesizability(sequence=seq['peptide'], charge_range=(1, 10), max_cysteines=4))
            mask.append(valid)

        valid_sequences = [sequence for sequence, valid in zip(candidates, mask) if valid]
        return valid_sequences
    
    def stricter_synthesis_based_masking(self, candidates):
        
        mask: list[bool] = []
        for seq in candidates:
            valid = (check_sequence_for_hydrophobic_clusters(seq['peptide'], max_run=4) 
                     and check_amp_synthesizability(sequence=seq['peptide'], charge_range=(2, 10), max_cysteines=0)
                     and 8 <=len(seq['peptide'])<= 25)
            mask.append(valid)

        valid_sequences = [sequence for sequence, valid in zip(candidates, mask) if valid]
        return valid_sequences


