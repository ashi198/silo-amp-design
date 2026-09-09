from pathlib import Path
from Bio import Align
from config import SequenceConfig
import torch, os, sys, math, glob, json, tempfile
import pandas as pd 
import numpy as np
from apex import APEX_models
from apex.utils import onehot_encoding, make_vocab
sys.modules["APEX_models"] = APEX_models
from seqme.metrics import Uniqueness, Diversity, ConformityScore
from evaluation_metrics.metrics_utils import novelty_against_reference, mmseqs_marlys_similarity, calculate_clustering_coverage, calculate_physchem_prop, precalculate_embeddings, calculate_hydrophobicmoment, calculate_distributional_embeddings_esm, calculate_charge, calculate_hydrophobicity
from Bio import SeqIO


class APEXEnsemble:
    def __init__(self, config, device):
        # Load the 8 pretrained APEX-pathogen models (relative to this file, not the cwd).
        MODEL_DIR = os.path.join(config.apex_work_dir, "APEX_pathogen_models")
        self.APEX_models = []
        self.device = device
        self.max_len = 52 #maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, idx2word = make_vocab()
        self.batch_size = 1000 
        self.STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

        for a_model in sorted(glob.glob(os.path.join(MODEL_DIR, "APEX_*"))):
            model = torch.load(a_model, map_location=self.device, weights_only=False)
            model.eval()
            self.APEX_models.append(model)

    def calculate_mic_scores(self, sequences):
        if not isinstance(sequences[0], str): 
            seq_list = [seq.seq_string for seq in sequences]
        else:
            seq_list = sequences
        seq_list = np.array(seq_list)

        for ensemble_id in range(len(self.APEX_models)):
            if self.device != 'cpu':
                AMP_model = self.APEX_models[ensemble_id].cuda().eval()
            else:
                AMP_model = self.APEX_models[ensemble_id].cpu().eval()

            data_len = len(seq_list)
            
            for i in range(int(math.ceil(data_len/float(self.batch_size)))):

                seq_batch = seq_list[i*self.batch_size:(i+1)*self.batch_size]
                seq_rep = onehot_encoding(seq_batch, self.max_len, self.word2idx) #make input

                if self.device != 'cpu':
                    X_seq = torch.LongTensor(seq_rep).cuda()
                    AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy() #make predictions
                else:
                    X_seq = torch.LongTensor(seq_rep)
                    AMP_pred_batch = AMP_model(X_seq).detach().numpy() #make predictions

                AMP_pred_batch = 10**(6-AMP_pred_batch) #transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                if i == 0:
                    AMP_pred = AMP_pred_batch
                else:
                    AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

            #sum up the predictions made by different APEX models
            if ensemble_id == 0:
                AMP_sum = AMP_pred
            else:
                AMP_sum += AMP_pred


        AMP_pred = AMP_sum /len(self.APEX_models)

        return AMP_pred
        
class PeptideChecks:
    def __init__(self, config):
        STANDARD_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
        self.config = config
        self.reference_fasta_path = config.antibacterial_fasta
        self.reference_sequences = read_fasta_sequences(self.reference_fasta_path)
        self.marlys_reference = read_fasta_sequences(config.marlys_fasta)
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
                and seq['peptide'] not in self.reference_sequences
                and seq['peptide'] not in self.marlys_reference)
            
            mask.append(valid)

        return mask
    

class BigLibraryMetrics:
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.apex_ensemble = APEXEnsemble(config, device)
        self.peptide_checks = PeptideChecks(config)
    
    def calculate_metrics_big_library(self, config, path_to_generated_peptides, path_to_training_amps, path_to_reference_sequences):

        """
            3 main files are needed: 
                1. Path to generated peptides by SILO
                2. Path to training data (GRAMPA + AMPDiffusion dataset)
                3. Path to reference data (antibacterial.fasta)

        """

        print("------")
        print("Running evaluation metrics on generated peptide library. This may take a while :/")
        print("------")

        generated_peptides = read_fasta_return_sequence_list(path_to_generated_peptides)
        
        assert len(generated_peptides) == config.total_peptide_count

        generated_df = pd.DataFrame(generated_peptides, columns=["id", "sequence"])

        full_data_analyis = {}

        generated_peptides_list = [seq[1] for seq in generated_peptides]

        # 1. Sequence based metrics 
        uniquenss_metric = Uniqueness()
        uniqueness_ratio = uniquenss_metric(generated_peptides_list) #uniqueness amongst generated peptides 
        full_data_analyis["Uniqueness"] = uniqueness_ratio

        diversity_metric = Diversity(k= 10, seed=config.seed)
        diversity = diversity_metric(generated_peptides_list) #Diversity of all generated peptides against a small subset of generated peptides
        full_data_analyis["Diversity"] = diversity


        #3. MIC calculation
        apex_pathogen_scores = self.apex_ensemble.calculate_mic_scores(generated_peptides_list)
        apex_mean_scores = np.mean(apex_pathogen_scores, axis=1)

        apex_df = pd.DataFrame({"id": generated_df["id"].values, "apex_mean_mic": apex_mean_scores, 
                                "A_baumannii": apex_pathogen_scores[:, 0], "E_coli_11775": apex_pathogen_scores[:, 1], 
                                "E_coli_AIC221": apex_pathogen_scores[:, 2], "E_coli_AIC222": apex_pathogen_scores[:, 3],
                                "K_pneumoniae": apex_pathogen_scores[:, 4], "P_aeruginosa_PAO1": apex_pathogen_scores[:, 5],
                                "P_aeruginosa_PA14": apex_pathogen_scores[:, 6], "S_aureus": apex_pathogen_scores[:, 7],
                                "MRSA": apex_pathogen_scores[:, 8], "VRE_faecalis": apex_pathogen_scores[:, 9], "VRE_faecium": apex_pathogen_scores[:, 10],
                                "apex_mic50": self.apex_metrics("apex_mic50", apex_pathogen_scores), "apex_mic90": self.apex_metrics("apex_mic90", apex_pathogen_scores), "apex_gram_positive_mean": self.apex_metrics("apex_gram_positive_mean", apex_pathogen_scores), 
                                "apex_gram_negative_mean": self.apex_metrics("apex_gram_negative_mean", apex_pathogen_scores), "gram_negative_selectivity": self.apex_metrics("gram_negative_selectivity", apex_pathogen_scores),  
                                "gram_positive_selectivity": self.apex_metrics("gram_positive_selectivity", apex_pathogen_scores)
                                })
        

        generated_peptides_list = [peptide for _, peptide in sorted(zip(apex_mean_scores, generated_peptides_list), key=lambda x: x[0])]
        generated_peptides = [peptide for _, peptide in sorted(zip(apex_mean_scores, generated_peptides), key=lambda x: x[0])]
        
    
        # Calculate novelty metrics 
        # -- Use Levenshtein ratio for antibacterial peptide set (< 0.8 cut off) 
        # -- Use Normalized Smith-Waterman local-alignment similarity against known AMP (training dataset) (0.6 cutoff) 

        novelty_scores = []
        print("------")
        print("Novelty score calculation using local similarity and Levenshtein ratio")
        print("------")
        training_ref_amps = read_fasta_return_sequence_list(config.training_repre_seq_fasta)
        antibacterial_ref_amps = read_fasta_return_sequence_list(config.antibacterial_repre_seq_fasta)

        for seq in generated_peptides:
            novelty_results = novelty_against_reference(seq, training_ref_amps, antibacterial_ref_amps)
            novelty_scores.append({
                "id": novelty_results["id"],
                "training_data_novelty": novelty_results["training_data_novelty"], 
                "max_train_reference_similarity": novelty_results["max_train_reference_similarity"], 
                "passes_local_similarity_check": novelty_results["passes_local_similarity_check"], 
                "closest_train_id": novelty_results["closest_train_reference_id"],
            "closest_train_seq": novelty_results["closest_train_reference_sequence"],
            "levenshtein_novelty": novelty_results["ab_data_novelty"],
            "levenshtein_similarity":  novelty_results["max_ab_reference_similarity"],  
            "passes_levenshtein_check": novelty_results["passes_ab_novelty"],             
            "closest_ab_id":novelty_results["closest_ab_reference_id"],
            "closest_ab_seq":novelty_results["closest_ab_reference_sequence"]}
            )
        
        novelty_df = pd.DataFrame(novelty_scores)

        # MMSeq2 similarity of generated sequences against Marlys database 
        print("------")
        print("MMseqs cluster analysis against MarLys database")
        print("------")

        marlys_results = mmseqs_marlys_similarity(config=config, query_fasta=path_to_generated_peptides, marlys_fasta=config.marlys_fasta)
        cluster_results, global_clustering_results = calculate_clustering_coverage(config=config, query_fasta= path_to_generated_peptides, reference_fasta=path_to_training_amps)
        full_data_analyis["clustering"] = global_clustering_results

        mmseqs_df = pd.DataFrame.from_dict(marlys_results, orient="index").reset_index(drop=True)
        clustering_df = pd.DataFrame.from_dict(cluster_results, orient="index").reset_index(drop=True)

        # 2. Property distribution
        physchem_properites = calculate_physchem_prop(generated_peptides_list)
        property_df = pd.DataFrame({"id": generated_df["id"].values, "hydrophobicity": physchem_properites["hydrophobicity"], 
                            "hydrophobic_moment": physchem_properites["hydrophobic_moment"], "charge": physchem_properites["charge"],
                            "isoelectric_point": physchem_properites["isoelectric_point"]
                            })

        generated_df = (generated_df.merge(novelty_df, on="id", how="left", validate="one_to_one").merge(apex_df, on="id", how="left", validate="one_to_one").merge(mmseqs_df, on="id", how="left", validate="one_to_one").merge(clustering_df, on="id", how="left", validate="one_to_one").merge(property_df, on="id", how="left", validate="one_to_one"))
        #generated_df.to_csv(f"{config.results_path}/per_peptide_metrics.csv", index=False)

        return generated_df, full_data_analyis
    
    '''def calculate_metrics_big_library(self, config, path_to_generated_peptides):

        """
            3 main files are needed: 
                1. Path to generated peptides by SILO
                2. Path to training data (GRAMPA + AMPDiffusion dataset)
                3. Path to reference data (antibacterial.fasta)

        """

        print("------")
        print("Running evaluation metrics on generated peptide library. This may take a while :/")
        print("------")

        generated_peptides = read_fasta_return_sequence_list(path_to_generated_peptides)
        
        assert len(generated_peptides) == config.total_peptide_count

        generated_df = pd.DataFrame(generated_peptides, columns=["id", "sequence"])

        full_data_analyis = {}

        generated_peptides_list = [seq[1] for seq in generated_peptides]

        # 1. Sequence based metrics 
        uniquenss_metric = Uniqueness()
        uniqueness_ratio = uniquenss_metric(generated_peptides_list) #uniqueness amongst generated peptides 
        full_data_analyis["Uniqueness"] = uniqueness_ratio

        diversity_metric = Diversity(k= 10, seed=config.seed)
        diversity = diversity_metric(generated_peptides_list) #diversity of all generated peptides against a small subset of generated peptides
        full_data_analyis["Diversity"] = diversity


        #3. MIC calculation
        apex_pathogen_scores = self.apex_ensemble.calculate_mic_scores(generated_peptides_list)
        apex_mean_scores = np.mean(apex_pathogen_scores, axis=1)

        apex_df = pd.DataFrame({"id": generated_df["id"].values, "apex_mean_mic": apex_mean_scores, 
                                "A_baumannii": apex_pathogen_scores[:, 0], "E_coli_11775": apex_pathogen_scores[:, 1], 
                                "E_coli_AIC221": apex_pathogen_scores[:, 2], "E_coli_AIC222": apex_pathogen_scores[:, 3],
                                "K_pneumoniae": apex_pathogen_scores[:, 4], "P_aeruginosa_PAO1": apex_pathogen_scores[:, 5],
                                "P_aeruginosa_PA14": apex_pathogen_scores[:, 6], "S_aureus": apex_pathogen_scores[:, 7],
                                "MRSA": apex_pathogen_scores[:, 8], "VRE_faecalis": apex_pathogen_scores[:, 9], "VRE_faecium": apex_pathogen_scores[:, 10],
                                "apex_mic50": self.apex_metrics("apex_mic50", apex_pathogen_scores), "apex_mic90": self.apex_metrics("apex_mic90", apex_pathogen_scores), "apex_gram_positive_mean": self.apex_metrics("apex_gram_positive_mean", apex_pathogen_scores), 
                                "apex_gram_negative_mean": self.apex_metrics("apex_gram_negative_mean", apex_pathogen_scores), "gram_negative_selectivity": self.apex_metrics("gram_negative_selectivity", apex_pathogen_scores),  
                                "gram_positive_selectivity": self.apex_metrics("gram_positive_selectivity", apex_pathogen_scores)
                                })
        

        generated_peptides_list = [peptide for _, peptide in sorted(zip(apex_mean_scores, generated_peptides_list), key=lambda x: x[0])]
        generated_peptides = [peptide for _, peptide in sorted(zip(apex_mean_scores, generated_peptides), key=lambda x: x[0])]
        
    
        # 2. Property distribution
        physchem_properites = calculate_physchem_prop(generated_peptides_list)
        property_df = pd.DataFrame({"id": generated_df["id"].values, "hydrophobicity": physchem_properites["hydrophobicity"], 
                            "hydrophobic_moment": physchem_properites["hydrophobic_moment"], "charge": physchem_properites["charge"],
                            "isoelectric_point": physchem_properites["isoelectric_point"]
                            })

        generated_df = (generated_df.merge(apex_df, on="id", how="left", validate="one_to_one").merge(property_df, on="id", how="left", validate="one_to_one"))

        return generated_df, full_data_analyis'''
    
    def calculate_distributional_properties(self, config, path_to_top_peptides, path_to_training_amps, path_to_top_100_embeds_pickle, path_to_known_amp_embeds_pickle,
                        full_data_analyis):

        top_peptides = read_fasta_return_sequence_list(path_to_top_peptides)
        
        assert len(top_peptides) == config.top_k_peptides

        training_amps = read_fasta_return_sequence_list(path_to_training_amps)
        generated_peptides_list = [seq[1] for seq in top_peptides]
        training_amp_list = [seq[1] for seq in training_amps]
        
        # 2. Property distribution
        
        charge_conformity = ConformityScore(reference=training_amp_list, predictors=[calculate_charge])
        charge_conformity_results = charge_conformity(generated_peptides_list)
        charge_conformity_score = charge_conformity_results.value

        full_data_analyis["conformity_score_charge"] = charge_conformity_score

        # Synthesizability (charge: 2-10, length: 8–50, hydrophobicity: −0.5 to 0.8, amphipathicity (0.2-0.6))
        amphiphilicity_conformity = ConformityScore(reference=training_amp_list, predictors=[calculate_hydrophobicmoment])
        amphiphilicities_results = amphiphilicity_conformity(generated_peptides_list)
        amphiphilicities_conformity_score = amphiphilicities_results.value

        full_data_analyis["amphiphilicities_conformity_score"] = amphiphilicities_conformity_score

        # 3. Distributional similarity in embedding space 
        fbd_amp_result, mmd_amp_result, precision_amp_result, recall_amp_result = calculate_distributional_embeddings_esm(config,
                                                                                                path_to_top_100_embeds_pickle, path_to_known_amp_embeds_pickle,
                                                                                                generated_peptides_list, training_amp_list)
        full_data_analyis["FBD"] = fbd_amp_result
        full_data_analyis["MMD"] = mmd_amp_result
        full_data_analyis["precision"] = precision_amp_result
        full_data_analyis["recall"] = recall_amp_result

        return full_data_analyis
    
    def calculate_embedding_properties(self, config, top_100_df, full_data_analyis):

        path_pickle_files = precalculate_embeddings(config, f"{config.results_path}/top_100_peptides.fasta", config.training_fasta)
        path_pickle_files = ['./results/inference_pretrained/42/top_100_embeds.pkl', './results/inference_pretrained/42/known_amps_embeds.pkl']    
        log_metrics = self.calculate_distributional_properties(config, path_to_top_peptides=f"{config.results_path}/top_100_peptides.fasta", path_to_training_amps=config.training_fasta,
                                                                        path_to_top_100_embeds_pickle=path_pickle_files[0], path_to_known_amp_embeds_pickle=path_pickle_files[1],
                                                                        full_data_analyis=full_data_analyis)
        
        log_metrics = log_full_results(config, log_metrics)

    
    def apex_metrics(self, metrics, scores):
        
        gram_neg_scores = scores[:, 0:6]
        gram_pos_scores = scores[:, 7:11]

        if metrics == 'apex_mic50':
            return np.median(scores, axis=1)
        if metrics == 'apex_mic90':
            return np.quantile(scores,0.90,axis=1,method="higher")
        if metrics == 'apex_gram_positive_mean':
            return np.mean(gram_pos_scores, axis=1)
        if metrics == 'apex_gram_negative_mean':
            return np.mean(gram_neg_scores, axis=1)
        
        if metrics == 'gram_positive_selectivity' or metrics == 'gram_negative_selectivity':

            gram_neg_median = np.median(gram_neg_scores, axis=1)
            gram_pos_median = np.median(gram_pos_scores, axis=1)

            if metrics == 'gram_positive_selectivity':
                return (gram_pos_median / gram_neg_median)
            else:
                return (gram_neg_median / gram_pos_median)
            


def dataframe_to_temp_fasta(df):
    """
    Create a temporary FASTA file from a DataFrame
    containing 'id' and 'sequence' columns.

    Returns the path to the temporary FASTA file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".fasta",
        delete=False
    )

    for _, row in df.iterrows():
        tmp.write(f">{row['id']}\n{row['sequence']}\n")

    tmp.close()

    return tmp.name


def log_full_results(config, log_metrics):
    os.makedirs(config.results_path, exist_ok=True)
    file_log_path = os.path.join(
        config.results_path,
        "global_metrics.txt"
    )
    with open(file_log_path, "a+") as f:
        f.write(json.dumps(log_metrics))
        f.write("\n")


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


def read_fasta_return_sequence_list(path_to_fasta_file):
    all_sequences = []
    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))
    return all_sequences

def save_fasta(sequences, path):
    with open(path, "w") as f:
        for identifier, sequence in sequences:
            f.write(f">{identifier}\n{sequence}\n")


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
    
    if not check_sequence_for_hydrophobic_clusters(sequence):
        return False

    if n_cys > max_cysteines:
        return False
    
    return True

def check_sequence_for_hydrophobic_clusters(sequence: str, max_run: int = 4) -> bool:

    # filter out of three hydrophobic residues consecutively
    #Not more than 3 consequetive hydrobic AA
    
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
