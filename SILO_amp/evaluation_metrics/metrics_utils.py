import tempfile, subprocess, os, Levenshtein
from Bio import Align, SeqIO
import pandas as pd 
import modlamp.analysis as manalysis
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import pickle
import numpy as np


def local_similarity(a: str, b: str) -> float:
    _aligner = Align.PairwiseAligner()
    _aligner.mode = "local"
    _aligner.match_score = 1.0
    _aligner.mismatch_score = -1.0
    _aligner.open_gap_score = -1.0
    _aligner.extend_gap_score = -1.0

    """Normalized Smith-Waterman local-alignment similarity between two peptides, in [0, 1]."""
    score = _aligner.score(a, b)
    return max(0.0, score) / max(len(a), len(b))
    

def calculate_hydrophobicity(data: List[str]) -> np.ndarray:
    """
        Taken from https://github.com/szczurek-lab/hydramp/blob/master/amp/utils/phys_chem_propterties.py 
    """
    h = manalysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return h.H[0]


def calculate_hydrophobicmoment(data: List[str]) -> np.ndarray:
    h = manalysis.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return h.descriptor.flatten()


def calculate_charge(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge[0]


def calculate_isoelectricpoint(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return h.descriptor.flatten()


def calculate_length(sequences: List[str]) -> np.ndarray:
    return np.array([len(seq) for seq in sequences])


def calculate_physchem_prop(sequences: List[str]) -> Dict[str, object]:
    return {
        "length": calculate_length(sequences).tolist(),
        "hydrophobicity": calculate_hydrophobicity(sequences).tolist(),
        "hydrophobic_moment": calculate_hydrophobicmoment(sequences).tolist(),
        "charge": calculate_charge(sequences).tolist(),
        "isoelectric_point": calculate_isoelectricpoint(sequences).tolist(),
    }


def calculate_clustering_coverage(config, query_fasta, reference_fasta):


    generated_peptides_list = read_fasta_return_sequence_list(query_fasta)
    generated_peptides_dict = dict(generated_peptides_list)
    generated_sequences = [seq[1] for seq in generated_peptides_list]

    mmseq_dir = os.path.join(config.results_path, "mmseq", "cluster")
    os.makedirs(mmseq_dir, exist_ok=True)

    cluster_prefix = os.path.join(mmseq_dir,"reference_clusters")
    cluster_tmp = os.path.join(mmseq_dir,"cluster_tmp")
    search_tmp = os.path.join(mmseq_dir,"search_tmp")
    search_output = os.path.join(mmseq_dir,"generated_to_clusters.tsv")
        
    cluster_command = ["mmseqs","easy-cluster", reference_fasta, cluster_prefix, cluster_tmp, "--min-seq-id", "0.9", "-c", "0.8",
        "--cov-mode", "0", "--threads", "16"]
        
    subprocess.run(cluster_command,check=True)

    # MMSeq clustering 
    cluster_tsv = (cluster_prefix + "_cluster.tsv")
    representatives_fasta = (cluster_prefix + "_rep_seq.fasta")
    clusters = pd.read_csv(cluster_tsv,sep="\t",names=["cluster_rep","member_id",])
    reference_clusters = set(clusters["cluster_rep"])

    # Search command 
    search_command = ["mmseqs","easy-search", query_fasta, representatives_fasta,search_output,search_tmp,
        "-k", "5", "--mask", "0", "--comp-bias-corr", "0", "--min-length", "8", "-s", "2", "-e", "inf",
        "--format-output","query,target,pident,alnlen,qlen,tlen,bits,evalue", "--threads", "16"]
    
    subprocess.run(search_command,check=True)
    columns = ["query", "target", "pident","alnlen","qlen","tlen","evalue","bits"]
    hits = pd.read_csv(search_output,sep="\t",names=columns)

    # No generated sequences matched a cluster
    if hits.empty:
        return {
            "n_reference_clusters": len(reference_clusters),
            "n_covered_clusters": 0,
            "cluster_coverage": 0.0,
            "cluster_counts": {},
            "sequence_to_cluster": {},
        }
    
    # Pick best cluster for each generated peptide
    best_hits = (hits.sort_values(["query","pident","alnlen","bits",],
    ascending=[True,False,False,False,],).drop_duplicates("query"))

    # number of covered reference clusters 

    covered_clusters = set(best_hits["target"])
    cluster_coverage = (len(covered_clusters) / len(reference_clusters) if reference_clusters else 0.0)
    cluster_counts = (best_hits["target"].value_counts().to_dict())

    # Map generated sample -> reference cluster
    sequence_to_cluster = {}

    for _, row in best_hits.iterrows():
        query_id = row["query"]
        seq = generated_peptides_dict[query_id]

        sequence_to_cluster[query_id] = {
            "id": query_id,
            "cluster": row["target"],
            "cluster_identity": float(row["pident"]),
            "alignment_length": int(row["alnlen"]),
            "cluster_bitscore": float(row["bits"]),
        }

    global_results = {
            "n_reference_clusters": len(reference_clusters),
            "n_covered_clusters": len(covered_clusters),
            "cluster_coverage": cluster_coverage,
            "cluster_counts": cluster_counts,
            "sequence_to_cluster": sequence_to_cluster,
        }
    
    return sequence_to_cluster, global_results
    

def read_fasta_return_sequence_list(path_to_fasta_file):
     
    all_sequences = []

    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))

    return all_sequences


def mmseqs_marlys_similarity(config, query_fasta, marlys_fasta: str, identity_threshold: float = 80.0) -> dict:
    
    """
    Read MMseqs2 search results and return the best MarLys match
    for each generated peptide.
    """
    generated_peptides_list = read_fasta_return_sequence_list(query_fasta)
    generated_peptides_dict = dict(generated_peptides_list)


    mmseq_dir = os.path.join(config.results_path, "mmseq")
    os.makedirs(mmseq_dir, exist_ok=True)
    output_tsv = os.path.join(mmseq_dir, "marlys_hits.tsv")
    mmseqs_tmp = os.path.join(mmseq_dir, "mmseqs_tmp")

    # run mmseq2 command 
    # Parameters for running for peptides from here https://www.biorxiv.org/content/10.64898/2026.09.01.747572v1.full.pdf

    command = ["mmseqs", "easy-search", query_fasta, marlys_fasta, output_tsv,  mmseqs_tmp, "--comp-bias-corr", "0",
        "--prefilter-mode", "2", "-e", "1000", "--alignment-mode", "3", "-c", "0.8", "--cov-mode", "2", "--format-output", "query,target,pident,alnlen,qlen,tlen,evalue,bits", 
        "--threads", "16"]
    
    subprocess.run(command,check=True)

    # read outputs 
    columns = ["query", "target", "pident","alnlen","qlen","tlen","evalue","bits"]
    hits = pd.read_csv(output_tsv,sep="\t",names=columns)
    
    # load marlys sequences
    marlys_sequences = {record.id: str(record.seq)for record in SeqIO.parse(marlys_fasta, "fasta")}

    # choose best hit 
    best_hits = (hits.sort_values(["query", "pident", "alnlen", "bits"],ascending=[True, False, False, False],).drop_duplicates("query"))

    results = {}

    for _, row in best_hits.iterrows():
        query_id = str(row["query"])
        seq = generated_peptides_dict[query_id]
        identity = float(row["pident"])
        target_id = row["target"]

        results[query_id] = {
            "id": query_id,
            "max_marlys_identity": identity,
            "closest_marlys_id": target_id,
            "closest_marlys_sequence": marlys_sequences.get(target_id),
            "marlys_bitscore": float(row["bits"]),
            "marlys_evalue": float(row["evalue"]),
            "passes_marlys_80": identity <= identity_threshold,
        }

    return results

def novelty_against_reference(seq, training_amps, antibacterial_reference_amps):

    """
    references should look like:
    [
        {
            "id": "DRAMP00001",
            "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
        },
        ...
    ]
    """
    train_best_similarity = -1.0
    train_best_reference = None
    
    anti_best_similarity = -1.0
    anti_best_reference = None

    for ref in training_amps: 
        similarity = local_similarity(seq["sequence"], ref[1])
        if similarity > train_best_similarity:
            train_best_similarity = similarity
            train_best_reference = ref

    for anti_ref in antibacterial_reference_amps:
        lv_similarity = Levenshtein.ratio(seq["sequence"], anti_ref[1])
        if lv_similarity > anti_best_similarity:
            anti_best_similarity = lv_similarity
            anti_best_reference = anti_ref

    return {
        # for training data matching
        "training_data_novelty": 1.0 - train_best_similarity,
        "max_train_reference_similarity": train_best_similarity,
        "passes_local_similarity_check": train_best_similarity < 0.60,
        "closest_train_reference_id": train_best_reference[0],
        "closest_train_reference_sequence": train_best_reference[1],

        # for antibacterial.fasta matching 
        "ab_data_novelty": 1.0 - anti_best_similarity,
        "max_ab_reference_similarity": anti_best_similarity,
        "passes_ab_novelty": anti_best_similarity < 0.80,
        "closest_ab_reference_id": anti_best_reference[0],
        "closest_ab_reference_sequence": anti_best_reference[1],

    }


def precalculate_embeddings(config, path_to_top_100, path_to_reference_amps):

    top_100_embeds_dict = {}
    known_amp_embeds_dict = {}

    
    esmc = ESM3(config)
    top_100_sequences = read_fasta_return_sequence_list(path_to_top_100)
    known_amps = read_fasta_return_sequence_list(path_to_reference_amps) 

    top_100_seq_list = [seq[1] for seq in top_100_sequences]
    known_amps_list = [seq[1] for seq in known_amps]

    top_peptide_embeds = esmc.embed(top_100_seq_list)
    training_amp_embeds = esmc.embed(known_amps_list)

    # Map identifers with sequence embeddings
    for seq, seq_embed in zip(top_100_sequences, top_peptide_embeds):
        top_100_embeds_dict[seq[1]] = seq_embed
    
    for amp, amp_embed in zip(known_amps, training_amp_embeds):
        known_amp_embeds_dict[amp[1]] = amp_embed


    with open(f"{config.results_path}/top_100_embeds.pkl", "wb") as f:
        pickle.dump(top_100_embeds_dict, f)

    with open(f"{config.results_path}/known_amps_embeds.pkl", "wb") as f:
        pickle.dump(known_amp_embeds_dict, f)

    return [f"{config.results_path}/top_100_embeds.pkl", f"{config.results_path}/known_amps_embeds.pkl"]


def make_cached_embedder(precomputed_embeddings):
    def cached_embedder(sequences):

        seq_list = [seq if isinstance(seq, str) else seq.seq_string for seq in sequences]

        missing = [seq for seq in seq_list if seq not in precomputed_embeddings]

        if missing:
            raise KeyError(
                f"{len(missing)} sequences are missing from the"
            )

        return np.stack([
            precomputed_embeddings[seq]
            for seq in seq_list
        ])

    return cached_embedder


def read_fasta_return_sequence_list(path_to_fasta_file):
     
    all_sequences = []
    for seq_record in SeqIO.parse(path_to_fasta_file, "fasta"):
        all_sequences.append((seq_record.id, str(seq_record.seq)))
    return all_sequences

def save_fasta(sequences, path):
    with open(path, "w") as f:
        for identifier, sequence in sequences:
            f.write(f">{identifier}\n{sequence}\n")
