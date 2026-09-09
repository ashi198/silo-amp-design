"""
In this script, we create a dataset from a file of peptide strings to pretrain our model. We create a `SequenceDesign`
from each peptide to obtain a sequence of actions.
"""
import time
import pickle, argparse

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sequence_design import SequenceDesign
from config import SequenceConfig
from typing import List
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="Random seed")
    
    parser.add_argument("--device",
                    type=str,
                    default='cuda:0',
                    help="specify device name: either cuda:gpu_num (cuda:0) or cpu")

    parser.add_argument("--results",
                    type=str,
                    default="./results",
                    help="specify directory for storing results")
    
    args = parser.parse_args()
    return args

def main(args):
    datatypes = ["final_train", "final_validation"]
    config = SequenceConfig(args=args)

    for datatype in datatypes:
        start_time = time.perf_counter()

        sequences_dict : List[dict] = []

        path_to_smiles = f"/home/akhanna/AMP/SILO_amp/training/training_data/{datatype}.fasta"
        destination_path = f"/home/akhanna/AMP/SILO_amp/pretrain/{datatype}_pretrain.pickle"

        print("Converting sequence strings to datasets")
        sequences = []
        current_header, current_seq = None, []

        with open(path_to_smiles) as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header and current_seq:
                        sequences.append((current_header, "".join(current_seq)))
                    current_header = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header and current_seq:
                sequences.append((current_header[1:], "".join(current_seq)))

        print(f"Created {len(sequences)}  from the fasta file")
        instances_dict = dict()

        for s in tqdm(sequences):
            sequence_instance = SequenceDesign.make_instance_sequence(config=config, sequence=s[1])
            instances_dict[sequence_instance.seq_string] = dict(
                identifier= s[0],
                peptide=(''.join(sequence_instance.seq_string)), 
                action_seq=sequence_instance.history,
                seq_list=sequence_instance.seq_list,
                residues = sequence_instance.residues, 
                objective=0.0,
                apex_mean_score = 0.0)

        print(f"Generation took {time.perf_counter() - start_time} seconds.")

        with open(destination_path, "wb") as f:
            pickle.dump(instances_dict, f)


if __name__=='__main__':
    args = parse_args()
    main(args)