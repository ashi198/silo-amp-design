"""Run the existing SILO policy as deterministic, inference-only generation.

This is deliberately a root-level entry point: it loads the project's real
PyTorch checkpoint, builds the existing ``SequenceTransformer``/APEX/Ray
runtime, and then uses the generated package's selection and artifact code.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from .config import SequenceConfig
from .sequence_evaluator import SequenceEvaluator, SelectionPolicy, select_candidates
from .utils import inference, set_seed, write_submission_artifacts, candidate_records_from_metrics
from .evaluation_metrics.utils import read_fasta_return_sequence_list, BigLibraryMetrics
from .model.transformer_architecture import SequenceTransformer
import pandas as pd

import os
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray, torch, os, argparse, copy


#PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SILO for AMP design reproducible inference")
    parser.add_argument("--checkpoint", type=Path, default='./results/FT_3_1_GP_with_mdr/42/')
    parser.add_argument("--output_dir", type=Path, default='./results/FT_3_1_GP_with_mdr_test_model')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-peptide-count", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=100)
    return parser


def run_inference(args: argparse.Namespace) -> dict[str, Any]:

    """Generate, select, and write candiates using the SILO."""

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    if not checkpoint_path:
        raise ValueError(f"checkpoint not found: {checkpoint_path}")
    if args.total_peptide_count < 1 or args.top_k < 1:
        raise ValueError("total-peptide-count and top-k must be positive")
    if args.top_k > args.total_peptide_count:
        raise ValueError("top-k cannot exceed total-peptide-count")

    config = SequenceConfig(args)
    config.results_path = str(output_dir)
    config.total_peptide_count = args.total_peptide_count
    config.top_k_peptides = args.top_k
    config.training_device = args.device
    config.do_inference = True
    config.self_improvement_learning["devices_for_workers"] = [args.device]
    config.self_improvement_learning["beam_width"] = 32
    os.makedirs(output_dir, exist_ok=True)

    evaluator = SequenceEvaluator(config, torch.device(args.device))
    big_library_worker = BigLibraryMetrics(config, config.training_device, evaluator)
    network = SequenceTransformer(config, config.training_device)

    set_seed(args.seed)
    checkpoint = torch.load(os.path.join(checkpoint_path, "best_model.pt"), weights_only=False)

    network.load_state_dict(checkpoint["model_weights"])
    print(f"Loading checkpoint from path {checkpoint_path} for inference")

    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer["lr"], weight_decay=config.optimizer["weight_decay"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state"])) 

    started_ray = False

    try:
        ray.shutdown()
        if not ray.is_initialized():
            ray.init(runtime_env={
            "excludes": [
                ".git/",
                "/SILO_amp/OmegAMP/data/",
                "/SILO_amp/data/",
                "/SILO_amp/apex/APEX_pathogen_models/",
            ],
        })
            
            started_ray = True
        print(f"Policy network is on device {config.training_device}")
        network.to(network.device)
        network.eval()

        network_weights = copy.deepcopy(network.get_weights())

        print("---Running SILO under reproducible sampling conditions to generate 50K library and top 100 candidate list ---")

        generated_50k_fasta_path = inference(
            epoch="submission",
            config=config,
            network_weights=network_weights,
            evalutor=evaluator)
        
        #generated_50k_fasta_path = './results/test_better_model/generated_50k_peptides_library.fasta'
        
        #generated_50k_df = pd.read_csv('/home/akhanna/AMP/SILO_for_ampdesign/results/with_double_aa/42/generated_50k.csv')

        generated_50k_df = big_library_worker.calculate_metrics_big_library(config, generated_50k_fasta_path)
        records = candidate_records_from_metrics(generated_50k_df.to_dict("records"))
        references = read_fasta_return_sequence_list(config.antibacterial_fasta)
        marlys_set = [sequence for _, sequence in read_fasta_return_sequence_list(config.marlys_fasta)]
        training_set = read_fasta_return_sequence_list(config.training_fasta)
        selection = select_candidates(records, references=references, training_amps=training_set, marlys_references=marlys_set, policy=SelectionPolicy(), top_k=args.top_k)
        return write_submission_artifacts(
            output_dir,
            selection.valid_50k,
            selection.selected,
            metadata={
                "mode": "root-legacy-runtime",
                "seed": args.seed,
                "selection_rejection_counts": selection.rejection_counts,
            },
            expected_50k=args.total_peptide_count,
            expected_top_k=args.top_k,
            config=config
        )
    finally:
        if started_ray:
            ray.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.epoches = 1
    args.comments=None
    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
