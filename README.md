# SILO for AMP: Self-Improvement imitation Learning for Antimicrobial Peptide Design

This repository contains the implementation of SILO (Self-Improvement imitation Learning for protein Optimization) for de novo antimicrobial peptide (AMP) sequence design. SILO trains and/or loads a sequence policy, uses incremental stochastic beam search to generate candidate peptides, evaluates a large candidate library with APEX pathogen MIC prediction model and physicochemical checks, and selects a diverse, novel top-`K` submission. The competition inference entry point is [`inference.sh`](inference.sh).

The method is described in:

> [Self-Improvement Imitation with Biologically Guided Search for Protein Design Under Oracle Budgets](https://arxiv.org/abs/2605.26690)

## What the organizers should run

From the repository root:

```bash
chmod +x inference.sh
./inference.sh
```

The script invokes `inference.py` with a seed of `42`, a CUDA device of `cuda:0`, a 50,000-sequence generation budget, and a final top-100 selection.

Before running the script, the competition checkpoint must be placed at:

```text
./results/FT_3_encoder_with_OMA_only_pickle/42/best_model.pt
```

The checkpoint is not included in this source checkout because of its size. The inference runtime expects a PyTorch checkpoint containing the `model_weights` and `optimizer_state` entries used by the training code.

## End-to-end inference pipeline

`inference.py` performs the following operations:

1. Loads `best_model.pt` and reconstructs the `SequenceTransformer` policy.
2. Enables deterministic beam-search decoding and seeds Python, NumPy, and PyTorch randomness with `42` by default.
3. Generates exactly 50,000 unique candidate peptide sequences with lengths from 8 to 50 residues.
4. Removes invalid, duplicated, known, or otherwise disallowed sequences.
5. Scores the generated library with the APEX pathogen MIC prediction ensemble model and the OmegAMP AMP classifier/scorer.
6. Computes sequence properties including charge, hydrophobicity, hydrophobic moment, cysteine count, and hydrophobic runs.
reject a candidate if an actual local alignment has pident > 80% and covers at least 80%
of that generated candidate.
7. Applies novelty, synthesizability, physicochemical, activity, and diversity criteria.
8. Selects a diverse top-100 set across overall, Gram-positive-selective, Gram-negative-selective, and broad-spectrum candidate pools.
9. Writes the complete library and competition submission artifacts to the output directory.

The implementation is inference-only: it does not update model weights or perform additional training.

## Output files

On a successful run, the output directory contains:

| File | Description |
| --- | --- |
| `generated_50k_peptides_library.fasta` | Intermediate generated peptide library before metric-based selection. |
| `library_50k.fasta` | Validated 50,000-sequence library in FASTA format. |
| `library_50k.csv` | The validated library with computed metrics and metadata. |
| `top_100.fasta` | Selected top-100 peptide sequences for submission. |
| `top_100.csv` | Top-100 sequences with ranks, scores, selection categories, and checks. |
| `manifest.json` | Artifact counts, metadata, and SHA-256 checksums for the four submission files. |
| `diagnostic_error.json` | Written if the declared 50k/top-100 artifact contract cannot be satisfied. |

The runtime fails explicitly if it cannot produce exactly the requested number of library or top-`K` sequences, rather than silently emitting an incomplete submission.

## Environment and installation

The reference environment uses Linux, Python 3.10, an NVIDIA GPU, CUDA 12.8-compatible PyTorch wheels, and Ray for parallel evaluation. The original development configuration was tested on an NVIDIA A40 with CUDA 12.2; the pinned PyTorch build is `2.8.0+cu128`.

Create an environment and install the dependencies:

```bash
conda create -n silo python=3.10 -y
conda activate silo
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes the CUDA-enabled PyTorch build, Ray, Biopython, APEX-related scientific Python packages, and packages used for metric calculation and logging.

## Required models and data

The repository includes the reference FASTA files and supporting data used by the inference pipeline:

```text
objectives/reference_data/antibacterial.fasta
objectives/reference_data/marlys.fasta
objectives/reference_data/training.fasta
training/training_data/final_SILO_training_dataset.fasta
```

The APEX model assets are expected under `apex/`, and the OmegAMP implementation and its model/data assets are organized under `OmegAMP/`. Any model weights or large assets supplied separately by the competition must preserve the paths expected by the code or be copied into the corresponding project directories.

The current `config.py` contains an absolute development-machine checkpoint path retained for the training workflow. The supplied inference entry point overrides the checkpoint from its `--checkpoint` argument, but the repository should still be run from its root so all relative data and model paths resolve correctly. If the repository is moved to another location, update any remaining absolute path in `sequence_evaluator.py` and use a checkpoint path appropriate to the new checkout.

## Command-line interface

The inference entry point can also be invoked directly:

```bash
python inference.py \
  --checkpoint PATH_TO_CHECKPOINT_DIRECTORY \
  --output_dir PATH_TO_OUTPUT_DIRECTORY \
  --seed 42 \
  --device cuda:0 \
  --total-peptide-count 50000 \
  --top-k 100
```

Arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Directory containing `best_model.pt`. |
| `--output_dir` | required | Directory for generated and submission artifacts. |
| `--seed` | `42` | Random seed for reproducible generation. |
| `--device` | `cpu` | PyTorch device, normally `cuda:0` for competition inference. |
| `--total-peptide-count` | `50000` | Number of generated library candidates. |
| `--top-k` | `100` | Number of final selected candidates. Must not exceed the library size. |

For a CPU smoke test or reduced local experiment, lower `--total-peptide-count` and choose a correspondingly smaller `--top-k`. The competition contract, however, is the 50,000-candidate library and 100-candidate submission configured in `inference.sh`.

## Training and pretraining entry points

The main training entry point is [`main.py`](main.py):

```bash
python main.py \
  --seed 42 \
  --epoches 100 \
  --device cuda:0 \
  --results ./results/experiment \
  --comments experiment
```

Training performs repeated self-improvement cycles. Each cycle generates candidates with the current policy, evaluates them, uses selected trajectories for policy training, tracks metrics through MLflow, saves `best_model.pt` and `last_model.pt`, and finally runs inference with the trained policy.

The optional pretraining utilities are in [`pretrain/`](pretrain/). The pretraining dataset files are already present in that directory; [`pretrain/pretrain.py`](pretrain/pretrain.py) contains the pretraining loop. These workflows are not required when the organizers are supplied with the competition checkpoint.

## Repository layout

```text
inference.sh                 Competition inference launcher
inference.py                 Inference-only entry point and artifact writer
main.py                      Training/fine-tuning entry point
config.py                    Model, search, data, and optimization configuration
model/                       Transformer policy and multi-head attention modules
core/                        Sequence generation and incremental beam-search runtime
sequence_design.py           Sequence/action representation used by the policy
sequence_dataset.py          Policy-training dataset utilities
sequence_evaluator.py        APEX/OmegAMP evaluation and candidate-selection policy
evaluation_metrics/          Activity, novelty, physicochemical, and diversity metrics
apex/                        APEX antibacterial activity models and scoring helpers
OmegAMP/                     OmegAMP scoring implementation and supporting assets
pretrain/                    Pretraining data preparation and training utilities
training/training_data/      SILO training and validation FASTA/data files
objectives/reference_data/   Reference sequences used for filtering and novelty checks
requirements.txt             Python dependencies
```

## Reproducibility notes

- Run commands from the repository root.
- Use the same checkpoint, seed, device configuration, and reference data when comparing runs.
- The inference path forces deterministic search, but GPU kernels and third-party numerical libraries can still introduce small platform-dependent differences.
- Ray is initialized by the inference runtime and shut down when the run completes.
- A successful `manifest.json` is the recommended handoff record because it contains artifact counts and SHA-256 checksums.

## Citation and acknowledgements

If you use SILO for AMP, please cite the SILO paper linked above. This repository also builds on or incorporates ideas and implementations from:

- [Gumbeldore](https://github.com/grimmlab/gumbeldore), for incremental stochastic beam search and candidate generation.
- [ProSpero](https://github.com/szczurek-lab/ProSpero), for protein fitness benchmark data and oracle-related components.
- [Stochastic Beam Search](https://github.com/wouterkool/stochastic-beam-search/tree/stochastic-beam-search), for search methodology and reference implementations.
- [OmegAMP](https://openreview.net/forum?id=hAq3XLZ9ex), for AMP generation/classification components used by the scoring pipeline.

## License

No repository-level license file is currently present. Please follow the competition’s submission and redistribution terms, and add the applicable license before public redistribution if required.
