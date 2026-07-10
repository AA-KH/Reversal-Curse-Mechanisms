# Mechanisms of In-Context Circumvention of the Reversal Curse

Official companion code and artifacts for the GPT-2 Medium experiments in the paper.

This repository contains the SynthRel-500 dataset generator, model-profiled GPT-2 supervised fine-tuning code, behavioral evaluation scripts, mechanistic interpretability analyses, statistical tests, figure generation scripts, and saved lightweight outputs needed to inspect and reproduce the reported results.

The scientific pipeline is intentionally conservative. Scripts are organized for clarity and reproducibility, but the experimental protocol, prompts, splits, hyperparameters, seeds, evaluation metrics, activation patching, Direct Logit Attribution, ablations, statistics, and figure logic are preserved.

## What Is Reproduced

The GPT-2 Medium pipeline reproduces:

- SynthRel-500 generation with fixed 70/15/15 per-relation splits.
- Forward-only supervised fine-tuning of GPT-2 Medium.
- Forward and reverse exact-match SFT evaluation.
- Reverse in-context learning evaluations for `k = 0, 1, 2, 4, 8`.
- Base-model `k=4` ICL evaluation and wrong-label controls.
- Bootstrap confidence intervals, McNemar's exact test, Wilcoxon signed-rank test, and paired effect sizes.
- Position-controlled activation patching.
- Direct Logit Attribution for attention heads and MLP sublayers.
- Attention-head, combined-MLP, and single-layer MLP ablations.
- Induction-head diagnostics and probing analyses.
- Dataset-size, checkpoint, and multi-seed robustness analyses.
- Manuscript figures and derived tables.

## Repository Structure

```text
data/
  generate_entities.py          SynthRel-500 generation and split creation
  synthrel_500.json             Generated dataset
  splits/                       Train/validation/test and size-ablation splits

src/
  repro_config.py               Model profiles and artifact roots
  pipeline_manifest.py          Paper-to-pipeline artifact manifest
  run_replication_pipeline.py   End-to-end orchestration entry point
  run_mechanistic_pipeline.py   Mechanistic analysis orchestration
  run_robustness_pipeline.py    Size/checkpoint/multi-seed orchestration
  sft/                          Fine-tuning and SFT evaluation
  icl/                          In-context learning evaluation
  mechanistic/                  DLA, patching, ablations, probing, figures
  analysis/                     Statistics, derived tables, validation, reports

results/
  gpt2_small/
    tables/                     GPT-2 Small scale-replication CSV outputs
    processed/                  GPT-2 Small scale-replication mechanistic outputs
  gpt2_medium/
    tables/                     Paper CSV evaluation/statistical outputs
    processed/                  Paper processed mechanistic outputs
  gpt2_large/
    tables/                     Optional scale-replication CSV outputs
    processed/                  Optional scale-replication mechanistic outputs

paper/
  main.tex                      Manuscript source
  figures/gpt2_small/           GPT-2 Small scale-replication figures
  figures/gpt2_medium/          GPT-2 Medium figures used by the manuscript
  figures/gpt2_large/           Optional scale-replication figures

reports/
  gpt2_small/                   Generated Small manifests/reports
  gpt2_medium/                  Generated Medium manifests/reports
  gpt2_large/                   Optional scale-replication manifests/reports

experiments/                    Local training checkpoints, ignored by Git when present
```

## Installation

Use Python 3.10 or newer. A CUDA GPU is recommended for training and mechanistic sweeps; CPU and Apple Silicon MPS are supported for smaller checks but can be slow.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main dependencies are PyTorch, Transformers, TransformerLens, pandas, scipy, scikit-learn, matplotlib, seaborn, and Weights & Biases in offline mode.

The same checks can also be run through `make` with `PYTHON=python3` by default.

## Quick Validation

These checks do not run expensive model experiments.

```bash
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium --require-outputs
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all --dry-run
```

Generate a manifest and local artifact status report:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage manifest
python3 src/analysis/generate_replication_report.py --model-profile gpt2_medium
```

Outputs:

- `reports/gpt2_medium/pipeline_manifest.json`
- `reports/gpt2_medium/replication_status.md`

## Full Reproduction Workflow

Run commands from the repository root.

### 1. Generate Dataset

```bash
python3 data/generate_entities.py
```

Outputs:

- `data/synthrel_500.json`
- `data/splits/train.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`
- `data/splits/train_35.jsonl`
- `data/splits/train_70.jsonl`
- `data/splits/train_140.jsonl`
- `data/splits/train_350.jsonl`
- `data/CHECKSUMS.sha256` records the released dataset hashes.

### 2. Train GPT-2 Medium

```bash
python3 src/sft/train.py --model-profile gpt2_medium --seed 42
```

The paper hyperparameters are fixed in `src/sft/train.py`.

Output:

- `experiments/gpt2_medium/run_<timestamp>_seed42/final_model/`

### 3. Run Behavioral Evaluations

Replace `<checkpoint>` with the final model directory from training.

```bash
python3 src/sft/evaluate_sft.py \
  --model-profile gpt2_medium \
  --checkpoint-path <checkpoint>

python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --model-path <checkpoint>

python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --model-path <checkpoint> \
  --k 4 \
  --wrong-labels

python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --k 4
```

Primary outputs are written under `results/gpt2_medium/tables/`.

### 4. Run Mechanistic Analyses

```bash
python3 src/run_mechanistic_pipeline.py \
  --model-profile gpt2_medium \
  --checkpoint-path <checkpoint>
```

This runs fixed ICL prompt generation, position-matched prompt generation, DLA, position-controlled activation patching, causal ablations, induction diagnostics, and probing.

Primary outputs are written under:

- `results/gpt2_medium/processed/`
- `results/gpt2_medium/tables/`

### 5. Run Robustness Analyses

```bash
python3 src/run_robustness_pipeline.py \
  --model-profile gpt2_medium \
  --stage all \
  --primary-checkpoint <checkpoint>
```

This covers dataset-size runs, checkpoint evaluations, multi-seed SFT/ICL runs, seed-specific DLA/ablation outputs, and head-overlap stability summaries.

### 6. Regenerate Statistics, Tables, and Figures

```bash
python3 src/analysis/compute_statistics.py --model-profile gpt2_medium
python3 src/analysis/table_generation.py --model-profile gpt2_medium
python3 src/analysis/plot_behavioral.py --model-profile gpt2_medium
python3 src/mechanistic/plot_figures.py --model-profile gpt2_medium
```

Outputs:

- Tables: `results/gpt2_medium/tables/*.csv`
- Processed mechanistic data: `results/gpt2_medium/processed/*.csv`
- Figures: `paper/figures/gpt2_medium/*.pdf` and selected `.png` files

## One-Command Orchestration

After confirming compute availability, the full Medium pipeline can be run with:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all
```

For staged execution, use repeated `--stage` flags:

```bash
python3 src/run_replication_pipeline.py \
  --model-profile gpt2_medium \
  --stage eval-sft \
  --stage eval-icl \
  --checkpoint-path <checkpoint>
```

Available stages are `manifest`, `dataset`, `train`, `eval-sft`, `eval-icl`, `mechanistic`, `robustness`, `stats`, `figures`, `report`, and `all`.

## Expected Primary Results

The saved GPT-2 Medium artifacts report:

- SFT forward: 100.00%.
- SFT reverse: 64.00%.
- Base GPT-2 Medium ICL reverse `k=4`: 99.56%.
- SFT model ICL reverse `k=0`: 64.00%.
- SFT model ICL reverse `k=1`: 30.22%.
- SFT model ICL reverse `k=2`: 83.11%.
- SFT model ICL reverse `k=4`: 98.67%.
- SFT model ICL reverse `k=8`: 100.00%.
- Constant-placeholder wrong-label control `k=4`: 0.00%.

See `results/gpt2_medium/tables/stats_results.csv` for confidence intervals and paired tests.

## Artifact Map

The complete paper-to-pipeline map is maintained in:

- `src/pipeline_manifest.py`
- `docs/REPRODUCTION_AUDIT.md`
- `docs/ARTIFACTS_AND_PROVENANCE.md`
- Generated JSON: `reports/gpt2_medium/pipeline_manifest.json`

Use these files to trace each reported figure, table, statistic, and mechanistic claim back to the script and artifact that produces it.

## GPT-2 Scale Replications

The repository also contains first-class model-profiled GPT-2 Small and GPT-2 Large scale-replication paths. They run the same pipeline components as the official GPT-2 Medium paper pipeline and write to model-specific directories:

- `experiments/gpt2_small/`
- `results/gpt2_small/tables/`
- `results/gpt2_small/processed/`
- `paper/figures/gpt2_small/`
- `reports/gpt2_small/`
- `experiments/gpt2_large/`
- `results/gpt2_large/tables/`
- `results/gpt2_large/processed/`
- `paper/figures/gpt2_large/`
- `reports/gpt2_large/`

Example dry run:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_small --stage all --dry-run
python3 src/run_replication_pipeline.py --model-profile gpt2_large --stage all --dry-run
```

## Public-Release Notes

- `experiments/`, `wandb/`, `logs/`, and `scratch/` are ignored because they contain local checkpoints, offline run state, logs, and legacy development scripts.
- Lightweight CSV outputs and paper figures are retained for auditability.
- No script in the maintained public pipeline depends on `scratch/`.
