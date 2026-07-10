# The Reversal Curse Across GPT-2 Scales: Behavioral Replication and Mechanistic Evidence

> **Aaradhya Khanna** · Chitkara University · `aaradh.kh@gmail.com`

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org/)

---

## What This Paper Studies

Large language models trained on the factual association **"A is B"** frequently fail to answer the symmetric query **"B is A"** — a phenomenon termed the **Reversal Curse**. This paper investigates two questions:

1. **Why does Supervised Fine-Tuning (SFT) fail to encode the reverse direction?**
2. **Why does In-Context Learning (ICL) succeed where SFT fails?**

We replicate the Reversal Curse across **three GPT-2 model scales** (Small 117M, Medium 345M, Large 774M) using **SynthRel-500**, a synthetic dataset of 500 abstract relational pairs built entirely from procedurally generated non-words — eliminating all pre-training confounds. We then conduct a detailed mechanistic investigation using activation patching, Direct Logit Attribution (DLA), and structural ablation studies.

### Core Findings

| Finding | Result |
|---|---|
| **Reversal Curse replicates at all scales** | SFT: 100% forward, only 9.33% / 64% / 40% reverse (Small / Medium / Large) |
| **ICL rescues reverse accuracy** | k=4 ICL: 95.56% / 98.67% / 100.00% across scales |
| **Scale-dependent k=1 Interference Effect** | k=1 *hurts* Small/Medium below zero-shot; k=1 *helps* Large (+35 pp) |
| **Causal bottleneck localized** | Layers 16–18 in Medium (+3.97 logit improvement); upper third at all scales |
| **ICL routing redundancy increases with scale** | Top-10 head ablation: −46 pp Small, −20 pp Medium, **0 pp Large** |
| **MLP ensemble is universally required** | Combined MLP ablation collapses ICL to 0% in Medium and Large |

---

## Repository Structure

```
.
├── README.md                         This file
├── LICENSE                           MIT (code) + CC BY 4.0 (dataset)
├── requirements.txt                  Pip-installable direct dependencies
├── environment.yml                   Conda environment (alternative)
├── Makefile                          Convenience make targets
│
├── scripts/                          Shell scripts for common workflows
│   ├── reproduce_gpt2_medium.sh      Full Medium reproduction (one command)
│   ├── reproduce_all_scales.sh       All three scales sequentially
│   ├── validate_artifacts.sh         Validate artifacts (no GPU needed)
│   └── regenerate_figures_and_tables.sh  Regenerate from saved CSVs
│
├── data/                             Dataset and splits
│   ├── generate_entities.py          SynthRel-500 generator
│   ├── synthrel_500.json             Generated dataset (500 pairs)
│   ├── CHECKSUMS.sha256              Dataset integrity hashes
│   └── splits/                       Fixed train/val/test + size-ablation splits
│
├── src/                              All source code
│   ├── repro_config.py               Model profiles and artifact path roots
│   ├── pipeline_manifest.py          Machine-readable paper-to-pipeline map
│   ├── run_replication_pipeline.py   End-to-end orchestration entry point
│   ├── run_mechanistic_pipeline.py   Mechanistic analysis orchestration
│   ├── run_robustness_pipeline.py    Robustness/stability orchestration
│   ├── sft/                          Fine-tuning and SFT evaluation
│   ├── icl/                          k-shot ICL evaluation
│   ├── mechanistic/                  Patching, DLA, ablations, probing, figures
│   └── analysis/                     Statistics, tables, validation, reports
│
├── experiments/                      Training checkpoints (git-ignored; stubs tracked)
│   ├── gpt2_small/
│   ├── gpt2_medium/
│   └── gpt2_large/
│
├── results/                          Saved lightweight CSV outputs
│   ├── gpt2_small/tables/            Scale-replication evaluation CSVs
│   ├── gpt2_small/processed/         Mechanistic CSVs and JSONs
│   ├── gpt2_medium/tables/           Primary paper evaluation CSVs
│   ├── gpt2_medium/processed/        Primary mechanistic outputs
│   └── gpt2_large/{tables,processed}/
│
├── paper/                            Manuscript and figures
│   ├── main.tex                      Full LaTeX manuscript source
│   └── figures/{gpt2_small,gpt2_medium,gpt2_large}/  PDF figures
│
├── reports/                          Generated audit artifacts
│   └── {gpt2_small,gpt2_medium,gpt2_large}/
│       ├── pipeline_manifest.json    Machine-readable artifact map
│       └── replication_status.md     Artifact health check report
│
└── docs/
    ├── REPRODUCTION_AUDIT.md         Claim → script → artifact map for reviewers
    └── ARTIFACTS_AND_PROVENANCE.md   Provenance guide and determinism notes
```

---

## Hardware Requirements

| Setup | Supported | Notes |
|---|---|---|
| **Apple Silicon (M1/M2/M3/M4)** | ✅ Recommended | All experiments in the paper ran on MacBook Air M4, 16 GB RAM, MPS acceleration |
| **NVIDIA GPU (CUDA)** | ✅ Supported | Any GPU with ≥8 GB VRAM works; 16 GB for Large |
| **CPU only** | ✅ Supported | Works but training is very slow (~10× slower than MPS) |
| **Cloud (Colab/Kaggle)** | ✅ Supported | Use `--device cuda`; free T4 sufficient for Small and Medium |

**Memory requirements:**
- GPT-2 Small (117M): 4 GB RAM minimum
- GPT-2 Medium (345M): 8 GB RAM minimum
- GPT-2 Large (774M): 12 GB RAM minimum
- Mechanistic analysis (TransformerLens hooks): add ~2–4 GB on top of model size

---

## Runtime Estimates

All benchmarks measured on **MacBook Air M4 (16 GB, MPS)**. CUDA times are approximate.

### Per-Stage Estimates (GPT-2 Medium)

| Stage | MPS (M-series) | CUDA (RTX 3080) | CPU |
|---|---|---|---|
| Dataset generation | < 1 min | < 1 min | < 1 min |
| SFT training (5 epochs) | ~45 min | ~15 min | ~5 hours |
| Behavioral evaluation | ~15 min | ~5 min | ~1 hour |
| Mechanistic analysis | ~60 min | ~20 min | ~4 hours |
| Robustness (size/seed/ckpt) | ~90 min | ~30 min | ~6 hours |
| Stats + tables + figures | ~3 min | ~3 min | ~3 min |
| **Total (Medium)** | **~3.5 hours** | **~1.5 hours** | **~16 hours** |

### All Three Scales (Sequential)

| Model | MPS | CUDA |
|---|---|---|
| GPT-2 Small (117M) | ~2 hours | ~45 min |
| GPT-2 Medium (345M) | ~3.5 hours | ~1.5 hours |
| GPT-2 Large (774M) | ~5.5 hours | ~2.5 hours |
| **Total** | **~11 hours** | **~5 hours** |

> **No cloud compute is required.** All experiments were run locally on Apple Silicon.

---

## Installation

**Requirements:** Python 3.10+, pip or conda.

### Option A — pip + venv (recommended)

```bash
git clone <repo-url>
cd Paper1

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Option B — conda

```bash
git clone <repo-url>
cd Paper1

conda env create -f environment.yml
conda activate reversal-curse
```

### Verify installation

```bash
python3 -c "import torch, transformer_lens, transformers; print('OK')"
```

---

## Reproducing Results

### Option 1: One-command reproduction (recommended)

```bash
# GPT-2 Medium (primary paper results) — ~3.5 hours on M-series
bash scripts/reproduce_gpt2_medium.sh --device mps

# Or for CUDA:
bash scripts/reproduce_gpt2_medium.sh --device cuda

# All three scales — ~11 hours on M-series
bash scripts/reproduce_all_scales.sh --device mps
```

### Option 2: Quick validation without retraining

If you only want to verify the saved artifacts are present and pipelines are wired correctly (~30 seconds, no GPU):

```bash
bash scripts/validate_artifacts.sh
```

### Option 3: Regenerate figures and tables from saved CSVs

All figures and tables can be regenerated from the committed CSV outputs without any model loading (~2–3 minutes):

```bash
bash scripts/regenerate_figures_and_tables.sh
```

---

## Step-by-Step Manual Reproduction

Run all commands from the repository root.

### Step 1 — Generate the dataset

```bash
python3 data/generate_entities.py
```

Outputs (deterministic, seed 42):
```
data/synthrel_500.json
data/splits/train.jsonl       (350 examples)
data/splits/val.jsonl         (75 examples)
data/splits/test.jsonl        (75 examples)
data/splits/train_{35,70,140,350}.jsonl
```

### Step 2 — Fine-tune

```bash
# Primary paper model
python3 src/sft/train.py --model-profile gpt2_medium --seed 42 --device mps

# Other scales
python3 src/sft/train.py --model-profile gpt2_small --seed 42 --device mps
python3 src/sft/train.py --model-profile gpt2_large --seed 42 --device mps
```

Output: `experiments/gpt2_medium/run_<timestamp>_seed42/final_model/`

### Step 3 — Behavioral evaluation

Replace `<ckpt>` with the `final_model/` path from Step 2.

```bash
# SFT forward/reverse exact-match
python3 src/sft/evaluate_sft.py --model-profile gpt2_medium --checkpoint-path <ckpt>

# ICL k-shot sweep (k=0,1,2,4,8)
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --model-path <ckpt>

# Wrong-label control (k=4, labels → ZXQFVB)
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --model-path <ckpt> --k 4 --wrong-labels

# Base model ICL (pre-SFT, k=4)
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --k 4
```

### Step 4 — Mechanistic analysis

```bash
python3 src/run_mechanistic_pipeline.py \
  --model-profile gpt2_medium \
  --checkpoint-path <ckpt>
```

Runs in sequence: ICL prompt generation → position-matched prompts → activation patching → DLA → ablations → induction scores → probing.

### Step 5 — Robustness analysis

```bash
python3 src/run_robustness_pipeline.py \
  --model-profile gpt2_medium \
  --stage all \
  --primary-checkpoint <ckpt>
```

Covers: dataset-size ablations (N=35/70/140/350), checkpoint evaluations, multi-seed replicates (seeds 101/202/303).

### Step 6 — Statistics, tables, and figures

```bash
python3 src/analysis/compute_statistics.py --model-profile gpt2_medium
python3 src/analysis/table_generation.py   --model-profile gpt2_medium
python3 src/analysis/plot_behavioral.py    --model-profile gpt2_medium
python3 src/mechanistic/plot_figures.py    --model-profile gpt2_medium
```

Outputs:
- `results/gpt2_medium/tables/*.csv`
- `paper/figures/gpt2_medium/*.pdf`

### One-command orchestration

```bash
# Full pipeline (all stages)
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all

# Dry-run preview (no model loading, prints full command sequence)
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all --dry-run

# Staged execution (e.g., skip retraining if checkpoint exists)
python3 src/run_replication_pipeline.py \
  --model-profile gpt2_medium \
  --stage eval-sft --stage eval-icl --stage mechanistic --stage stats --stage figures \
  --checkpoint-path <ckpt>
```

Available stages: `manifest` `dataset` `train` `eval-sft` `eval-icl` `mechanistic` `robustness` `stats` `figures` `report` `all`

### Makefile shortcuts

```bash
make validate          # Validate script paths
make validate-outputs  # Validate saved artifact presence
make dry-run           # Preview full command sequence
make figures           # Regenerate figures
make stats             # Recompute statistics
make tables            # Regenerate tables
make report            # Regenerate replication status report
```

---

## Expected Results

Exact-match accuracy on the 75-example test set (10,000-resample bootstrap CIs, seed 42):

| Condition | Small | Medium | Large |
|---|---|---|---|
| SFT Forward | 100.00% | 100.00% | 100.00% |
| SFT Reverse | 9.33% | 64.00% | 40.00% |
| ICL Reverse k=1 | 0.89% ↓ | 30.22% ↓ | **75.11% ↑** |
| ICL Reverse k=4 | 95.56% | 98.67% | 100.00% |
| ICL k=4 wrong-label | 0.00% | 0.00% | 0.00% |

Full statistical outputs (p-values, CIs, effect sizes) in `results/gpt2_medium/tables/stats_results.csv`.

---

## Dataset: SynthRel-500

500 abstract factual pairs across five relation types. All entity names are procedurally generated pronounceable non-words (e.g., *Defap*, *Gasoz*) with zero overlap with GPT-2's vocabulary.

| Relation | Example fact |
|---|---|
| `IS_CAPITAL_OF` | Defap is the capital of Gasoz |
| `IS_CREATOR_OF` | Wupen is the creator of Lofar |
| `IS_DIRECTOR_OF` | Bivak is the director of Torep |
| `IS_BIRTHPLACE_OF` | Rinok is the birthplace of Semul |
| `IS_PREDECESSOR_OF` | Cakup preceded Dolar |

- 100 pairs per relation type
- Fixed 70/15/15 splits (350 train / 75 val / 75 test)
- Checksums: `data/CHECKSUMS.sha256`

---

## Models

| Model | Parameters | Layers | Heads | d_model |
|---|---|---|---|---|
| GPT-2 Small | 117M | 12 | 12 | 768 |
| GPT-2 Medium | 345M | 24 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 20 | 1280 |

All accessed via [TransformerLens](https://github.com/neelnanda-io/TransformerLens). SFT via [HuggingFace Transformers](https://github.com/huggingface/transformers) `Trainer`.

---

## Audit and Provenance

Every reported number traces back to a specific script and output file:

| Document | Purpose |
|---|---|
| [`docs/REPRODUCTION_AUDIT.md`](docs/REPRODUCTION_AUDIT.md) | Claim → script → artifact map, organized by paper section |
| [`docs/ARTIFACTS_AND_PROVENANCE.md`](docs/ARTIFACTS_AND_PROVENANCE.md) | Provenance guide, determinism notes, reviewer validation commands |
| [`src/pipeline_manifest.py`](src/pipeline_manifest.py) | Machine-readable manifest generator |
| `reports/*/pipeline_manifest.json` | Generated JSON manifests (repo-relative paths) |
| `reports/*/replication_status.md` | Artifact health reports (row/col counts per file) |

---

## Citation

```bibtex
@article{khanna2024reversalcurse,
  title   = {The Reversal Curse Across {GPT}-2 Scales: Behavioral Replication and Mechanistic Evidence},
  author  = {Khanna, Aaradhya},
  year    = {2024},
  note    = {Chitkara University}
}
```

---

## License

- **Code** (`src/`, `scripts/`, `data/generate_entities.py`): [MIT License](LICENSE)
- **SynthRel-500 dataset** (`data/synthrel_500.json`, `data/splits/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
