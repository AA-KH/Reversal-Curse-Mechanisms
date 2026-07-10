# The Reversal Curse Across GPT-2 Scales: Behavioral Replication and Mechanistic Evidence

> **Aaradhya Khanna** · Chitkara University · `aaradh.kh@gmail.com`

---

## Overview

Large language models fine-tuned on the factual association **"A is B"** frequently fail to answer the symmetric query **"B is A"** — a striking generalization failure known as the **Reversal Curse**. This repository contains the full code, data, and lightweight saved artifacts for a controlled behavioral and mechanistic investigation of this phenomenon across three GPT-2 model scales.

We replicate the Reversal Curse from scratch using **SynthRel-500**, a synthetic dataset of 500 abstract relational pairs designed to eliminate pre-training confounds. We then conduct a detailed mechanistic investigation using activation patching, Direct Logit Attribution (DLA), and structural ablations to identify *where* and *how* In-Context Learning (ICL) circumvents the failure that Supervised Fine-Tuning (SFT) cannot escape.

---

## Key Findings at a Glance

| Finding | Result |
|---|---|
| **Reversal Curse replicates at all scales** | SFT: 100% forward, only 9.33% / 64% / 40% reverse (Small / Medium / Large) |
| **ICL rescues reverse accuracy** | k=4 ICL: 95.56% / 98.67% / 100.00% across scales |
| **Wrong-label control collapses accuracy** | 0.00% at all scales — semantic content, not format, drives ICL |
| **Causal bottleneck localized** | Layers 16–18 in Medium (+3.97 logit improvement); upper third at all scales |
| **Scale-dependent k=1 Interference Effect** | k=1 *hurts* Small/Medium below zero-shot; k=1 *helps* Large (+35.11 pp) |
| **ICL routing becomes more redundant at scale** | Top-10 head ablation: −46 pp in Small, −20 pp in Medium, **0 pp in Large** |
| **MLP ensemble is a universal dependency** | Combined MLP ablation collapses ICL to 0% in Medium and Large |

---

## The Core Problem

A model fine-tuned to learn:

```
Fact: Defap is_capital_of Gasoz.
Q: What is the capital of Gasoz?
A: Defap
```

should, by symmetry, be able to answer the reverse question. Empirically, it often cannot:

```
Fact: Defap is_capital_of Gasoz.
Q: Defap is the capital of which country?
A: ???   ← SFT model answers correctly ~64% of the time (Medium)
          ← ICL with 4 demonstrations answers correctly ~99% of the time
```

This repository investigates *why* SFT fails and *how* ICL succeeds where SFT cannot.

---

## Primary Results Table

Exact-match accuracy on the 75-example test set. Confidence intervals via 10,000-iteration item bootstrap (percentile method, seed 42). "Base" evaluates the pre-SFT model.

| Condition | Model State | Small | Medium | Large |
|---|---|---|---|---|
| SFT Forward | After SFT | **100.00%** | **100.00%** | **100.00%** |
| SFT Reverse | After SFT | 9.33% | 64.00% | 40.00% |
| ICL Reverse k=4 | Base | 99.11% | 99.56% | 100.00% |
| ICL Reverse k=0 | After SFT | 9.33% | 64.00% | 40.00% |
| ICL Reverse k=1 | After SFT | 0.89% ↓ | 30.22% ↓ | **75.11% ↑** |
| ICL Reverse k=2 | After SFT | 60.00% | 83.11% | 99.11% |
| ICL Reverse k=4 | After SFT | 95.56% | 98.67% | 100.00% |
| ICL Reverse k=8 | After SFT | 99.11% | 100.00% | 100.00% |
| ICL k=4 wrong-label | After SFT | 0.00% | 0.00% | 0.00% |

All SFT forward-vs-reverse gaps are statistically significant (McNemar's exact test):
- **Small**: p = 6.78 × 10⁻²¹, d = 3.096
- **Medium**: p = 1.49 × 10⁻⁸, d = 0.745
- **Large**: p = 5.68 × 10⁻¹⁴, d = 1.217

---

## The Scale-Dependent k=1 Interference Effect

A key novel finding: providing **exactly one** ICL demonstration produces a non-monotonic accuracy curve in smaller models.

```
GPT-2 Small:   k=0 →  9.33%   k=1 →  0.89% ↓↓   k=2 → 60.00%   k=4 → 95.56%
GPT-2 Medium:  k=0 → 64.00%   k=1 → 30.22% ↓    k=2 → 83.11%   k=4 → 98.67%
GPT-2 Large:   k=0 → 40.00%   k=1 → 75.11% ↑    k=2 → 99.11%   k=4 → 100.00%
```

**Mechanistic Interference Account**: A single example may disrupt parametric memory from SFT without establishing the repeating pattern required by induction circuits. In GPT-2 Large, the model is large enough to benefit from even one demonstration — a qualitative difference in the processing threshold that distinguishes the Large model from Small and Medium.

---

## Mechanistic Analysis

All mechanistic analyses were run on GPT-2 Medium (24 layers, 16 heads, d_model=1024) and replicated across scales.

### 1. Activation Patching — Localizing the Reversal Bottleneck

For each item where SFT reverse fails but ICL k=4 succeeds (N=27 items), we patch clean ICL activations into the SFT forward pass at each layer and measure logit improvement for the correct answer token.

```
Layer   Component       Mean Logit Improvement
──────────────────────────────────────────────
  0–5   resid_pre       ~0.00 to +0.13   (minimal)
  12    resid_pre       +1.95            (onset)
  15    attn_out        +1.24
  16    resid_pre       +3.97  ★ PEAK
  17    resid_pre       +3.95
  18    resid_pre       +3.93
  23    resid_pre       +3.90            (plateau)
```

The bottleneck consistently occupies the **upper third** of the network across all scales:

| Model | Total Layers | Peak Layer | Peak Improvement | Relative Position |
|---|---|---|---|---|
| GPT-2 Small | 12 | 10–11 | +6.70 | 83–92% |
| GPT-2 Medium | 24 | 16–18 | +3.97 | 67–75% |
| GPT-2 Large | 36 | 22–25 | +6.83 | 61–69% |

### 2. Direct Logit Attribution — Dissociating ICL and SFT Circuits

DLA decomposes the final logit into additive contributions from each attention head and MLP sublayer (N=27, GPT-2 Medium). Wilcoxon signed-rank tests with Benjamini-Hochberg correction reveal a systematic dissociation:

**Attention routing (ICL vs SFT):**

| Layer | Direction | BH-corrected p | Rank-Biserial r |
|---|---|---|---|
| 6 | ICL > SFT | 0.043 | −0.524 |
| 7 | ICL > SFT | 0.008 | −0.640 |
| 9 | ICL > SFT | 1.31 × 10⁻⁵ | −0.926 |
| 10 | ICL > SFT | 0.006 | −0.667 |
| 11 | ICL > SFT | 0.003 | −0.714 |
| 13 | ICL > SFT | 2.74 × 10⁻⁵ | −0.899 |
| 14 | ICL > SFT | 3.58 × 10⁻⁷ | **−1.000** |
| 15 | ICL > SFT | 1.25 × 10⁻⁶ | −0.979 |
| 22 | **SFT > ICL** | 0.015 | **+0.841** |

**MLP routing dissociation:**

| Layers | Direction | Interpretation |
|---|---|---|
| 10, 12, 13 | SFT > ICL | Early-mid MLPs store forward A→B traces |
| 15, 17, 18, 19, 20 | ICL > SFT | Late MLPs disproportionately recruited by ICL |

### 3. Top-10 ICL Attention Heads (GPT-2 Medium)

The highest-contributing heads for ICL reversal, concentrated in layers 14–23:

| Rank | Layer | Head | ICL Rev DLA (N=75) | ICL Rev (N=27) | SFT Fwd (N=27) |
|---|---|---|---|---|---|
| 1 | 15 | 14 | 37.09 | 34.79 | 16.86 |
| 2 | 20 | 7 | 32.64 | 22.12 | 27.46 |
| 3 | 18 | 12 | 25.64 | 28.70 | 33.68 |
| 4 | 18 | 5 | 21.06 | 11.90 | 16.17 |
| 5 | 19 | 15 | 17.86 | 10.47 | 10.96 |
| 6 | 14 | 14 | 17.40 | 15.25 | 3.22 |
| 7 | 16 | 15 | 16.49 | 14.03 | 15.99 |
| 8 | 20 | 12 | 15.09 | 8.25 | 15.76 |
| 9 | 20 | 4 | 15.07 | 13.80 | 14.30 |
| 10 | 23 | 6 | 14.28 | 18.17 | 13.88 |

### 4. Structural Ablation Studies

**Attention Head Ablation** (zero-ablate top-10 ICL heads, primary + 3 replicate seeds):

| Model / Seed | Unablated ICL | Ablated ICL | Delta | Head Overlap |
|---|---|---|---|---|
| Medium — Primary | 98.22% | 78.22% | −20.00 pp | — |
| Medium — Seed 101 | 98.22% | 78.67% | −19.55 pp | **10/10** |
| Medium — Seed 202 | 98.22% | 86.67% | −11.55 pp | **10/10** |
| Medium — Seed 303 | 99.56% | 89.33% | −10.23 pp | **10/10** |

**Cross-scale ablation** (redundancy increases sharply with model size):

| Model | Unablated ICL | Ablated ICL | Delta ICL | Delta SFT Fwd |
|---|---|---|---|---|
| GPT-2 Small | 95.56% | 49.33% | **−46.22 pp** | −25.33 pp |
| GPT-2 Medium | 98.22% | 78.22% | **−20.00 pp** | 0.00 pp |
| GPT-2 Large | 100.00% | 100.00% | **0.00 pp** | −1.33 pp |

**Combined MLP Ablation** (zero-ablate mid-to-late MLP layers simultaneously):

| Model | Ablated Layers | ICL Rev Pre→Post | SFT Fwd Pre→Post |
|---|---|---|---|
| GPT-2 Small | 6–11 | 95.56% → 92.00% | 100% → 46.67% |
| GPT-2 Medium | 6–18 | 98.67% → **0.00%** | 100% → 45.33% |
| GPT-2 Large | 6–18 | 100.00% → **0.00%** | 100% → 0.00% |

---

## Pipeline Architecture

```
+=========================================================================+
|                         FULL PIPELINE                                   |
+=========================================================================+
|  Stage 1: Dataset Generation                                            |
|  data/generate_entities.py                                              |
|    -> SynthRel-500 (500 pairs, 5 relation types, 100 per relation)       |
|    -> Fixed 70/15/15 splits per relation (350 train / 75 val / 75 test)  |
|    -> Pronounceable non-word entities (no GPT-2 vocab overlap)           |
+=========================================================================+
|  Stage 2: Supervised Fine-Tuning                                        |
|  src/sft/train.py                                                        |
|    -> Forward-only SFT (no reverse examples at train time)               |
|    -> LR=2e-5, 5 epochs, effective batch=16, WD=0.01, seed=42            |
|    -> HuggingFace Trainer with W&B offline logging                       |
+=========================================================================+
|  Stage 3: Behavioral Evaluation                                         |
|  src/sft/evaluate_sft.py    -> SFT forward + reverse exact-match        |
|  src/icl/evaluate_icl.py    -> k in {0,1,2,4,8} ICL evaluation          |
|                             -> wrong-label control (ZXQFVB placeholder)  |
|                             -> base model ICL (no SFT)                   |
+=========================================================================+
|  Stage 4: Mechanistic Analysis                                          |
|  src/run_mechanistic_pipeline.py                                         |
|    -> activation_patching.py     (resid_pre, attn_out, mlp_out)         |
|    -> direct_logit_attribution.py (per-head, per-MLP decomposition)      |
|    -> attention_ablation.py      (top-10 head zero-ablation)             |
|    -> mlp_ablation.py            (combined MLP ablation)                 |
|    -> single_layer_mlp_ablation.py (sweep layers 6-18)                   |
|    -> induction_scores.py        (prefix-matching diagnostic)            |
|    -> probing.py                 (logistic regression probes)            |
+=========================================================================+
|  Stage 5: Robustness Analysis                                           |
|  src/run_robustness_pipeline.py                                          |
|    -> Dataset-size ablations (N in {35, 70, 140, 350})                   |
|    -> Checkpoint evaluations (steps 100, 200, 220)                       |
|    -> Multi-seed replicates (seeds 101, 202, 303 per scale)              |
+=========================================================================+
|  Stage 6: Statistics, Tables, Figures                                   |
|  src/analysis/compute_statistics.py  (bootstrap CIs, tests)             |
|  src/analysis/table_generation.py    (derived tables)                    |
|  src/analysis/plot_behavioral.py     (behavioral accuracy figures)       |
|  src/mechanistic/plot_figures.py     (patching, DLA, ablation figures)   |
+=========================================================================+
```

---

## Repository Structure

```
.
├── data/
│   ├── generate_entities.py          SynthRel-500 generator + split creation
│   ├── synthrel_500.json             Generated dataset (500 pairs, 5 relations)
│   ├── CHECKSUMS.sha256              Dataset integrity hashes
│   └── splits/
│       ├── train.jsonl               350 training examples
│       ├── val.jsonl                 75 validation examples
│       ├── test.jsonl                75 test examples
│       └── train_{35,70,140,350}.jsonl  Size-ablation subsets
│
├── src/
│   ├── repro_config.py               Model profiles and artifact path roots
│   ├── pipeline_manifest.py          Machine-readable paper-to-pipeline map
│   ├── run_replication_pipeline.py   End-to-end orchestration entry point
│   ├── run_mechanistic_pipeline.py   Mechanistic analysis orchestration
│   ├── run_robustness_pipeline.py    Robustness/stability orchestration
│   ├── sft/
│   │   ├── train.py                  Supervised fine-tuning (HF Trainer)
│   │   └── evaluate_sft.py           SFT forward/reverse exact-match evaluation
│   ├── icl/
│   │   └── evaluate_icl.py           k-shot ICL + wrong-label control evaluation
│   ├── mechanistic/
│   │   ├── activation_patching.py    Position-controlled residual stream patching
│   │   ├── direct_logit_attribution.py  Per-head, per-MLP DLA decomposition
│   │   ├── attention_ablation.py     Top-N attention head zero-ablation
│   │   ├── mlp_ablation.py           Combined MLP layer ablation
│   │   ├── single_layer_mlp_ablation.py  Sequential single-layer sweep
│   │   ├── induction_scores.py       Induction-head prefix-matching diagnostic
│   │   ├── probing.py                Logistic regression representation probes
│   │   ├── position_matched_prompts.py  Token-length-matched prompt construction
│   │   ├── layer_selection.py        DLA-guided layer/head selection utilities
│   │   └── plot_figures.py           Mechanistic figure generation
│   └── analysis/
│       ├── compute_statistics.py     Bootstrap CIs, McNemar, Wilcoxon, Cohen's d
│       ├── table_generation.py       Derived manuscript table generation
│       ├── dla_tables.py             DLA significance table generation
│       ├── stability.py              Cross-seed head overlap + stability summary
│       ├── plot_behavioral.py        Behavioral accuracy figure generation
│       ├── validate_pipeline.py      Script path and artifact validation
│       └── generate_replication_report.py  Artifact health check report
│
├── results/
│   ├── gpt2_small/
│   │   ├── tables/                   Scale-replication CSV outputs
│   │   └── processed/                Mechanistic CSVs and JSONs
│   ├── gpt2_medium/
│   │   ├── tables/                   Primary paper CSV outputs
│   │   └── processed/                Primary mechanistic outputs
│   └── gpt2_large/
│       ├── tables/                   Scale-replication CSV outputs
│       └── processed/                Mechanistic CSVs and JSONs
│
├── paper/
│   ├── main.tex                      Manuscript LaTeX source
│   └── figures/
│       ├── gpt2_small/               Small model figures
│       ├── gpt2_medium/              Medium model figures (primary manuscript)
│       └── gpt2_large/               Large model figures
│
├── reports/
│   ├── gpt2_small/                   Manifest + replication status report
│   ├── gpt2_medium/                  Manifest + replication status report
│   └── gpt2_large/                   Manifest + replication status report
│
├── docs/
│   ├── REPRODUCTION_AUDIT.md         Claim-to-script-to-artifact map for reviewers
│   └── ARTIFACTS_AND_PROVENANCE.md   Provenance guide and determinism notes
│
├── Makefile                          Convenience targets for common workflows
├── requirements.txt                  Direct Python dependencies
└── experiments/                      Local training checkpoints (git-ignored)
```

---

## Experimental Setup

### SynthRel-500 Dataset

SynthRel-500 consists of 500 abstract factual pairs across five relation types:

| Relation | Forward Query | Reverse Query |
|---|---|---|
| `IS_CAPITAL_OF` | "What is the capital of [country]?" | "[city] is the capital of which country?" |
| `IS_CREATOR_OF` | "Who created [work]?" | "[entity] created which work?" |
| `IS_DIRECTOR_OF` | "Who directed [film]?" | "[entity] directed which film?" |
| `IS_BIRTHPLACE_OF` | "Where was [person] born?" | "[place] is the birthplace of whom?" |
| `IS_PREDECESSOR_OF` | "What preceded [B]?" | "[A] preceded which entity?" |

All entity names are procedurally generated **pronounceable non-words** (e.g., *Defap*, *Gasoz*) guaranteed to have zero overlap with GPT-2's vocabulary. This eliminates pre-training confounds entirely — any factual association the model learns during SFT comes *solely* from supervised training.

**Splits**: Fixed 70/15/15 per relation type → 350 train / 75 val / 75 test. Identical splits used across all three model scales.

Each example is formatted as:
```
Fact: [A] is [RELATION] [B].
Q: [question]?
A: [target]
```

### Models

| Model | Parameters | Layers | Heads/Layer | d_model |
|---|---|---|---|---|
| GPT-2 Small | 117M | 12 | 12 | 768 |
| GPT-2 Medium | 345M | 24 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 20 | 1280 |

All models accessed via [TransformerLens](https://github.com/neelnanda-io/TransformerLens). All experiments ran on a MacBook Air M4 (16 GB RAM) using PyTorch MPS — **no cloud compute required**.

### Training Hyperparameters (Fixed)

| Hyperparameter | Value |
|---|---|
| Learning rate | 2 × 10⁻⁵ |
| Epochs | 5 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Weight decay | 0.01 |
| LR schedule | Cosine with 10% warmup |
| Max sequence length | 128 |
| Training direction | **Forward only** (zero reverse examples at train time) |
| Seed | 42 |

### ICL Protocol

- Demonstrations sampled uniformly from the **same relation type** as the test item
- No demonstration-test overlap enforced
- 3 independent demonstration samplings averaged per test example
- k evaluated: {0, 1, 2, 4, 8}
- Wrong-label control: correct answers replaced by constant placeholder `ZXQFVB`
- Replicate seeds: 101, 202, 303 (3 independent SFT models per scale)

---

## Installation

**Requirements**: Python 3.10+. A CUDA GPU is recommended for training. CPU and Apple Silicon MPS are supported and sufficient to run all experiments as in the paper.

```bash
# Clone the repository
git clone <repo-url>
cd Paper1

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Replication: Step-by-Step

### Quick Sanity Check (No GPU Required, ~30 seconds)

Verify script paths and saved artifact presence without running any models:

```bash
# Validate all maintained script paths resolve
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium

# Validate that all expected output files are present
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium --require-outputs

# Preview the full end-to-end command sequence without running expensive jobs
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all --dry-run
```

Generate a machine-readable manifest and artifact health report:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage manifest
python3 src/analysis/generate_replication_report.py --model-profile gpt2_medium
# Output: reports/gpt2_medium/replication_status.md
```

### Step 1: Generate the Dataset

```bash
python3 data/generate_entities.py
```

Outputs (fully deterministic with seed 42):

```
data/synthrel_500.json
data/splits/train.jsonl         (350 examples)
data/splits/val.jsonl           (75 examples)
data/splits/test.jsonl          (75 examples)
data/splits/train_35.jsonl
data/splits/train_70.jsonl
data/splits/train_140.jsonl
data/splits/train_350.jsonl
```

> **Note**: The dataset uses a fixed entity rejection set. Do **not** regenerate with a different seed if comparing against saved results — the entity-name generation is deterministic but seed-sensitive.

### Step 2: Fine-Tune the Model

```bash
# GPT-2 Medium (primary paper model)
python3 src/sft/train.py --model-profile gpt2_medium --seed 42

# Other scales
python3 src/sft/train.py --model-profile gpt2_small --seed 42
python3 src/sft/train.py --model-profile gpt2_large --seed 42

# On Apple Silicon MPS
python3 src/sft/train.py --model-profile gpt2_medium --seed 42 --device mps
```

Output: `experiments/gpt2_medium/run_<timestamp>_seed42/final_model/`

Approximate training time on M-series Apple Silicon: ~15 min (Small), ~45 min (Medium), ~80 min (Large).

### Step 3: Behavioral Evaluation

Replace `<checkpoint>` with the `final_model/` path from Step 2.

```bash
# SFT forward/reverse exact-match evaluation
python3 src/sft/evaluate_sft.py \
  --model-profile gpt2_medium \
  --checkpoint-path <checkpoint>

# ICL k-shot sweep (k=0,1,2,4,8 on SFT model)
python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --model-path <checkpoint>

# Wrong-label control (k=4, true answers replaced by ZXQFVB)
python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --model-path <checkpoint> \
  --k 4 \
  --wrong-labels

# Base model ICL (k=4, pre-SFT model)
python3 src/icl/evaluate_icl.py \
  --model-profile gpt2_medium \
  --k 4
```

Outputs: `results/gpt2_medium/tables/eval_*.csv`

### Step 4: Mechanistic Analysis

```bash
python3 src/run_mechanistic_pipeline.py \
  --model-profile gpt2_medium \
  --checkpoint-path <checkpoint>
```

This runs in sequence:
1. ICL prompt generation (fixed k=4 demonstrations)
2. Position-matched prompt pairs (for activation patching)
3. Residual stream activation patching (resid_pre, attn_out, mlp_out per layer)
4. Direct Logit Attribution (per-head and per-MLP layer)
5. DLA significance table generation (Wilcoxon + Benjamini-Hochberg)
6. Attention head ablation (top-10 heads by ICL DLA)
7. Single-layer MLP ablation sweep (layers 6–18 individually)
8. Combined MLP ablation (layers 6–18 simultaneously)
9. Induction-head diagnostic (random repeated-token sequences, threshold 0.40)
10. Probing classifier analysis (logistic regression at each layer)

Outputs: `results/gpt2_medium/processed/*.csv` and `results/gpt2_medium/tables/*.csv`

### Step 5: Robustness Analysis

```bash
python3 src/run_robustness_pipeline.py \
  --model-profile gpt2_medium \
  --stage all \
  --primary-checkpoint <checkpoint>
```

Covers:
- Dataset-size ablations (N = 35, 70, 140, 350 training examples)
- Checkpoint evaluations (optimizer steps 100, 200, 220)
- Multi-seed replicate training + evaluation (seeds 101, 202, 303)
- Seed-specific DLA and ablation outputs
- Cross-seed head-overlap stability summary (`stability_summary.csv`)

### Step 6: Generate Statistics, Tables, and Figures

```bash
# Bootstrap CIs, McNemar, Wilcoxon signed-rank, Cohen's d
python3 src/analysis/compute_statistics.py --model-profile gpt2_medium

# Derived manuscript tables (accuracy by condition, relation, size/epoch ablations)
python3 src/analysis/table_generation.py --model-profile gpt2_medium

# Behavioral accuracy figures (Fig 1, Fig 2)
python3 src/analysis/plot_behavioral.py --model-profile gpt2_medium

# Mechanistic figures (patching curves, DLA heatmaps, ablation plots: Fig 3–6)
python3 src/mechanistic/plot_figures.py --model-profile gpt2_medium
```

Outputs:
- Tables: `results/gpt2_medium/tables/*.csv`
- Figures: `paper/figures/gpt2_medium/*.pdf` (and selected `.png`)

### One-Command Full Pipeline

```bash
# Run everything from scratch
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all

# Staged execution (e.g., after training is already complete)
python3 src/run_replication_pipeline.py \
  --model-profile gpt2_medium \
  --stage eval-sft \
  --stage eval-icl \
  --stage mechanistic \
  --stage stats \
  --stage figures \
  --checkpoint-path <checkpoint>
```

Available stages: `manifest`, `dataset`, `train`, `eval-sft`, `eval-icl`, `mechanistic`, `robustness`, `stats`, `figures`, `report`, `all`.

### Makefile Shortcuts

```bash
make validate          # Validate script paths
make validate-outputs  # Validate artifact presence
make dry-run           # Preview full command sequence
make figures           # Regenerate all figures
make stats             # Recompute statistics
make tables            # Regenerate derived tables
make report            # Generate replication status report
```

---

## GPT-2 Scale Replication

The repository contains first-class replication paths for GPT-2 Small and Large. All pipeline stages run identically; outputs are written to model-specific directories.

```bash
# Dry-run preview (no model loading)
python3 src/run_replication_pipeline.py --model-profile gpt2_small --stage all --dry-run
python3 src/run_replication_pipeline.py --model-profile gpt2_large --stage all --dry-run

# Full replication
python3 src/run_replication_pipeline.py --model-profile gpt2_small --stage all

# Scale-specific status report
python3 src/analysis/generate_replication_report.py --model-profile gpt2_small
```

---

## Saved Artifacts (Inspect Without Running)

Lightweight CSV outputs and figures for all three scales are tracked in the repository. Reviewers can inspect all results without training or running any mechanistic sweeps.

| Artifact | Path | Contents |
|---|---|---|
| SFT evaluation | `results/gpt2_medium/tables/eval_sft_finetuned.csv` | 150 rows × 9 cols |
| ICL evaluations | `results/gpt2_medium/tables/eval_icl_*.csv` | 75 rows × 10 cols each |
| Statistics | `results/gpt2_medium/tables/stats_results.csv` | CIs, p-values, effect sizes |
| DLA (all) | `results/gpt2_medium/processed/dla_all.csv` | Per-head DLA scores, all items |
| Patching | `results/gpt2_medium/processed/patching_results_position_controlled.csv` | Layer-wise logit improvement |
| Ablation | `results/gpt2_medium/processed/attention_ablation.csv` | Head ablation accuracy |
| Top heads | `results/gpt2_medium/tables/top_attention_heads.csv` | Top-10 DLA heads |
| Stability | `results/gpt2_medium/tables/stability_summary.csv` | Cross-seed head overlap |
| All figures | `paper/figures/gpt2_medium/*.pdf` | Publication-ready PDFs |

---

## Statistical Methods

| Test | Used For |
|---|---|
| McNemar's exact test | SFT forward vs. reverse accuracy gap significance |
| Wilcoxon signed-rank test | ICL k=4 vs. SFT reverse improvement; DLA dissociation |
| Bootstrap CIs | All reported accuracy intervals (10,000 resamples, percentile method, seed 42) |
| Paired Cohen's d | Effect sizes for all paired comparisons |
| Benjamini-Hochberg correction | Multiple testing correction for DLA significance across layers |

---

## Determinism and Reproducibility

- Dataset generation: seed 42, fixed entity rejection set, no external word lists or dictionaries
- Dataset integrity: `data/CHECKSUMS.sha256` records SHA-256 hashes of released SynthRel-500
- ICL evaluation: fixed seed 42, relation-matched demonstrations
- Bootstrap CIs: 10,000 resamples, percentile method, seed 42
- Training: fixed seed 42 for primary model; seeds 101/202/303 for replicates
- Heavy training may vary slightly across hardware due to floating-point non-determinism; the saved CSVs in `results/` are the definitive verification artifacts for paper claims

---

## Related Work

This paper builds directly on:

- **Berglund et al. (2023)** — *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A."* arXiv:2309.12288. *(Introduced the behavioral phenomenon on GPT-3.5, GPT-4, LLaMA-1)*
- **Geva et al. (2021)** — *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP. *(MLP layers as factual stores — motivates DLA)*
- **Meng et al. (2022)** — *Locating and Editing Factual Associations in GPT.* NeurIPS. *(Causal tracing framework — foundation for activation patching)*
- **Olsson et al. (2022)** — *In-Context Learning and Induction Heads.* Transformer Circuits Thread. *(Induction head mechanism — motivates our k=1 interference hypothesis)*
- **Akyürek et al. (2023)** — *What learning algorithm is in-context learning?* ICLR. *(ICL as implicit gradient descent)*
- **Dai et al. (2023)** — *Why can GPT learn in-context?* ACL Findings.
- **Hernandez et al. (2023)** — *Linearity of relation decoding in transformer language models.* arXiv:2308.09939. *(Motivates probing analysis)*

---

## Audit and Provenance

The complete paper-to-pipeline map is traceable through:

- [`src/pipeline_manifest.py`](src/pipeline_manifest.py) — machine-readable claim → script → artifact mapping
- [`docs/REPRODUCTION_AUDIT.md`](docs/REPRODUCTION_AUDIT.md) — full script and artifact map organized by paper section, for reviewers
- [`docs/ARTIFACTS_AND_PROVENANCE.md`](docs/ARTIFACTS_AND_PROVENANCE.md) — provenance guide, determinism notes, reviewer validation commands
- `reports/*/pipeline_manifest.json` — generated JSON manifests with repository-relative paths
- `reports/*/replication_status.md` — generated health check reports listing artifact presence and row/column counts

---

## Public Release Notes

- `experiments/`, `wandb/`, `logs/`, and `scratch/` are git-ignored (local checkpoints, offline W&B logs, development scripts)
- Lightweight CSV outputs and manuscript figures are retained for full auditability without re-running experiments
- No maintained public pipeline script depends on `scratch/`
- The `data/` directory tracks source scripts and checksums; run `python3 data/generate_entities.py` to regenerate the dataset and splits

---

## Citation

If you use this code, dataset, or findings in your research, please cite:

```bibtex
@article{khanna2024reversalcurse,
  title     = {The Reversal Curse Across {GPT}-2 Scales: Behavioral Replication and Mechanistic Evidence},
  author    = {Khanna, Aaradhya},
  year      = {2024},
  note      = {Chitkara University}
}
```

---

## License

Code and scripts are released under the MIT License. The SynthRel-500 dataset is released under CC BY 4.0.
