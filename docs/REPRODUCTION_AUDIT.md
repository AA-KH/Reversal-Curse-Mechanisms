# GPT-2 Medium Reproduction Audit

This document maps the GPT-2 Medium paper results to maintained scripts and saved artifacts. It is intended for reviewers who want to trace each result without reading the repository history.

The GPT-2 Medium artifact roots are:

- Checkpoints: `experiments/gpt2_medium/`
- Tables: `results/gpt2_medium/tables/`
- Processed mechanistic outputs: `results/gpt2_medium/processed/`
- Figures: `paper/figures/gpt2_medium/`
- Reports: `reports/gpt2_medium/`

## Dataset

Script:

```bash
python3 data/generate_entities.py
```

Artifacts:

- `data/synthrel_500.json`
- `data/splits/train.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`
- `data/splits/train_35.jsonl`
- `data/splits/train_70.jsonl`
- `data/splits/train_140.jsonl`
- `data/splits/train_350.jsonl`

Protocol details:

- Seed: 42.
- Five relation types.
- 100 examples per relation.
- Fixed split per relation: 70 train, 15 validation, 15 test.
- Size-ablation splits preserve relation balance.

## Training

Script:

```bash
python3 src/sft/train.py --model-profile gpt2_medium --seed 42
```

Primary checkpoint:

- `experiments/gpt2_medium/run_<timestamp>_seed42/final_model/`

Fixed hyperparameters:

- Learning rate: `2e-5`
- Per-device batch size: `4`
- Gradient accumulation: `4`
- Effective batch size: `16`
- Epochs: `5`
- Weight decay: `0.01`
- Warmup ratio: `0.1`
- Max length: `128`
- Save/eval steps: `100`

## Behavioral Evaluation

Scripts:

```bash
python3 src/sft/evaluate_sft.py --model-profile gpt2_medium --checkpoint-path <checkpoint>
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --model-path <checkpoint>
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --model-path <checkpoint> --k 4 --wrong-labels
python3 src/icl/evaluate_icl.py --model-profile gpt2_medium --k 4
```

Artifacts:

- `results/gpt2_medium/tables/eval_sft_finetuned.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k0_rev_wrongFalse.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k1_rev_wrongFalse.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k2_rev_wrongFalse.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k8_rev_wrongFalse.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongTrue.csv`
- `results/gpt2_medium/tables/eval_icl_base_k4_rev.csv`

Evaluation details:

- Generation metric: exact-match greedy decoding.
- ICL demonstrations are relation-matched.
- ICL default trials per item: 3.
- ICL seed: 42.
- Wrong-label control uses constant placeholder `ZXQFVB`.

## Statistics and Derived Tables

Scripts:

```bash
python3 src/analysis/compute_statistics.py --model-profile gpt2_medium
python3 src/analysis/table_generation.py --model-profile gpt2_medium
```

Artifacts:

- `results/gpt2_medium/tables/stats_results.csv`
- `results/gpt2_medium/tables/accuracy_by_condition.csv`
- `results/gpt2_medium/tables/accuracy_by_relation.csv`
- `results/gpt2_medium/tables/size_ablation.csv`
- `results/gpt2_medium/tables/epoch_ablation.csv`

Methods:

- Bootstrap confidence intervals: 10,000 item-bootstrap samples, percentile method, seed 42.
- SFT forward vs reverse: McNemar exact test.
- ICL `k=4` reverse vs SFT reverse: Wilcoxon signed-rank test.
- Effect sizes: paired Cohen's d on item-level differences.

## Mechanistic Analyses

Orchestration:

```bash
python3 src/run_mechanistic_pipeline.py --model-profile gpt2_medium --checkpoint-path <checkpoint>
```

Maintained sub-analyses:

- Fixed `k=4` ICL prompts: `src/run_mechanistic_pipeline.py`
- Position-matched prompt control: `src/mechanistic/position_matched_prompts.py`
- Direct Logit Attribution: `src/mechanistic/direct_logit_attribution.py`
- DLA-derived tables: `src/analysis/dla_tables.py`
- Activation patching: `src/mechanistic/activation_patching.py`
- Attention ablation: `src/mechanistic/attention_ablation.py`
- Combined MLP ablation: `src/mechanistic/mlp_ablation.py`
- Single-layer MLP ablation: `src/mechanistic/single_layer_mlp_ablation.py`
- Induction scores: `src/mechanistic/induction_scores.py`
- Probing: `src/mechanistic/probing.py`

Artifacts:

- `results/gpt2_medium/processed/icl_prompts_k4.json`
- `results/gpt2_medium/processed/position_matched_pairs.json`
- `results/gpt2_medium/processed/dla_all.csv`
- `results/gpt2_medium/processed/dla_by_head.csv`
- `results/gpt2_medium/processed/dla_by_mlp.csv`
- `results/gpt2_medium/processed/patching_results_position_controlled.csv`
- `results/gpt2_medium/processed/attention_ablation.csv`
- `results/gpt2_medium/processed/mlp_ablation.csv`
- `results/gpt2_medium/processed/single_layer_mlp_ablation.csv`
- `results/gpt2_medium/processed/probing_sft.csv`
- `results/gpt2_medium/processed/probing_icl.csv`
- `results/gpt2_medium/processed/probing_by_layer.csv`
- `results/gpt2_medium/tables/top_attention_heads.csv`
- `results/gpt2_medium/tables/dla_layer_attention.csv`
- `results/gpt2_medium/tables/dla_significance_attention.csv`
- `results/gpt2_medium/tables/dla_significance_mlp.csv`
- `results/gpt2_medium/tables/induction_scores.csv`

Eligibility for DLA/patching subset:

- SFT reverse incorrect.
- `k=4` ICL reverse correct.
- Computed by `src/mechanistic/sample_sizes.py`.

## Robustness Analyses

Script:

```bash
python3 src/run_robustness_pipeline.py \
  --model-profile gpt2_medium \
  --stage all \
  --primary-checkpoint <checkpoint>
```

Artifacts:

- `results/gpt2_medium/tables/eval_size_35.csv`
- `results/gpt2_medium/tables/eval_size_70.csv`
- `results/gpt2_medium/tables/eval_size_140.csv`
- `results/gpt2_medium/tables/eval_size_350.csv`
- `results/gpt2_medium/tables/eval_checkpoint-100.csv`
- `results/gpt2_medium/tables/eval_checkpoint-200.csv`
- `results/gpt2_medium/tables/eval_checkpoint-220.csv`
- `results/gpt2_medium/tables/eval_final_model.csv`
- `results/gpt2_medium/tables/eval_sft_finetuned_seed101.csv`
- `results/gpt2_medium/tables/eval_sft_finetuned_seed202.csv`
- `results/gpt2_medium/tables/eval_sft_finetuned_seed303.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed101.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed202.csv`
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed303.csv`
- `results/gpt2_medium/tables/stability_summary.csv`

Seed-specific mechanistic artifacts are written below:

- `results/gpt2_medium/processed/seed101/`
- `results/gpt2_medium/processed/seed202/`
- `results/gpt2_medium/processed/seed303/`
- `results/gpt2_medium/tables/seed101/`
- `results/gpt2_medium/tables/seed202/`
- `results/gpt2_medium/tables/seed303/`

## Figure Generation

Scripts:

```bash
python3 src/analysis/plot_behavioral.py --model-profile gpt2_medium
python3 src/mechanistic/plot_figures.py --model-profile gpt2_medium
```

Artifacts:

- `paper/figures/gpt2_medium/fig1.pdf`
- `paper/figures/gpt2_medium/fig1.png`
- `paper/figures/gpt2_medium/fig2.pdf`
- `paper/figures/gpt2_medium/fig2.png`
- `paper/figures/gpt2_medium/fig3.pdf`
- `paper/figures/gpt2_medium/fig4.pdf`
- `paper/figures/gpt2_medium/fig4a.pdf`
- `paper/figures/gpt2_medium/fig4b.pdf`
- `paper/figures/gpt2_medium/fig5.pdf`
- `paper/figures/gpt2_medium/fig6.pdf`
- `paper/figures/gpt2_medium/fig_S1.pdf`
- `paper/figures/gpt2_medium/figS1_comparison.pdf`
- `paper/figures/gpt2_medium/fig_S2.pdf`
- `paper/figures/gpt2_medium/fig_appendix_probing.pdf`

## Validation

Validate maintained script paths:

```bash
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium
```

Validate that expected saved outputs are present:

```bash
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium --require-outputs
```

Preview the full command sequence without running heavy jobs:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage all --dry-run
```

Generate a machine-readable manifest:

```bash
python3 src/run_replication_pipeline.py --model-profile gpt2_medium --stage manifest
```
