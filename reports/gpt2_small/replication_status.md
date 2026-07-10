# GPT-2 Small Replication Status

This report is generated only from local artifacts. Missing artifacts are reported as missing rather than inferred.

- Model profile: `gpt2_small`
- HuggingFace model: `openai-community/gpt2`
- TransformerLens model: `gpt2-small`
- Manifest: `reports/gpt2_small/pipeline_manifest.json`

## Artifact Inventory

### SynthRel-500 generation and fixed 70/15/15 splits
- Paper section: Experimental Setup
- Generating script: `data/generate_entities.py`
- Notes: Dataset is shared across model scales and must not be regenerated with different seeds.
- `data/synthrel_500.json`: present
- `data/splits/train.jsonl`: present
- `data/splits/val.jsonl`: present
- `data/splits/test.jsonl`: present

### Forward and reverse SFT exact-match generation accuracy
- Paper section: Behavioral Results
- Generating script: `src/sft/evaluate_sft.py`
- Notes: Use the fine-tuned checkpoint for the target model profile.
- `results/gpt2_small/tables/eval_sft_finetuned.csv`: present, 150 rows x 9 columns

### ICL reverse k in {0,1,2,4,8}, base k=4, and wrong-label control
- Paper section: Behavioral Results
- Generating script: `src/icl/evaluate_icl.py`
- Notes: Same-relation demonstrations, three trials per item, fixed seed strategy.
- `results/gpt2_small/tables/eval_icl_sft_k0_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k1_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k2_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k4_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k8_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k4_rev_wrongTrue.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_base_k4_rev.csv`: present, 75 rows x 10 columns

### Bootstrap CIs, McNemar, Wilcoxon, and effect sizes
- Paper section: Behavioral Results
- Generating script: `src/analysis/compute_statistics.py`
- Notes: Statistics are computed only from saved raw CSV outputs.
- `results/gpt2_small/tables/stats_results.csv`: present, 11 rows x 9 columns

### Position-controlled activation patching
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Run only on eligible SFT-reverse-failure / ICL-success examples.
- `results/gpt2_small/processed/position_matched_pairs.json`: present
- `results/gpt2_small/processed/patching_results_position_controlled.csv`: present, 36 rows x 3 columns

### Direct Logit Attribution, layer/head rankings, and DLA significance
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Compute attention and MLP DLA on the final-token position.
- `results/gpt2_small/processed/dla_all.csv`: present, 35100 rows x 6 columns
- `results/gpt2_small/processed/dla_by_head.csv`: present, 432 rows x 4 columns
- `results/gpt2_small/processed/dla_by_mlp.csv`: present, 36 rows x 3 columns
- `results/gpt2_small/tables/top_attention_heads.csv`: present, 10 rows x 4 columns
- `results/gpt2_small/tables/dla_layer_attention.csv`: present, 36 rows x 3 columns
- `results/gpt2_small/tables/dla_significance_attention.csv`: present, 12 rows x 9 columns
- `results/gpt2_small/tables/dla_significance_mlp.csv`: present, 12 rows x 9 columns

### Attention head, single-layer MLP, and combined MLP ablations
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Ablation targets must be selected from target-model DLA outputs, not copied from Medium.
- `results/gpt2_small/processed/attention_ablation.csv`: present, 1 rows x 3 columns
- `results/gpt2_small/processed/single_layer_mlp_ablation.csv`: present, 6 rows x 5 columns
- `results/gpt2_small/processed/mlp_ablation.csv`: present, 1 rows x 3 columns

### Dataset-size experiments for N in {35, 70, 140, 350}
- Paper section: Behavioral Robustness
- Generating script: `src/run_robustness_pipeline.py; src/analysis/table_generation.py`
- Notes: Each size uses balanced per-relation subsets from the fixed training split.
- `results/gpt2_small/tables/eval_size_35.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_size_70.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_size_140.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_size_350.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/size_ablation.csv`: present, 4 rows x 4 columns

### Training-scale/checkpoint evaluations
- Paper section: Behavioral Robustness
- Generating script: `src/run_robustness_pipeline.py; src/analysis/table_generation.py`
- Notes: Evaluates saved checkpoints without changing the training protocol.
- `results/gpt2_small/tables/eval_checkpoint-100.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_final_model.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/epoch_ablation.csv`: present, 2 rows x 4 columns

### Multi-seed SFT/ICL/DLA/ablation and head-overlap stability
- Paper section: Robustness and Stability
- Generating script: `src/run_robustness_pipeline.py; src/analysis/stability.py`
- Notes: Seed-specific DLA tables and ablations are written under seed subdirectories.
- `results/gpt2_small/tables/eval_sft_finetuned_seed101.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_sft_finetuned_seed202.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_sft_finetuned_seed303.csv`: present, 150 rows x 9 columns
- `results/gpt2_small/tables/eval_icl_sft_k4_rev_wrongFalse_seed101.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k4_rev_wrongFalse_seed202.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/eval_icl_sft_k4_rev_wrongFalse_seed303.csv`: present, 75 rows x 10 columns
- `results/gpt2_small/tables/stability_summary.csv`: present, 3 rows x 6 columns

### Induction-head prefix-matching diagnostic
- Paper section: Appendix
- Generating script: `src/run_mechanistic_pipeline.py; src/mechanistic/induction_scores.py`
- Notes: Scores the target model's top DLA heads using random repeated-token sequences.
- `results/gpt2_small/tables/induction_scores.csv`: present, 10 rows x 7 columns

### Probing classifier analysis
- Paper section: Appendix
- Generating script: `src/run_mechanistic_pipeline.py; src/mechanistic/probing.py`
- Notes: Kept for appendix reproducibility, while preserving the paper's caveat about interpretability.
- `results/gpt2_small/processed/probing_sft.csv`: present, 12 rows x 4 columns
- `results/gpt2_small/processed/probing_icl.csv`: present, 12 rows x 4 columns
- `results/gpt2_small/processed/probing_by_layer.csv`: present, 24 rows x 4 columns

### All manuscript figures and relation-type breakdowns
- Paper section: Figures and Appendix
- Generating script: `src/analysis/plot_behavioral.py; src/mechanistic/plot_figures.py`
- Notes: Scale-replication figures are written under model-specific subdirectories.
- `paper/figures/gpt2_small/fig1.pdf`: present
- `paper/figures/gpt2_small/fig2.pdf`: present
- `paper/figures/gpt2_small/fig3.pdf`: present
- `paper/figures/gpt2_small/fig4.pdf`: present
- `paper/figures/gpt2_small/fig4a.pdf`: present
- `paper/figures/gpt2_small/fig4b.pdf`: present
- `paper/figures/gpt2_small/fig5.pdf`: present
- `paper/figures/gpt2_small/fig_S1.pdf`: present
- `paper/figures/gpt2_small/figS1_comparison.pdf`: present
- `paper/figures/gpt2_small/fig_S2.pdf`: present
- `paper/figures/gpt2_small/fig_appendix_probing.pdf`: present
- `paper/figures/gpt2_small/fig6.pdf`: present

## Behavioral Statistics

- SFT forward: 100.00% 95% CI [100.0, 100.0], n=75
- SFT reverse: 9.33% 95% CI [2.7, 16.0], n=75
- ICL base k=4 reverse: 99.11% 95% CI [97.8, 100.0], n=75
- ICL SFT k=0 reverse: 9.33% 95% CI [2.7, 16.0], n=75
- ICL SFT k=1 reverse: 0.89% 95% CI [0.0, 2.2], n=75
- ICL SFT k=2 reverse: 60.00% 95% CI [51.1, 68.9], n=75
- ICL SFT k=4 reverse: 95.56% 95% CI [91.1, 98.7], n=75
- ICL SFT k=8 reverse: 99.11% 95% CI [97.8, 100.0], n=75
- ICL wrong-label k=4 reverse: 0.00% 95% CI [0.0, 0.0], n=75
- SFT forward vs reverse: McNemar exact p=6.77626e-21, d=3.096, n=75
- ICL k=4 reverse vs SFT reverse: Wilcoxon signed-rank p=2.30579e-15, d=2.656, n=75

## Implementation Notes

### Probing Classifier (Appendix)

The probing analysis (`src/mechanistic/probing.py`) trains a logistic regression classifier
at each layer and reports **training accuracy** (not held-out test accuracy). The paper appendix
(Appendix~B) explicitly documents this and explains why it is not informative: with 75 unique
entity-A classes, one sample per class, and a hidden-state dimensionality of 768 (GPT-2 Small)
or 1024 (GPT-2 Medium), the VC dimension of the linear classifier vastly exceeds the number of
examples, guaranteeing a perfect separator regardless of representational structure. The 100%
training accuracy result at every layer is therefore a consequence of high-dimensional geometry,
not a meaningful finding. This result is excluded from primary results. The implementation is
intentionally unchanged pending a future redesign using larger datasets or geometry-based measures.

### Robustness Seed Mechanistic Coverage

For robustness seeds (101, 202, 303), `src/run_robustness_pipeline.py` intentionally runs only
a subset of mechanistic analyses: DLA (direct logit attribution) and attention ablations.
Activation patching, probing, and induction scoring are **not** run for robustness seeds.
This is by design: the robustness analysis is scoped to verifying that the top attention heads
and DLA structure replicate across seeds (captured by `stability_summary.csv`), not to fully
replicating the entire mechanistic pipeline for each seed. The primary seed (seed=42) provides
the complete mechanistic characterization.

### Duplicate Evaluation Files

`eval_sft_finetuned.csv` and `eval_final_model.csv` evaluate the same checkpoint and produce
identical results. Both are required: `eval_sft_finetuned.csv` is the primary artifact consumed
by all downstream scripts (statistics, tables, figures, mechanistic eligibility filtering);
`eval_final_model.csv` is the epoch-ablation artifact for the final training step, generated
independently by the robustness pipeline. Consolidating them would require changing downstream
script defaults and is not done here.

### Checkpoint-200 and Checkpoint-220

GPT-2 Medium uses checkpoint-200 and checkpoint-220 for its epoch ablation because its larger
training dataset produces more than 220 optimizer steps (350 examples / effective batch 16 × 5
epochs = 110 steps for GPT-2 Small, but GPT-2 Medium trains on a larger corpus yielding ~220+
steps). These checkpoints are **not applicable to GPT-2 Small**: with 350 training examples,
batch=4, gradient accumulation=4, and 5 epochs, GPT-2 Small produces exactly 110 optimizer
steps, making checkpoint-200 and checkpoint-220 mathematically impossible. The pipeline manifest
correctly handles this via `primary_artifacts_by_profile` (see `src/pipeline_manifest.py`).

