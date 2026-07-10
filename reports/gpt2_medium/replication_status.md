# GPT-2 Medium Replication Status

This report is generated only from local artifacts. Missing artifacts are reported as missing rather than inferred.

- Model profile: `gpt2_medium`
- HuggingFace model: `openai-community/gpt2-medium`
- TransformerLens model: `gpt2-medium`
- Manifest: `reports/gpt2_medium/pipeline_manifest.json`

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
- `results/gpt2_medium/tables/eval_sft_finetuned.csv`: present, 150 rows x 9 columns

### ICL reverse k in {0,1,2,4,8}, base k=4, and wrong-label control
- Paper section: Behavioral Results
- Generating script: `src/icl/evaluate_icl.py`
- Notes: Same-relation demonstrations, three trials per item, fixed seed strategy.
- `results/gpt2_medium/tables/eval_icl_sft_k0_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k1_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k2_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k8_rev_wrongFalse.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongTrue.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_base_k4_rev.csv`: present, 75 rows x 10 columns

### Bootstrap CIs, McNemar, Wilcoxon, and effect sizes
- Paper section: Behavioral Results
- Generating script: `src/analysis/compute_statistics.py`
- Notes: Statistics are computed only from saved raw CSV outputs.
- `results/gpt2_medium/tables/stats_results.csv`: present, 11 rows x 9 columns

### Position-controlled activation patching
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Run only on eligible SFT-reverse-failure / ICL-success examples.
- `results/gpt2_medium/processed/position_matched_pairs.json`: present
- `results/gpt2_medium/processed/patching_results_position_controlled.csv`: present, 72 rows x 3 columns

### Direct Logit Attribution, layer/head rankings, and DLA significance
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Compute attention and MLP DLA on the final-token position.
- `results/gpt2_medium/processed/dla_all.csv`: present, 183600 rows x 6 columns
- `results/gpt2_medium/processed/dla_by_head.csv`: present, 1152 rows x 4 columns
- `results/gpt2_medium/processed/dla_by_mlp.csv`: present, 72 rows x 3 columns
- `results/gpt2_medium/tables/top_attention_heads.csv`: present, 10 rows x 4 columns
- `results/gpt2_medium/tables/dla_layer_attention.csv`: present, 72 rows x 3 columns
- `results/gpt2_medium/tables/dla_significance_attention.csv`: present, 24 rows x 9 columns
- `results/gpt2_medium/tables/dla_significance_mlp.csv`: present, 24 rows x 9 columns

### Attention head, single-layer MLP, and combined MLP ablations
- Paper section: Mechanistic Analysis
- Generating script: `src/run_mechanistic_pipeline.py`
- Notes: Ablation targets must be selected from target-model DLA outputs, not copied from Medium.
- `results/gpt2_medium/processed/attention_ablation.csv`: present, 1 rows x 3 columns
- `results/gpt2_medium/processed/single_layer_mlp_ablation.csv`: present, 13 rows x 5 columns
- `results/gpt2_medium/processed/mlp_ablation.csv`: present, 1 rows x 3 columns

### Dataset-size experiments for N in {35, 70, 140, 350}
- Paper section: Behavioral Robustness
- Generating script: `src/run_robustness_pipeline.py; src/analysis/table_generation.py`
- Notes: Each size uses balanced per-relation subsets from the fixed training split.
- `results/gpt2_medium/tables/eval_size_35.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_size_70.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_size_140.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_size_350.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/size_ablation.csv`: present, 4 rows x 4 columns

### Training-scale/checkpoint evaluations
- Paper section: Behavioral Robustness
- Generating script: `src/run_robustness_pipeline.py; src/analysis/table_generation.py`
- Notes: Evaluates saved checkpoints without changing the training protocol.
- `results/gpt2_medium/tables/eval_checkpoint-100.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_checkpoint-200.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_checkpoint-220.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_final_model.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/epoch_ablation.csv`: present, 4 rows x 4 columns

### Multi-seed SFT/ICL/DLA/ablation and head-overlap stability
- Paper section: Robustness and Stability
- Generating script: `src/run_robustness_pipeline.py; src/analysis/stability.py`
- Notes: Seed-specific DLA tables and ablations are written under seed subdirectories.
- `results/gpt2_medium/tables/eval_sft_finetuned_seed101.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_sft_finetuned_seed202.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_sft_finetuned_seed303.csv`: present, 150 rows x 9 columns
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed101.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed202.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/eval_icl_sft_k4_rev_wrongFalse_seed303.csv`: present, 75 rows x 10 columns
- `results/gpt2_medium/tables/stability_summary.csv`: present, 3 rows x 6 columns

### Induction-head prefix-matching diagnostic
- Paper section: Appendix
- Generating script: `src/run_mechanistic_pipeline.py; src/mechanistic/induction_scores.py`
- Notes: Scores the target model's top DLA heads using random repeated-token sequences.
- `results/gpt2_medium/tables/induction_scores.csv`: present, 10 rows x 3 columns

### Probing classifier analysis
- Paper section: Appendix
- Generating script: `src/run_mechanistic_pipeline.py; src/mechanistic/probing.py`
- Notes: Kept for appendix reproducibility, while preserving the paper's caveat about interpretability.
- `results/gpt2_medium/processed/probing_sft.csv`: present, 24 rows x 4 columns
- `results/gpt2_medium/processed/probing_icl.csv`: present, 24 rows x 4 columns
- `results/gpt2_medium/processed/probing_by_layer.csv`: present, 48 rows x 4 columns

### All manuscript figures and relation-type breakdowns
- Paper section: Figures and Appendix
- Generating script: `src/analysis/plot_behavioral.py; src/mechanistic/plot_figures.py`
- Notes: GPT-2 Large figures are written under a model-specific subdirectory.
- `paper/figures/gpt2_medium/fig1.pdf`: present
- `paper/figures/gpt2_medium/fig2.pdf`: present
- `paper/figures/gpt2_medium/fig3.pdf`: present
- `paper/figures/gpt2_medium/fig4.pdf`: present
- `paper/figures/gpt2_medium/fig4a.pdf`: present
- `paper/figures/gpt2_medium/fig4b.pdf`: present
- `paper/figures/gpt2_medium/fig5.pdf`: present
- `paper/figures/gpt2_medium/fig_S1.pdf`: present
- `paper/figures/gpt2_medium/figS1_comparison.pdf`: present
- `paper/figures/gpt2_medium/fig_S2.pdf`: present
- `paper/figures/gpt2_medium/fig_appendix_probing.pdf`: present
- `paper/figures/gpt2_medium/fig6.pdf`: present

## Behavioral Statistics

- SFT forward: 100.00% 95% CI [100.0, 100.0], n=75
- SFT reverse: 64.00% 95% CI [53.3, 74.7], n=75
- ICL base k=4 reverse: 99.56% 95% CI [98.7, 100.0], n=75
- ICL SFT k=0 reverse: 64.00% 95% CI [53.3, 74.7], n=75
- ICL SFT k=1 reverse: 30.22% 95% CI [23.1, 37.3], n=75
- ICL SFT k=2 reverse: 83.11% 95% CI [76.9, 88.4], n=75
- ICL SFT k=4 reverse: 98.67% 95% CI [96.9, 100.0], n=75
- ICL SFT k=8 reverse: 100.00% 95% CI [100.0, 100.0], n=75
- ICL wrong-label k=4 reverse: 0.00% 95% CI [0.0, 0.0], n=75
- SFT forward vs reverse: McNemar exact p=1.49012e-08, d=0.745, n=75
- ICL k=4 reverse vs SFT reverse: Wilcoxon signed-rank p=4.54276e-07, d=0.724, n=75

