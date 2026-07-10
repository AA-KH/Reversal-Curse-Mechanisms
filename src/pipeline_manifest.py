"""Paper-to-pipeline manifest for replication audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.repro_config import ROOT, ModelProfile, ensure_profile_dirs


@dataclass(frozen=True)
class PipelineItem:
    paper_section: str
    requirement: str
    generating_script: str
    primary_artifacts: list[str]
    notes: str
    primary_artifacts_by_profile: dict[str, list[str]] | None = None


PIPELINE_ITEMS = [
    PipelineItem(
        paper_section="Experimental Setup",
        requirement="SynthRel-500 generation and fixed 70/15/15 splits",
        generating_script="data/generate_entities.py",
        primary_artifacts=[
            "data/synthrel_500.json",
            "data/splits/train.jsonl",
            "data/splits/val.jsonl",
            "data/splits/test.jsonl",
        ],
        notes="Dataset is shared across model scales and must not be regenerated with different seeds.",
    ),
    PipelineItem(
        paper_section="Behavioral Results",
        requirement="Forward and reverse SFT exact-match generation accuracy",
        generating_script="src/sft/evaluate_sft.py",
        primary_artifacts=["{tables}/eval_sft_finetuned.csv"],
        notes="Use the fine-tuned checkpoint for the target model profile.",
    ),
    PipelineItem(
        paper_section="Behavioral Results",
        requirement="ICL reverse k in {0,1,2,4,8}, base k=4, and wrong-label control",
        generating_script="src/icl/evaluate_icl.py",
        primary_artifacts=[
            "{tables}/eval_icl_sft_k0_rev_wrongFalse.csv",
            "{tables}/eval_icl_sft_k1_rev_wrongFalse.csv",
            "{tables}/eval_icl_sft_k2_rev_wrongFalse.csv",
            "{tables}/eval_icl_sft_k4_rev_wrongFalse.csv",
            "{tables}/eval_icl_sft_k8_rev_wrongFalse.csv",
            "{tables}/eval_icl_sft_k4_rev_wrongTrue.csv",
            "{tables}/eval_icl_base_k4_rev.csv",
        ],
        notes="Same-relation demonstrations, three trials per item, fixed seed strategy.",
    ),
    PipelineItem(
        paper_section="Behavioral Results",
        requirement="Bootstrap CIs, McNemar, Wilcoxon, and effect sizes",
        generating_script="src/analysis/compute_statistics.py",
        primary_artifacts=["{tables}/stats_results.csv"],
        notes="Statistics are computed only from saved raw CSV outputs.",
    ),
    PipelineItem(
        paper_section="Mechanistic Analysis",
        requirement="Position-controlled activation patching",
        generating_script="src/run_mechanistic_pipeline.py",
        primary_artifacts=[
            "{processed}/position_matched_pairs.json",
            "{processed}/patching_results_position_controlled.csv",
        ],
        notes="Run only on eligible SFT-reverse-failure / ICL-success examples.",
    ),
    PipelineItem(
        paper_section="Mechanistic Analysis",
        requirement="Direct Logit Attribution, layer/head rankings, and DLA significance",
        generating_script="src/run_mechanistic_pipeline.py",
        primary_artifacts=[
            "{processed}/dla_all.csv",
            "{processed}/dla_by_head.csv",
            "{processed}/dla_by_mlp.csv",
            "{tables}/top_attention_heads.csv",
            "{tables}/dla_layer_attention.csv",
            "{tables}/dla_significance_attention.csv",
            "{tables}/dla_significance_mlp.csv",
        ],
        notes="Compute attention and MLP DLA on the final-token position.",
    ),
    PipelineItem(
        paper_section="Mechanistic Analysis",
        requirement="Attention head, single-layer MLP, and combined MLP ablations",
        generating_script="src/run_mechanistic_pipeline.py",
        primary_artifacts=[
            "{processed}/attention_ablation.csv",
            "{processed}/single_layer_mlp_ablation.csv",
            "{processed}/mlp_ablation.csv",
        ],
        notes="Ablation targets must be selected from target-model DLA outputs, not copied from Medium.",
    ),
    PipelineItem(
        paper_section="Behavioral Robustness",
        requirement="Dataset-size experiments for N in {35, 70, 140, 350}",
        generating_script="src/run_robustness_pipeline.py; src/analysis/table_generation.py",
        primary_artifacts=[
            "{tables}/eval_size_35.csv",
            "{tables}/eval_size_70.csv",
            "{tables}/eval_size_140.csv",
            "{tables}/eval_size_350.csv",
            "{tables}/size_ablation.csv",
        ],
        notes="Each size uses balanced per-relation subsets from the fixed training split.",
    ),
    PipelineItem(
        paper_section="Behavioral Robustness",
        requirement="Training-scale/checkpoint evaluations",
        generating_script="src/run_robustness_pipeline.py; src/analysis/table_generation.py",
        primary_artifacts=[
            "{tables}/eval_checkpoint-100.csv",
            "{tables}/eval_checkpoint-200.csv",
            "{tables}/eval_checkpoint-220.csv",
            "{tables}/eval_final_model.csv",
            "{tables}/epoch_ablation.csv",
        ],
        primary_artifacts_by_profile={
            "gpt2_small": [
                "{tables}/eval_checkpoint-100.csv",
                "{tables}/eval_final_model.csv",
                "{tables}/epoch_ablation.csv",
            ],
            "gpt2_medium": [
                "{tables}/eval_checkpoint-100.csv",
                "{tables}/eval_checkpoint-200.csv",
                "{tables}/eval_checkpoint-220.csv",
                "{tables}/eval_final_model.csv",
                "{tables}/epoch_ablation.csv",
            ],
            "gpt2_large": [
                "{tables}/eval_checkpoint-100.csv",
                "{tables}/eval_final_model.csv",
                "{tables}/epoch_ablation.csv",
            ],
        },
        notes="Evaluates saved checkpoints without changing the training protocol.",
    ),
    PipelineItem(
        paper_section="Robustness and Stability",
        requirement="Multi-seed SFT/ICL/DLA/ablation and head-overlap stability",
        generating_script="src/run_robustness_pipeline.py; src/analysis/stability.py",
        primary_artifacts=[
            "{tables}/eval_sft_finetuned_seed101.csv",
            "{tables}/eval_sft_finetuned_seed202.csv",
            "{tables}/eval_sft_finetuned_seed303.csv",
            "{tables}/eval_icl_sft_k4_rev_wrongFalse_seed101.csv",
            "{tables}/eval_icl_sft_k4_rev_wrongFalse_seed202.csv",
            "{tables}/eval_icl_sft_k4_rev_wrongFalse_seed303.csv",
            "{tables}/stability_summary.csv",
        ],
        notes="Seed-specific DLA tables and ablations are written under seed subdirectories.",
    ),
    PipelineItem(
        paper_section="Appendix",
        requirement="Induction-head prefix-matching diagnostic",
        generating_script="src/run_mechanistic_pipeline.py; src/mechanistic/induction_scores.py",
        primary_artifacts=["{tables}/induction_scores.csv"],
        notes="Scores the target model's top DLA heads using random repeated-token sequences.",
    ),
    PipelineItem(
        paper_section="Appendix",
        requirement="Probing classifier analysis",
        generating_script="src/run_mechanistic_pipeline.py; src/mechanistic/probing.py",
        primary_artifacts=[
            "{processed}/probing_sft.csv",
            "{processed}/probing_icl.csv",
            "{processed}/probing_by_layer.csv",
        ],
        notes="Kept for appendix reproducibility, while preserving the paper's caveat about interpretability.",
    ),
    PipelineItem(
        paper_section="Figures and Appendix",
        requirement="All manuscript figures and relation-type breakdowns",
        generating_script="src/analysis/plot_behavioral.py; src/mechanistic/plot_figures.py",
        primary_artifacts=[
            "{figures}/fig1.pdf",
            "{figures}/fig2.pdf",
            "{figures}/fig3.pdf",
            "{figures}/fig4.pdf",
            "{figures}/fig4a.pdf",
            "{figures}/fig4b.pdf",
            "{figures}/fig5.pdf",
            "{figures}/fig_S1.pdf",
            "{figures}/figS1_comparison.pdf",
            "{figures}/fig_S2.pdf",
            "{figures}/fig_appendix_probing.pdf",
            "{figures}/fig6.pdf",
        ],
        notes="Scale-replication figures are written under model-specific subdirectories.",
    ),
]


def repo_relative(path: str | Path) -> str:
    """Render repository-local paths without machine-specific prefixes."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    return candidate.relative_to(ROOT).as_posix()


def resolve_artifact(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def render_artifact(path_template: str, profile: ModelProfile) -> str:
    rendered = path_template.format(
        tables=profile.tables_dir,
        processed=profile.processed_dir,
        figures=profile.figures_dir,
    )
    return repo_relative(rendered)


def artifacts_for_profile(item: PipelineItem, profile: ModelProfile) -> list[str]:
    if item.primary_artifacts_by_profile and profile.slug in item.primary_artifacts_by_profile:
        return item.primary_artifacts_by_profile[profile.slug]
    return item.primary_artifacts


def item_payload(item: PipelineItem, profile: ModelProfile) -> dict:
    return {
        "paper_section": item.paper_section,
        "requirement": item.requirement,
        "generating_script": item.generating_script,
        "primary_artifacts": [
            render_artifact(path, profile) for path in artifacts_for_profile(item, profile)
        ],
        "notes": item.notes,
    }


def write_manifest(profile: ModelProfile) -> Path:
    ensure_profile_dirs(profile)
    payload = {
        "model": asdict(profile),
        "artifact_roots": {
            "experiments": repo_relative(profile.experiments_dir),
            "tables": repo_relative(profile.tables_dir),
            "processed": repo_relative(profile.processed_dir),
            "figures": repo_relative(profile.figures_dir),
            "reports": repo_relative(profile.report_dir),
        },
        "items": [item_payload(item, profile) for item in PIPELINE_ITEMS],
    }
    output = profile.report_dir / "pipeline_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output
