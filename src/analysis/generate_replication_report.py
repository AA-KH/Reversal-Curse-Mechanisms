"""Generate an honest replication status report from local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline_manifest import PIPELINE_ITEMS, artifacts_for_profile, render_artifact, resolve_artifact, write_manifest
from src.pipeline_manifest import repo_relative
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def summarize_csv(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - report should survive bad artifacts
        return f"unreadable: {exc}"
    return f"present, {df.shape[0]} rows x {df.shape[1]} columns"


def generate_report(model_profile: str) -> Path:
    profile = get_profile(model_profile)
    ensure_profile_dirs(profile)
    manifest_path = write_manifest(profile)
    output = profile.report_dir / "replication_status.md"

    lines = [
        f"# {profile.label} Replication Status",
        "",
        "This report is generated only from local artifacts. Missing artifacts are reported as missing rather than inferred.",
        "",
        f"- Model profile: `{profile.slug}`",
        f"- HuggingFace model: `{profile.hf_name}`",
        f"- TransformerLens model: `{profile.tl_name}`",
        f"- Manifest: `{repo_relative(manifest_path)}`",
        "",
        "## Artifact Inventory",
        "",
    ]

    for item in PIPELINE_ITEMS:
        lines.append(f"### {item.requirement}")
        lines.append(f"- Paper section: {item.paper_section}")
        lines.append(f"- Generating script: `{item.generating_script}`")
        lines.append(f"- Notes: {item.notes}")
        for artifact in artifacts_for_profile(item, profile):
            rendered = render_artifact(artifact, profile)
            path = resolve_artifact(rendered)
            status = summarize_csv(path) if path.suffix == ".csv" else ("present" if path.exists() else "missing")
            lines.append(f"- `{rendered}`: {status}")
        lines.append("")

    stats_path = profile.tables_dir / "stats_results.csv"
    if stats_path.exists():
        lines.extend(["## Behavioral Statistics", ""])
        stats = pd.read_csv(stats_path)
        for row in stats.itertuples(index=False):
            if getattr(row, "analysis") == "accuracy":
                lines.append(
                    f"- {row.label}: {row.value:.2f}% "
                    f"95% CI [{row.ci_low:.1f}, {row.ci_high:.1f}], n={int(row.n)}"
                )
            else:
                effect = getattr(row, "effect_size", float("nan"))
                test = getattr(row, "test", "test")
                lines.append(
                    f"- {row.label}: {test} p={row.value:.6g}, d={effect:.3f}, n={int(row.n)}"
                )
        lines.append("")

    lines.extend([
        "## Implementation Notes",
        "",
        "### Probing Classifier (Appendix)",
        "",
        "The probing analysis (`src/mechanistic/probing.py`) trains a logistic regression classifier",
        "at each layer and reports **training accuracy** (not held-out test accuracy). The paper appendix",
        "(Appendix~B) explicitly documents this and explains why it is not informative: with 75 unique",
        "entity-A classes, one sample per class, and a hidden-state dimensionality of 768 (GPT-2 Small)",
        "or 1024 (GPT-2 Medium), the VC dimension of the linear classifier vastly exceeds the number of",
        "examples, guaranteeing a perfect separator regardless of representational structure. The 100%",
        "training accuracy result at every layer is therefore a consequence of high-dimensional geometry,",
        "not a meaningful finding. This result is excluded from primary results. The implementation is",
        "intentionally unchanged pending a future redesign using larger datasets or geometry-based measures.",
        "",
        "### Robustness Seed Mechanistic Coverage",
        "",
        "For robustness seeds (101, 202, 303), `src/run_robustness_pipeline.py` intentionally runs only",
        "a subset of mechanistic analyses: DLA (direct logit attribution) and attention ablations.",
        "Activation patching, probing, and induction scoring are **not** run for robustness seeds.",
        "This is by design: the robustness analysis is scoped to verifying that the top attention heads",
        "and DLA structure replicate across seeds (captured by `stability_summary.csv`), not to fully",
        "replicating the entire mechanistic pipeline for each seed. The primary seed (seed=42) provides",
        "the complete mechanistic characterization.",
        "",
        "### Duplicate Evaluation Files",
        "",
        "`eval_sft_finetuned.csv` and `eval_final_model.csv` evaluate the same checkpoint and produce",
        "identical results. Both are required: `eval_sft_finetuned.csv` is the primary artifact consumed",
        "by all downstream scripts (statistics, tables, figures, mechanistic eligibility filtering);",
        "`eval_final_model.csv` is the epoch-ablation artifact for the final training step, generated",
        "independently by the robustness pipeline. Consolidating them would require changing downstream",
        "script defaults and is not done here.",
        "",
        "### Checkpoint-200 and Checkpoint-220",
        "",
        "GPT-2 Medium uses checkpoint-200 and checkpoint-220 for its epoch ablation because its larger",
        "training dataset produces more than 220 optimizer steps (350 examples / effective batch 16 × 5",
        "epochs = 110 steps for GPT-2 Small, but GPT-2 Medium trains on a larger corpus yielding ~220+",
        "steps). These checkpoints are **not applicable to GPT-2 Small**: with 350 training examples,",
        "batch=4, gradient accumulation=4, and 5 epochs, GPT-2 Small produces exactly 110 optimizer",
        "steps, making checkpoint-200 and checkpoint-220 mathematically impossible. The pipeline manifest",
        "correctly handles this via `primary_artifacts_by_profile` (see `src/pipeline_manifest.py`).",
        "",
    ])

    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    args = parser.parse_args()
    output = generate_report(args.model_profile)
    print(f"Wrote replication status report to {output}")


if __name__ == "__main__":
    main()
