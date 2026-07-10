"""Orchestrate dataset-size, checkpoint-scale, and multi-seed experiments.

Mechanistic analysis scope for robustness seeds
-------------------------------------------------
For each robustness seed (101, 202, 303), this pipeline intentionally runs only
a subset of the full mechanistic analysis:

  - DLA (direct logit attribution): run for each seed
  - Attention ablations: run for each seed
  - Single-layer MLP ablation: run for each seed

The following are intentionally NOT run for robustness seeds:

  - Activation patching (run_mechanistic_pipeline --stage patching)
  - Probing (run_mechanistic_pipeline --stage probing)
  - Induction scoring (run_mechanistic_pipeline --stage induction)

Rationale: The robustness analysis is scoped to verifying that the top
attention heads and DLA structure replicate across seeds, summarised in
stability_summary.csv. The complete mechanistic characterisation (including
patching, probing, and induction) is provided for the primary seed (seed=42)
only. This is a deliberate design decision, not an omission.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.repro_config import ROOT, get_profile, ensure_profile_dirs, profile_choices


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train_command(py: str, profile: str, seed: int, train_path: str, output_dir: Path, device: str | None) -> list[str]:
    cmd = [
        py,
        "src/sft/train.py",
        "--model-profile",
        profile,
        "--seed",
        str(seed),
        "--train-path",
        train_path,
        "--output-dir",
        str(output_dir),
    ]
    if device:
        cmd.extend(["--device", device])
    return cmd


def eval_sft_command(py: str, profile: str, checkpoint: Path, output: Path, device: str | None) -> list[str]:
    cmd = [
        py,
        "src/sft/evaluate_sft.py",
        "--model-profile",
        profile,
        "--checkpoint-path",
        str(checkpoint),
        "--output-csv",
        str(output),
    ]
    if device:
        cmd.extend(["--device", device])
    return cmd


def eval_icl_command(py: str, profile: str, checkpoint: Path, output_dir: Path, seed: int, device: str | None) -> list[str]:
    cmd = [
        py,
        "src/icl/evaluate_icl.py",
        "--model-profile",
        profile,
        "--model-path",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--k",
        "4",
        "--seed",
        str(seed),
        "--filename-suffix",
        f"_seed{seed}",
    ]
    if device:
        cmd.extend(["--device", device])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--stage", action="append", choices=["size", "checkpoint", "multiseed", "all"], default=None)
    parser.add_argument("--primary-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--sizes", nargs="*", type=int, default=[35, 70, 140, 350])
    parser.add_argument("--checkpoint-names", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303])
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["size", "checkpoint", "multiseed"]
    py = sys.executable

    if "size" in stages:
        for size in args.sizes:
            output_dir = profile.experiments_dir / f"size_{size}_seed42"
            checkpoint = output_dir / "final_model"
            if not checkpoint.exists():
                run(train_command(py, profile.slug, 42, f"data/splits/train_{size}.jsonl", output_dir, args.device), args.dry_run)
            run(eval_sft_command(py, profile.slug, checkpoint, profile.tables_dir / f"eval_size_{size}.csv", args.device), args.dry_run)

    if "checkpoint" in stages:
        root = args.checkpoint_root or (args.primary_checkpoint.parent if args.primary_checkpoint else None)
        if root is None:
            raise ValueError("--checkpoint-root or --primary-checkpoint is required for checkpoint stage")
        for name in args.checkpoint_names or list(profile.checkpoint_names):
            checkpoint = root / name
            if checkpoint.exists():
                run(eval_sft_command(py, profile.slug, checkpoint, profile.tables_dir / f"eval_{name}.csv", args.device), args.dry_run)

    if "multiseed" in stages:
        for seed in args.seeds:
            output_dir = profile.experiments_dir / f"seed_{seed}"
            checkpoint = output_dir / "final_model"
            if not checkpoint.exists():
                run(train_command(py, profile.slug, seed, "data/splits/train.jsonl", output_dir, args.device), args.dry_run)
            run(eval_sft_command(py, profile.slug, checkpoint, profile.tables_dir / f"eval_sft_finetuned_seed{seed}.csv", args.device), args.dry_run)
            run(eval_icl_command(py, profile.slug, checkpoint, profile.tables_dir, seed, args.device), args.dry_run)
            seed_processed = profile.processed_dir / f"seed{seed}"
            seed_tables = profile.tables_dir / f"seed{seed}"
            seed_processed.mkdir(parents=True, exist_ok=True)
            seed_tables.mkdir(parents=True, exist_ok=True)
            run(
                [
                    py,
                    "src/run_mechanistic_pipeline.py",
                    "--model-profile",
                    profile.slug,
                    "--checkpoint-path",
                    str(checkpoint),
                    "--stage",
                    "prompts",
                    "--stage",
                    "dla",
                    "--stage",
                    "ablations",
                    "--processed-dir",
                    str(seed_processed),
                    "--tables-dir",
                    str(seed_tables),
                    "--sft-eval-csv",
                    str(profile.tables_dir / f"eval_sft_finetuned_seed{seed}.csv"),
                    "--icl-eval-csv",
                    str(profile.tables_dir / "eval_icl_base_k4_rev.csv"),
                    "--seed",
                    str(seed),
                    *(["--device", args.device] if args.device else []),
                ],
                args.dry_run,
            )
        run(
            [
                py,
                "src/analysis/stability.py",
                "--model-profile",
                profile.slug,
                "--seeds",
                *[str(seed) for seed in args.seeds],
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
