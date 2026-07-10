"""Launch the model-scale replication pipeline with explicit artifact paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline_manifest import write_manifest
from src.repro_config import ROOT, get_profile, ensure_profile_dirs, profile_choices


def run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def latest_final_model(profile_slug: str) -> Path:
    profile = get_profile(profile_slug)
    candidates = sorted(profile.experiments_dir.glob("run_*/final_model"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No final_model checkpoint found under {profile.experiments_dir}. Run the train stage first."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument(
        "--stage",
        action="append",
        choices=[
            "manifest",
            "dataset",
            "train",
            "eval-sft",
            "eval-icl",
            "mechanistic",
            "robustness",
            "stats",
            "figures",
            "report",
            "all",
        ],
        default=None,
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    stages = args.stage or ["manifest"]
    if "all" in stages:
        stages = [
            "manifest",
            "dataset",
            "train",
            "eval-sft",
            "eval-icl",
            "mechanistic",
            "robustness",
            "stats",
            "figures",
            "report",
        ]

    py = sys.executable
    device_args = ["--device", args.device] if args.device else []

    if "manifest" in stages:
        path = write_manifest(profile)
        print(f"Wrote manifest to {path}")

    if "dataset" in stages:
        run([py, "data/generate_entities.py"], args.dry_run)

    if "train" in stages:
        run(
            [
                py,
                "src/sft/train.py",
                "--model-profile",
                profile.slug,
                "--seed",
                str(args.seed),
                *device_args,
            ],
            args.dry_run,
        )

    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else None
    if (
        "eval-sft" in stages
        or "eval-icl" in stages
        or "mechanistic" in stages
        or "robustness" in stages
    ) and checkpoint is None:
        if args.dry_run:
            checkpoint = profile.experiments_dir / f"run_<timestamp>_seed{args.seed}" / "final_model"
        else:
            checkpoint = latest_final_model(profile.slug)

    if "eval-sft" in stages:
        assert checkpoint is not None
        run(
            [
                py,
                "src/sft/evaluate_sft.py",
                "--model-profile",
                profile.slug,
                "--checkpoint-path",
                str(checkpoint),
                *device_args,
            ],
            args.dry_run,
        )

    if "eval-icl" in stages:
        assert checkpoint is not None
        run(
            [
                py,
                "src/icl/evaluate_icl.py",
                "--model-profile",
                profile.slug,
                "--model-path",
                str(checkpoint),
                *device_args,
            ],
            args.dry_run,
        )
        run(
            [
                py,
                "src/icl/evaluate_icl.py",
                "--model-profile",
                profile.slug,
                "--model-path",
                str(checkpoint),
                "--k",
                "4",
                "--wrong-labels",
                *device_args,
            ],
            args.dry_run,
        )
        run(
            [
                py,
                "src/icl/evaluate_icl.py",
                "--model-profile",
                profile.slug,
                "--k",
                "4",
                *device_args,
            ],
            args.dry_run,
        )

    if "mechanistic" in stages:
        assert checkpoint is not None
        run(
            [
                py,
                "src/run_mechanistic_pipeline.py",
                "--model-profile",
                profile.slug,
                "--checkpoint-path",
                str(checkpoint),
                *(["--device", args.device] if args.device else []),
            ],
            args.dry_run,
        )

    if "robustness" in stages:
        assert checkpoint is not None
        run(
            [
                py,
                "src/run_robustness_pipeline.py",
                "--model-profile",
                profile.slug,
                "--stage",
                "all",
                "--primary-checkpoint",
                str(checkpoint),
                *(["--device", args.device] if args.device else []),
            ],
            args.dry_run,
        )

    if "stats" in stages:
        run([py, "src/analysis/compute_statistics.py", "--model-profile", profile.slug], args.dry_run)
        run([py, "src/analysis/table_generation.py", "--model-profile", profile.slug], args.dry_run)

    if "figures" in stages:
        run([py, "src/analysis/plot_behavioral.py", "--model-profile", profile.slug], args.dry_run)
        run([py, "src/mechanistic/plot_figures.py", "--model-profile", profile.slug], args.dry_run)

    if "report" in stages:
        run([py, "src/analysis/generate_replication_report.py", "--model-profile", profile.slug], args.dry_run)


if __name__ == "__main__":
    main()
