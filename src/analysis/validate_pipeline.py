"""Validate pipeline wiring without executing model experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline_manifest import PIPELINE_ITEMS, artifacts_for_profile, resolve_artifact
from src.repro_config import ROOT, get_profile, ensure_profile_dirs, profile_choices


def script_paths(script_field: str) -> list[Path]:
    paths = []
    for part in script_field.split(";"):
        text = part.strip()
        if text.endswith(".py"):
            paths.append(ROOT / text)
    return paths


def validate(model_profile: str, require_outputs: bool = False) -> list[str]:
    profile = get_profile(model_profile)
    ensure_profile_dirs(profile)
    errors = []
    for item in PIPELINE_ITEMS:
        for script in script_paths(item.generating_script):
            if not script.exists():
                errors.append(f"Missing script for {item.requirement}: {script}")
        if require_outputs:
            for artifact in artifacts_for_profile(item, profile):
                rendered = artifact.format(
                    tables=profile.tables_dir,
                    processed=profile.processed_dir,
                    figures=profile.figures_dir,
                )
                path = resolve_artifact(rendered)
                if not path.exists():
                    errors.append(f"Missing output for {item.requirement}: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--require-outputs", action="store_true")
    args = parser.parse_args()
    errors = validate(args.model_profile, args.require_outputs)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Pipeline wiring valid for {args.model_profile}")


if __name__ == "__main__":
    main()
