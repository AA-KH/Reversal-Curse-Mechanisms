"""Multi-seed head-overlap and ablation stability summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def read_top_heads(path: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(path)
    return set(zip(df["layer"].astype(int), df["head"].astype(int)))


def summarize_stability(tables_dir: Path, processed_dir: Path, seeds: list[int], output: Path) -> pd.DataFrame:
    primary_path = tables_dir / "top_attention_heads.csv"
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    primary = read_top_heads(primary_path)
    rows = []
    for seed in seeds:
        seed_head_path = tables_dir / f"seed{seed}" / "top_attention_heads.csv"
        seed_heads = read_top_heads(seed_head_path)
        ablation_path = processed_dir / f"seed{seed}" / "attention_ablation.csv"
        ablated_acc = None
        sft_ablated_acc = None
        if ablation_path.exists():
            ablation = pd.read_csv(ablation_path).iloc[0]
            ablated_acc = float(ablation.get("icl_reverse_acc_ablated"))
            sft_ablated_acc = float(ablation.get("sft_forward_acc_ablated"))
        rows.append(
            {
                "seed": seed,
                "top_head_overlap_with_primary": len(primary & seed_heads),
                "primary_top_head_count": len(primary),
                "seed_top_head_count": len(seed_heads),
                "icl_reverse_acc_ablated": ablated_acc,
                "sft_forward_acc_ablated": sft_ablated_acc,
            }
        )
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    output = args.output or (profile.tables_dir / "stability_summary.csv")
    summarize_stability(profile.tables_dir, profile.processed_dir, args.seeds, output)
    print(f"Wrote stability summary to {output}")


if __name__ == "__main__":
    main()
