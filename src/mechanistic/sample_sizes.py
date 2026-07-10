from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

def compute_patching_dla_sample_size(sft_eval_csv, icl_eval_csv):
    sft_df = pd.read_csv(sft_eval_csv)
    icl_df = pd.read_csv(icl_eval_csv)

    sft_reverse = sft_df[sft_df["direction"] == "reverse"].set_index("id")["correct"]
    icl_reverse = icl_df.set_index("id")["mean_correct"] >= 0.5

    common_ids = sft_reverse.index.intersection(icl_reverse.index)
    eligible = [
        i for i in common_ids
        if sft_reverse[i] == 0 and icl_reverse[i] == True
    ]
    print(f"Items eligible for patching (SFT reverse wrong, ICL reverse correct): {len(eligible)}")
    print(f"Out of {len(common_ids)} total test items")
    return eligible

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--sft-eval-csv", type=Path, default=None)
    parser.add_argument("--icl-eval-csv", type=Path, default=None)
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    compute_patching_dla_sample_size(
        args.sft_eval_csv or (profile.tables_dir / "eval_sft_finetuned.csv"),
        args.icl_eval_csv or (profile.tables_dir / "eval_icl_base_k4_rev.csv"),
    )
