"""Generate derived behavioral tables from raw evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline_manifest import repo_relative
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def generate_accuracy_by_condition(tables_dir: Path) -> Path:
    rows = []
    sft = _require(tables_dir / "eval_sft_finetuned.csv")
    for direction in ["forward", "reverse"]:
        vals = sft[sft["direction"] == direction]["correct"]
        rows.append(
            {
                "condition": f"sft_{direction}",
                "accuracy": vals.mean(),
                "n": len(vals),
                "source": repo_relative(tables_dir / "eval_sft_finetuned.csv"),
            }
        )
    for k in [0, 1, 2, 4, 8]:
        path = tables_dir / f"eval_icl_sft_k{k}_rev_wrongFalse.csv"
        df = _require(path)
        rows.append({"condition": f"icl_sft_reverse_k{k}", "accuracy": df["mean_correct"].mean(), "n": len(df), "source": repo_relative(path)})
    wrong = tables_dir / "eval_icl_sft_k4_rev_wrongTrue.csv"
    df_wrong = _require(wrong)
    rows.append({"condition": "icl_sft_reverse_k4_wrong_label", "accuracy": df_wrong["mean_correct"].mean(), "n": len(df_wrong), "source": repo_relative(wrong)})
    base = tables_dir / "eval_icl_base_k4_rev.csv"
    if base.exists():
        df_base = pd.read_csv(base)
        rows.append({"condition": "icl_base_reverse_k4", "accuracy": df_base["mean_correct"].mean(), "n": len(df_base), "source": repo_relative(base)})
    output = tables_dir / "accuracy_by_condition.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def generate_accuracy_by_relation(tables_dir: Path) -> Path:
    sft = _require(tables_dir / "eval_sft_finetuned.csv")
    icl = _require(tables_dir / "eval_icl_sft_k4_rev_wrongFalse.csv").copy()
    icl["correct"] = icl["mean_correct"]
    sft_fwd = sft[sft["direction"] == "forward"].groupby("relation_id")["correct"].mean().reset_index(name="sft_forward_acc")
    sft_rev = sft[sft["direction"] == "reverse"].groupby("relation_id")["correct"].mean().reset_index(name="sft_reverse_acc")
    icl_rev = icl.groupby("relation_id")["correct"].mean().reset_index(name="icl_reverse_acc")
    output = tables_dir / "accuracy_by_relation.csv"
    sft_fwd.merge(sft_rev, on="relation_id").merge(icl_rev, on="relation_id").to_csv(output, index=False)
    return output


def generate_size_ablation(tables_dir: Path, sizes: list[int]) -> Path:
    rows = []
    for size in sizes:
        path = tables_dir / f"eval_size_{size}.csv"
        df = _require(path)
        rows.append(
            {
                "train_size": size,
                "forward_acc": df[df["direction"] == "forward"]["correct"].mean(),
                "reverse_acc": df[df["direction"] == "reverse"]["correct"].mean(),
                "source": repo_relative(path),
            }
        )
    output = tables_dir / "size_ablation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def generate_epoch_ablation(tables_dir: Path, checkpoint_names: list[str]) -> Path:
    rows = []
    for name in checkpoint_names:
        path = tables_dir / f"eval_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows.append(
            {
                "checkpoint": name,
                "forward_acc": df[df["direction"] == "forward"]["correct"].mean(),
                "reverse_acc": df[df["direction"] == "reverse"]["correct"].mean(),
                "source": repo_relative(path),
            }
        )
    output = tables_dir / "epoch_ablation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--table", action="append", choices=["condition", "relation", "size", "epoch", "all"], default=None)
    parser.add_argument("--sizes", nargs="*", type=int, default=[35, 70, 140, 350])
    parser.add_argument("--checkpoints", nargs="*", default=None)
    args = parser.parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    tables = args.table or ["all"]
    if "all" in tables:
        tables = ["condition", "relation", "size", "epoch"]
    outputs = []
    if "condition" in tables:
        outputs.append(generate_accuracy_by_condition(profile.tables_dir))
    if "relation" in tables:
        outputs.append(generate_accuracy_by_relation(profile.tables_dir))
    if "size" in tables:
        outputs.append(generate_size_ablation(profile.tables_dir, args.sizes))
    if "epoch" in tables:
        outputs.append(generate_epoch_ablation(profile.tables_dir, args.checkpoints or list(profile.checkpoint_names)))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
