"""DLA-derived tables: rankings, layer summaries, and significance tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices
from src.mechanistic.sample_sizes import compute_patching_dla_sample_size


def benjamini_hochberg(p_values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _rank_biserial_from_wilcoxon(statistic: float, n: int, median_direction: float) -> float:
    total_rank = n * (n + 1) / 2
    magnitude = 1 - (2 * statistic / total_rank)
    # Positive means SFT > ICL, negative means ICL > SFT, matching manuscript convention.
    return float(np.sign(median_direction) * magnitude)


def _paired_layer_test(df: pd.DataFrame, component: str, output_path: Path) -> pd.DataFrame:
    records = []
    comp = df[df["component"] == component]
    for layer in sorted(comp["layer"].unique()):
        layer_df = comp[comp["layer"] == layer]
        if component == "attention":
            item_mode = (
                layer_df.groupby(["item_id", "mode"])["dla"]
                .sum()
                .unstack("mode")
                .dropna(subset=["sft_forward", "icl_reverse"])
            )
        else:
            item_mode = (
                layer_df[layer_df["head"] == -1]
                .pivot_table(index="item_id", columns="mode", values="dla", aggfunc="mean")
                .dropna(subset=["sft_forward", "icl_reverse"])
            )
        if item_mode.empty:
            continue
        diff = item_mode["sft_forward"] - item_mode["icl_reverse"]
        if np.allclose(diff.to_numpy(), 0):
            stat, p_raw = 0.0, 1.0
        else:
            stat, p_raw = wilcoxon(item_mode["sft_forward"], item_mode["icl_reverse"])
        records.append(
            {
                "layer": int(layer),
                "n": int(len(item_mode)),
                "wilcoxon_stat": float(stat),
                "p_raw": float(p_raw),
                "rank_biserial_r": _rank_biserial_from_wilcoxon(
                    float(stat), len(item_mode), float(np.median(diff))
                ),
                "sft_mean": float(item_mode["sft_forward"].mean()),
                "icl_mean": float(item_mode["icl_reverse"].mean()),
            }
        )
    out = pd.DataFrame(records)
    if not out.empty:
        out["p_corrected_BH"] = benjamini_hochberg(out["p_raw"])
        out["significant"] = out["p_corrected_BH"] < 0.05
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def make_dla_tables(
    dla_csv: Path,
    output_processed_dir: Path,
    output_tables_dir: Path,
    eligible_ids: list[str] | None = None,
) -> dict[str, Path]:
    df = pd.read_csv(dla_csv)
    df_for_subset_tests = df
    if eligible_ids is not None:
        df_for_subset_tests = df[df["item_id"].isin(eligible_ids)]

    output_processed_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    # Rankings are computed over all test items, matching the top-head
    # diagnostic reported in the manuscript. Subset tests below use only the
    # SFT-reverse-failure / ICL-success examples when eligible_ids is provided.
    att = df[df["component"] == "attention"]
    by_head = (
        att.groupby(["mode", "layer", "head"])["dla"]
        .mean()
        .reset_index()
        .sort_values(["mode", "dla"], ascending=[True, False])
    )
    by_head_path = output_processed_dir / "dla_by_head.csv"
    by_head.to_csv(by_head_path, index=False)

    mlp = df[df["component"] == "mlp"]
    by_mlp = (
        mlp.groupby(["mode", "layer"])["dla"]
        .mean()
        .reset_index()
        .sort_values(["mode", "layer"])
    )
    by_mlp_path = output_processed_dir / "dla_by_mlp.csv"
    by_mlp.to_csv(by_mlp_path, index=False)

    top_heads = (
        att[att["mode"] == "icl_reverse"]
        .groupby(["layer", "head"])["dla"]
        .mean()
        .reset_index()
        .sort_values("dla", ascending=False)
        .head(10)
    )
    top_heads.insert(0, "rank", range(1, len(top_heads) + 1))
    top_heads_path = output_tables_dir / "top_attention_heads.csv"
    top_heads.to_csv(top_heads_path, index=False)

    att_sig_path = output_tables_dir / "dla_significance_attention.csv"
    mlp_sig_path = output_tables_dir / "dla_significance_mlp.csv"
    subset_att = df_for_subset_tests[df_for_subset_tests["component"] == "attention"]
    layer_attention = (
        subset_att.groupby(["item_id", "mode", "layer"])["dla"]
        .sum()
        .reset_index()
        .groupby(["mode", "layer"])["dla"]
        .mean()
        .reset_index()
    )
    layer_attention_path = output_tables_dir / "dla_layer_attention.csv"
    layer_attention.to_csv(layer_attention_path, index=False)

    _paired_layer_test(df_for_subset_tests, "attention", att_sig_path)
    _paired_layer_test(df_for_subset_tests, "mlp", mlp_sig_path)

    return {
        "dla_by_head": by_head_path,
        "dla_by_mlp": by_mlp_path,
        "dla_layer_attention": layer_attention_path,
        "top_attention_heads": top_heads_path,
        "dla_significance_attention": att_sig_path,
        "dla_significance_mlp": mlp_sig_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--dla-csv", type=Path, default=None)
    parser.add_argument("--eligible-ids", nargs="*", default=None)
    parser.add_argument("--sft-eval-csv", type=Path, default=None)
    parser.add_argument("--icl-eval-csv", type=Path, default=None)
    parser.add_argument(
        "--use-all-items",
        action="store_true",
        help="Use all items for subset statistics instead of the default eligible subset.",
    )
    args = parser.parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    eligible_ids = args.eligible_ids
    if eligible_ids is None and not args.use_all_items:
        sft_eval = args.sft_eval_csv or (profile.tables_dir / "eval_sft_finetuned.csv")
        icl_eval = args.icl_eval_csv or (profile.tables_dir / "eval_icl_base_k4_rev.csv")
        if sft_eval.exists() and icl_eval.exists():
            eligible_ids = compute_patching_dla_sample_size(sft_eval, icl_eval)
    paths = make_dla_tables(
        args.dla_csv or (profile.processed_dir / "dla_all.csv"),
        profile.processed_dir,
        profile.tables_dir,
        eligible_ids,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
