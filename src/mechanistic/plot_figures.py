"""Generate mechanistic figures from processed artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mechanistic.sample_sizes import compute_patching_dla_sample_size
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_mechanistic(processed_dir: Path, tables_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_patch = _require(processed_dir / "patching_results_position_controlled.csv")
    df_resid = df_patch[df_patch["component"] == "resid_pre"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_resid, x="layer", y="improvement", marker="o", color="green")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title("Layer-by-layer Patching Improvement (resid_pre, Position-Controlled)")
    plt.xlabel("Layer")
    plt.ylabel("Logit Improvement (Patched - Baseline)")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3.pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_patch, x="layer", y="improvement", hue="component", marker="o")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title("Full Layer-by-layer Patching Improvement (Position-Controlled)")
    plt.xlabel("Layer")
    plt.ylabel("Logit Improvement (Patched - Baseline)")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_S1.pdf")
    plt.close()

    raw_patch = processed_dir / "patching_results.csv"
    if raw_patch.exists():
        df_raw = pd.read_csv(raw_patch)
        raw_resid = df_raw[df_raw["component"] == "resid_pre"].assign(control="uncontrolled")
        controlled = df_resid.assign(control="position-controlled")
        compare = pd.concat([raw_resid, controlled], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=compare, x="layer", y="improvement", hue="control", marker="o")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title("Residual Patching Improvement Before/After Position Control")
        plt.xlabel("Layer")
        plt.ylabel("Logit Improvement")
        plt.tight_layout()
        plt.savefig(figures_dir / "figS1_comparison.pdf")
        plt.close()

    df_mlp = _require(processed_dir / "single_layer_mlp_ablation.csv")
    plt.figure(figsize=(10, 6))
    plt.plot(df_mlp["layer"], df_mlp["sft_forward_acc"], label="SFT Forward", marker="o")
    plt.plot(df_mlp["layer"], df_mlp["icl_reverse_acc"], label="ICL Reverse", marker="^")
    plt.title("Single-Layer MLP Ablation")
    plt.xlabel("Ablated Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5.pdf")
    plt.close()

    df_dla = _require(processed_dir / "dla_all.csv")
    eligible_ids = compute_patching_dla_sample_size(
        tables_dir / "eval_sft_finetuned.csv",
        tables_dir / "eval_icl_base_k4_rev.csv",
    )
    df_el = df_dla[df_dla["item_id"].isin(eligible_ids)]
    att_df = df_el[df_el["component"] == "attention"]
    mean_att = att_df.groupby(["mode", "layer", "head"])["dla"].mean().reset_index()
    sft = mean_att[mean_att["mode"] == "sft_forward"].set_index(["layer", "head"])["dla"]
    icl = mean_att[mean_att["mode"] == "icl_reverse"].set_index(["layer", "head"])["dla"]
    diff = (icl - sft).reset_index()
    pivot = diff.pivot(index="layer", columns="head", values="dla")

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
    plt.title(f"Attention DLA Difference (ICL Reverse - SFT Forward), N={len(eligible_ids)}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4a.pdf")
    plt.close()

    mlp_df = df_el[df_el["component"] == "mlp"]
    mean_mlp = mlp_df.groupby(["mode", "layer"])["dla"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mean_mlp, x="layer", y="dla", hue="mode", marker="o")
    plt.title(f"MLP DLA by Layer, N={len(eligible_ids)}")
    plt.xlabel("Layer")
    plt.ylabel("Mean DLA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4b.pdf")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 15))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False, ax=axes[0])
    axes[0].invert_yaxis()
    axes[0].set_title(f"Attention DLA Difference (ICL Reverse - SFT Forward), N={len(eligible_ids)}")
    sns.lineplot(data=mean_mlp, x="layer", y="dla", hue="mode", marker="o", ax=axes[1])
    axes[1].set_title(f"MLP DLA by Layer, N={len(eligible_ids)}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4.pdf")
    plt.close()

    probing_path = processed_dir / "probing_by_layer.csv"
    if probing_path.exists():
        probing = pd.read_csv(probing_path)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=probing, x="layer", y="mean_accuracy", hue="mode", marker="o")
        plt.title("Probing Classifier Accuracy by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Training Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_appendix_probing.pdf")
        plt.savefig(figures_dir / "fig6.pdf")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    plot_mechanistic(
        args.processed_dir or profile.processed_dir,
        args.tables_dir or profile.tables_dir,
        args.figures_dir or profile.figures_dir,
    )
    print(f"Wrote mechanistic figures to {args.figures_dir or profile.figures_dir}")


if __name__ == "__main__":
    main()
