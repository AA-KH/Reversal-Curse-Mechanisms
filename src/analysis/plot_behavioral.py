"""Generate behavioral figures from saved evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return pd.read_csv(path)


def plot_behavioral(tables_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sft = require(tables_dir / "eval_sft_finetuned.csv")
    sft_fwd = sft[sft["direction"] == "forward"]["correct"].mean() * 100
    sft_rev = sft[sft["direction"] == "reverse"]["correct"].mean() * 100

    icl = {}
    for k in [0, 1, 2, 4, 8]:
        icl[k] = require(tables_dir / f"eval_icl_sft_k{k}_rev_wrongFalse.csv")["mean_correct"].mean() * 100
    wrong = require(tables_dir / "eval_icl_sft_k4_rev_wrongTrue.csv")["mean_correct"].mean() * 100

    labels = [
        "SFT Forward",
        "SFT Reverse",
        "ICL Rev k=0",
        "ICL Rev k=1",
        "ICL Rev k=2",
        "ICL Rev k=4",
        "ICL Rev k=8",
        "Wrong Labels\n(k=4)",
    ]
    values = [sft_fwd, sft_rev, icl[0], icl[1], icl[2], icl[4], icl[8], wrong]
    colors = ["#2196F3", "#2196F3"] + ["#FF9800"] * 5 + ["#9E9E9E"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Condition - SynthRel-500 Test Set")
    ax.axhline(100, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color="#2196F3", label="SFT"),
            mpatches.Patch(color="#FF9800", label="ICL"),
            mpatches.Patch(color="#9E9E9E", label="Wrong-label control"),
        ],
        fontsize=10,
        framealpha=0.3,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(figures_dir / "fig1.png", dpi=150, bbox_inches="tight")
    plt.close()

    k_vals = [0, 1, 2, 4, 8]
    acc_vals = [icl[k] for k in k_vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_vals,
        acc_vals,
        marker="o",
        color="#FF9800",
        linewidth=2.5,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.5,
        label="ICL Reverse Accuracy",
    )
    ax.axhline(
        sft_rev,
        color="#2196F3",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"SFT Reverse baseline ({sft_rev:.1f}%)",
    )
    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of Demonstrations (k)")
    ax.set_ylabel("ICL Reverse Accuracy (%)")
    ax.set_title("k-Shot ICL Reverse Accuracy")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(figures_dir / "fig2.png", dpi=150, bbox_inches="tight")
    plt.close()

    relation_path = tables_dir / "accuracy_by_relation.csv"
    if relation_path.exists():
        rel = pd.read_csv(relation_path)
    else:
        sft_rel = sft[sft["direction"] == "forward"].groupby("relation_id")["correct"].mean().reset_index(name="sft_forward_acc")
        sft_rev_rel = sft[sft["direction"] == "reverse"].groupby("relation_id")["correct"].mean().reset_index(name="sft_reverse_acc")
        icl_rel = (
            require(tables_dir / "eval_icl_sft_k4_rev_wrongFalse.csv")
            .groupby("relation_id")["mean_correct"]
            .mean()
            .reset_index(name="icl_reverse_acc")
        )
        rel = sft_rel.merge(sft_rev_rel, on="relation_id").merge(icl_rel, on="relation_id")
    rel_melt = rel.melt(
        id_vars="relation_id",
        value_vars=["sft_forward_acc", "sft_reverse_acc", "icl_reverse_acc"],
        var_name="Condition",
        value_name="Accuracy",
    )
    rel_melt["Accuracy"] = rel_melt["Accuracy"] * 100
    label_map = {
        "sft_forward_acc": "SFT Forward",
        "sft_reverse_acc": "SFT Reverse",
        "icl_reverse_acc": "ICL Reverse k=4",
    }
    rel_melt["Condition"] = rel_melt["Condition"].map(label_map)
    plt.figure(figsize=(10, 6))
    import seaborn as sns

    sns.barplot(data=rel_melt, x="relation_id", y="Accuracy", hue="Condition")
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Relation Type")
    plt.title("Accuracy by Relation Type")
    plt.legend(framealpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_S2.pdf", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    plot_behavioral(args.tables_dir or profile.tables_dir, args.figures_dir or profile.figures_dir)
    print(f"Wrote behavioral figures to {args.figures_dir or profile.figures_dir}")


if __name__ == "__main__":
    main()
