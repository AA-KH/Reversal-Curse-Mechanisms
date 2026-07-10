"""Compute behavioral statistics from saved evaluation artifacts."""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline_manifest import repo_relative
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path)


def bootstrap_ci(values: pd.Series | np.ndarray, n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5]) * 100
    return float(lo), float(hi)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    std = diff.std(ddof=1)
    return float("nan") if std == 0 else float(diff.mean() / std)


def exact_mcnemar_p(discordant_successes: int, discordant_total: int) -> float:
    if discordant_total == 0:
        return 1.0
    k = min(discordant_successes, discordant_total - discordant_successes)
    tail = sum(comb(discordant_total, i) for i in range(k + 1)) / (2**discordant_total)
    return min(1.0, 2 * tail)


def add_accuracy_row(rows: list[dict], label: str, values: pd.Series, source: Path) -> None:
    lo, hi = bootstrap_ci(values)
    rows.append(
        {
            "analysis": "accuracy",
            "label": label,
            "value": values.mean() * 100,
            "ci_low": lo,
            "ci_high": hi,
            "n": len(values),
            "source": repo_relative(source),
        }
    )


def compute_behavioral_statistics(tables_dir: Path, output_path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    sft_path = tables_dir / "eval_sft_finetuned.csv"
    sft = read_csv(sft_path)
    assert sft is not None
    add_accuracy_row(rows, "SFT forward", sft[sft["direction"] == "forward"]["correct"], sft_path)
    add_accuracy_row(rows, "SFT reverse", sft[sft["direction"] == "reverse"]["correct"], sft_path)

    optional_specs = [
        ("ICL base k=4 reverse", "eval_icl_base_k4_rev.csv"),
        ("ICL SFT k=0 reverse", "eval_icl_sft_k0_rev_wrongFalse.csv"),
        ("ICL SFT k=1 reverse", "eval_icl_sft_k1_rev_wrongFalse.csv"),
        ("ICL SFT k=2 reverse", "eval_icl_sft_k2_rev_wrongFalse.csv"),
        ("ICL SFT k=4 reverse", "eval_icl_sft_k4_rev_wrongFalse.csv"),
        ("ICL SFT k=8 reverse", "eval_icl_sft_k8_rev_wrongFalse.csv"),
        ("ICL wrong-label k=4 reverse", "eval_icl_sft_k4_rev_wrongTrue.csv"),
    ]
    for label, filename in optional_specs:
        path = tables_dir / filename
        df = read_csv(path, required=False)
        if df is not None:
            add_accuracy_row(rows, label, df["mean_correct"], path)

    fwd = sft[sft["direction"] == "forward"].set_index("id")["correct"].sort_index()
    rev = sft[sft["direction"] == "reverse"].set_index("id")["correct"].sort_index()
    if fwd.index.equals(rev.index):
        fwd_only = int(((fwd == 1) & (rev == 0)).sum())
        rev_only = int(((fwd == 0) & (rev == 1)).sum())
        rows.append(
            {
                "analysis": "paired_test",
                "label": "SFT forward vs reverse",
                "value": exact_mcnemar_p(fwd_only, fwd_only + rev_only),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": len(fwd),
                "effect_size": cohens_d_paired(fwd.to_numpy(), rev.to_numpy()),
                "test": "McNemar exact",
                "source": repo_relative(sft_path),
            }
        )

    icl_k4_path = tables_dir / "eval_icl_sft_k4_rev_wrongFalse.csv"
    icl_k4 = read_csv(icl_k4_path, required=False)
    if icl_k4 is not None and fwd.index.equals(rev.index):
        icl = icl_k4.set_index("id")["mean_correct"].sort_index()
        sft_rev = rev.loc[icl.index]
        test = wilcoxon(icl.to_numpy(), sft_rev.to_numpy())
        rows.append(
            {
                "analysis": "paired_test",
                "label": "ICL k=4 reverse vs SFT reverse",
                "value": test.pvalue,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": len(icl),
                "effect_size": cohens_d_paired(icl.to_numpy(), sft_rev.to_numpy()),
                "test": "Wilcoxon signed-rank",
                "source": f"{repo_relative(icl_k4_path)}; {repo_relative(sft_path)}",
            }
        )

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    tables_dir = args.tables_dir or profile.tables_dir
    output = args.output or (tables_dir / "stats_results.csv")
    df = compute_behavioral_statistics(tables_dir, output)
    print(f"Wrote {len(df)} statistics rows to {output}")


if __name__ == "__main__":
    main()
