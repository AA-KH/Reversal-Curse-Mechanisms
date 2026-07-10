"""Prefix-matching induction-head diagnostic for selected attention heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def load_heads(heads_csv: Path) -> list[tuple[int, int]]:
    df = pd.read_csv(heads_csv)
    if not {"layer", "head"}.issubset(df.columns):
        raise ValueError(f"{heads_csv} must contain layer and head columns")
    return [(int(row.layer), int(row.head)) for row in df.itertuples(index=False)]


def compute_induction_scores(
    model,
    heads: list[tuple[int, int]],
    n_sequences: int = 128,
    prefix_len: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure attention to the token after the previous matching prefix token.

    Each sequence is `[prefix, prefix]`. For a query at position `prefix_len + i`,
    an induction head should attend to `i + 1`, the position following the
    previous occurrence of the same token. Scores average that attention mass.
    """

    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = model.cfg.d_vocab
    special = {getattr(model.tokenizer, "bos_token_id", None), getattr(model.tokenizer, "eos_token_id", None)}
    valid_tokens = torch.tensor([i for i in range(vocab_size) if i not in special and i is not None], dtype=torch.long)

    scores = {head: [] for head in heads}
    for _ in range(n_sequences):
        idx = torch.randint(0, len(valid_tokens), (prefix_len,), generator=generator)
        prefix = valid_tokens[idx]
        tokens = torch.cat([prefix, prefix], dim=0).unsqueeze(0).to(device)
        _, cache = model.run_with_cache(tokens)
        for layer, head in heads:
            pattern = cache["pattern", layer][0, head]
            vals = []
            for i in range(prefix_len - 1):
                query_pos = prefix_len + i
                key_pos = i + 1
                vals.append(pattern[query_pos, key_pos].detach().float().cpu())
            scores[(layer, head)].append(torch.stack(vals).mean().item())

    rows = []
    for rank, ((layer, head), vals) in enumerate(scores.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "head": head,
                "induction_score": float(pd.Series(vals).mean()),
                "n_sequences": n_sequences,
                "prefix_len": prefix_len,
                "passes_threshold_0_40": float(pd.Series(vals).mean()) > 0.40,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--heads-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--n-sequences", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    heads_csv = args.heads_csv or (profile.tables_dir / "top_attention_heads.csv")
    output = args.output or (profile.tables_dir / "induction_scores.csv")
    heads = load_heads(heads_csv)

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    if args.checkpoint_path:
        hf_model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
        model = HookedTransformer.from_pretrained(profile.tl_name, hf_model=hf_model, device=args.device)
    else:
        model = HookedTransformer.from_pretrained(profile.tl_name, device=args.device)

    df = compute_induction_scores(model, heads, args.n_sequences, args.prefix_len, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote induction scores to {output}")


if __name__ == "__main__":
    main()
