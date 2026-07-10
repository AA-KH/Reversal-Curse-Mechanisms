"""
Direct Logit Attribution using TransformerLens.
For each test example, compute per-head and per-MLP contribution
to the logit of the correct answer token.
"""

import transformer_lens
import torch
import pandas as pd
from tqdm import tqdm

def run_dla_analysis(
    model_path,     # path to SFT model OR "gpt2-medium" for base
    test_data,
    icl_prompts,    # pre-built ICL prompts for same test items
    output_prefix,
    tl_model_name="gpt2-medium",
    output_csv=None,
):
    # Load model into TransformerLens
    if model_path == tl_model_name:
        model = transformer_lens.HookedTransformer.from_pretrained(tl_model_name)
    else:
        # Load HuggingFace model then wrap with TransformerLens
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        model = transformer_lens.HookedTransformer.from_pretrained(
            tl_model_name,
            hf_model=hf_model,
        )
    model.eval()

    n_layers  = model.cfg.n_layers
    n_heads   = model.cfg.n_heads

    dla_records = []

    for idx, item in enumerate(tqdm(test_data, desc="DLA")):
        for mode in ["sft_forward", "sft_reverse", "icl_reverse"]:
            if mode == "sft_forward":
                prompt = f"Fact: {item['fact']}\nQ: {item['forward_question']}\nA:"
                expected = item["forward_answer"]
            elif mode == "sft_reverse":
                prompt = f"Fact: {item['fact']}\nQ: {item['reverse_question']}\nA:"
                expected = item["reverse_answer"]
            else:
                prompt = icl_prompts[idx]  # pre-built 4-shot reverse ICL prompt
                expected = item["reverse_answer"]

            input_ids = model.to_tokens(prompt)
            # DLA requires the exact token ID. If it tokenizes to multiple tokens, take the first.
            expected_tokens = model.to_tokens(" " + expected, prepend_bos=False)[0]
            answer_token_id = expected_tokens[0]

            # Run with cache
            _, cache = model.run_with_cache(input_ids)

            # Unembed direction for the correct answer token
            # W_U has shape [d_model, d_vocab]
            answer_dir = model.W_U[:, answer_token_id]

            # DLA per layer and component
            for layer in range(n_layers):
                # Attention contribution
                z = cache["z", layer]  # shape [batch, seq, heads, d_head]
                W_O = model.W_O[layer] # shape [heads, d_head, d_model]
                head_out = torch.einsum("h d, h d m -> h m", z[0, -1, :, :], W_O)
                attn_contribution = (head_out @ answer_dir).detach().cpu().numpy()

                # MLP contribution
                mlp_out = cache["mlp_out", layer]  # shape [batch, seq, d_model]
                mlp_contribution = (mlp_out[0, -1, :] @ answer_dir).detach().cpu().item()

                # Per head
                for head in range(n_heads):
                    dla_records.append({
                        "item_id": item["id"],
                        "mode": mode,
                        "layer": layer,
                        "head": head,
                        "component": "attention",
                        "dla": float(attn_contribution[head]),
                    })

                dla_records.append({
                    "item_id": item["id"],
                    "mode": mode,
                    "layer": layer,
                    "head": -1,
                    "component": "mlp",
                    "dla": mlp_contribution,
                })

    df = pd.DataFrame(dla_records)
    out_path = output_csv or f"{output_prefix}_dla.csv"
    df.to_csv(out_path, index=False)
    print(f"DLA saved: {out_path}")
    return df
