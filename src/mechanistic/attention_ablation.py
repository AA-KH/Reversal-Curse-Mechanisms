"""
Attention Head Ablation Analysis
Identify top DLA heads and ablate them to measure causal impact on ICL vs SFT.
"""

import pandas as pd
from tqdm import tqdm


def _top_icl_heads(dla_csv_path):
    # Load DLA results to find top 10 heads
    try:
        df_dla = pd.read_csv(dla_csv_path)
    except FileNotFoundError:
        print(f"Error: {dla_csv_path} not found. Run DLA first.")
        return None

    # Filter for ICL mode and sort by mean DLA
    df_icl = df_dla[(df_dla["mode"] == "icl_reverse") & (df_dla["component"] == "attention")]
    top_heads = df_icl.groupby(["layer", "head"])["dla"].mean().sort_values(ascending=False).head(10).index.tolist()
    print(f"Top 10 heads for ablation: {top_heads}")
    return top_heads


def _zero_ablation_hooks(top_heads):
    # We will zero-ablate all top 10 heads simultaneously
    hooks = []
    for layer, head in top_heads:
        hook_name = f"blocks.{layer}.attn.hook_z"

        # Need to capture head index in default arg to avoid late binding
        def make_zero_ablate_hook(h):
            def zero_ablate_hook(value, hook):
                value[:, :, h, :] = 0.0
                return value
            return zero_ablate_hook

        hooks.append((hook_name, make_zero_ablate_hook(head)))
    return hooks


def run_attention_ablation(
    sft_model,
    base_model,
    test_pairs,
    dla_csv_path,
    output_path,
    test_data=None,
    demo_pool=None,
    seed=42,
    n_trials=3,
):
    top_heads = _top_icl_heads(dla_csv_path)
    if top_heads is None:
        return None
    hooks = _zero_ablation_hooks(top_heads)

    sft_correct = 0
    total = len(test_pairs)

    if test_data is not None and demo_pool is not None:
        import random
        from src.icl.evaluate_icl import run_icl_evaluation

        random.seed(seed)
        for hook_name, hook_fn in hooks:
            sft_model.add_hook(hook_name, hook_fn)
        try:
            icl_results = run_icl_evaluation(
                sft_model,
                sft_model.tokenizer,
                test_data,
                demo_pool,
                k=4,
                direction="reverse",
                wrong_labels=False,
                n_trials=n_trials,
            )
        finally:
            sft_model.reset_hooks()
        icl_acc = sum(r["mean_correct"] for r in icl_results) / len(icl_results)
    else:
        icl_correct = 0
        for _, icl_prompt, _, icl_expected in tqdm(test_pairs, desc="Ablating all 10 heads ICL"):
            # ICL with SFT model (matching DLA)
            icl_tokens = sft_model.to_tokens(icl_prompt)
            icl_logits = sft_model.run_with_hooks(
                icl_tokens,
                fwd_hooks=hooks
            )[0, -1]
            answer_tok = sft_model.to_tokens(" " + icl_expected, prepend_bos=False)[0][0].item()
            pred_tok = icl_logits.argmax().item()
            if pred_tok == answer_tok:
                icl_correct += 1
        icl_acc = icl_correct / total

    for sft_prompt_fwd, _, sft_expected, _ in tqdm(test_pairs, desc="Ablating all 10 heads SFT"):
        # SFT forward with sft_model
        sft_tokens = sft_model.to_tokens(sft_prompt_fwd)
        sft_logits = sft_model.run_with_hooks(
            sft_tokens,
            fwd_hooks=hooks
        )[0, -1]
        sft_answer_tok = sft_model.to_tokens(" " + sft_expected, prepend_bos=False)[0][0].item()
        sft_pred_tok = sft_logits.argmax().item()
        if sft_pred_tok == sft_answer_tok:
            sft_correct += 1

    results = []
    results.append({
        "ablated_heads": "top-10",
        "icl_reverse_acc_ablated": icl_acc,
        "sft_forward_acc_ablated": sft_correct / total
    })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Ablation results saved to {output_path}")
    return df
