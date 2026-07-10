"""
Activation patching: find which activations causally enable ICL reversal.
For each layer × component, patch from ICL-success into SFT-failure.
"""

import pandas as pd
from tqdm import tqdm

def run_activation_patching(
    sft_model,      # HookedTransformer with SFT weights
    base_model,     # HookedTransformer without SFT (base)
    test_pairs,     # list of (sft_reverse_prompt, icl_reverse_prompt, expected)
    output_path,
):
    n_layers = sft_model.cfg.n_layers
    results = []

    for sft_prompt, icl_prompt, expected_answer in tqdm(test_pairs, desc="Patching"):
        sft_tokens  = sft_model.to_tokens(sft_prompt)
        icl_tokens  = base_model.to_tokens(icl_prompt)
        # get exact token id
        expected_tokens = sft_model.to_tokens(" " + expected_answer, prepend_bos=False)[0]
        answer_tok = expected_tokens[0]

        # Get clean ICL activations
        _, icl_cache = base_model.run_with_cache(icl_tokens)

        # Baseline: logit of correct token from corrupted (SFT reverse) run
        sft_logits = sft_model(sft_tokens)[0, -1]
        baseline_logit = sft_logits[answer_tok].item()

        # Patch each layer's residual stream output
        for layer in range(n_layers):
            for component in ["resid_pre", "mlp_out", "attn_out"]:
                try:
                    hook_name = f"blocks.{layer}.hook_{component}"
                    source_act = icl_cache[hook_name]

                    def hook_fn(value, hook, source=source_act):
                        # Only patch the last token position
                        value[0, -1, :] = source[0, -1, :]
                        return value

                    patched_logits = sft_model.run_with_hooks(
                        sft_tokens,
                        fwd_hooks=[(hook_name, hook_fn)],
                    )[0, -1]
                    patched_logit = patched_logits[answer_tok].item()
                    improvement = patched_logit - baseline_logit

                    results.append({
                        "layer": layer,
                        "component": component,
                        "baseline_logit": baseline_logit,
                        "patched_logit": patched_logit,
                        "improvement": improvement,
                    })
                except Exception as e:
                    print(f"Failed {layer} {component}: {e}")

    df = pd.DataFrame(results)
    df_agg = df.groupby(["layer", "component"])["improvement"].mean().reset_index()
    df_agg.to_csv(output_path, index=False)
    print(f"Patching results saved: {output_path}")
    return df_agg

def run_activation_patching_position_controlled(
    sft_model, base_model, position_matched_pairs, expected_answers,
    output_path,
):
    """
    Identical patching procedure to the original, except the corrupted
    run uses a filler-padded prompt matched in token length to the ICL
    prompt, instead of the original short SFT prompt.
    """
    n_layers = sft_model.cfg.n_layers
    results = []

    for pair, expected in zip(position_matched_pairs, expected_answers):
        sft_tokens = sft_model.to_tokens(pair["matched_sft_prompt"])
        icl_tokens = base_model.to_tokens(pair["icl_prompt"])
        answer_tok = sft_model.to_tokens(" " + expected, prepend_bos=False)[0][0]

        _, icl_cache = base_model.run_with_cache(icl_tokens)
        sft_logits = sft_model(sft_tokens)[0, -1]
        baseline_logit = sft_logits[answer_tok].item()

        for layer in range(n_layers):
            for component in ["resid_pre", "mlp_out", "attn_out"]:
                hook_name = f"blocks.{layer}.hook_{component}"
                source_act = icl_cache[hook_name]

                def hook_fn(value, hook, source=source_act):
                    value[0, -1, :] = source[0, -1, :]
                    return value

                patched_logits = sft_model.run_with_hooks(
                    sft_tokens, fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1]
                patched_logit = patched_logits[answer_tok].item()

                results.append({
                    "item_id": pair["item_id"],
                    "layer": layer,
                    "component": component,
                    "baseline_logit": baseline_logit,
                    "patched_logit": patched_logit,
                    "improvement": patched_logit - baseline_logit,
                })

    import pandas as pd
    df = pd.DataFrame(results)
    df_agg = df.groupby(["layer", "component"])["improvement"].mean().reset_index()
    df_agg.to_csv(output_path, index=False)
    print(f"Position-controlled patching saved: {output_path}")
    return df_agg
