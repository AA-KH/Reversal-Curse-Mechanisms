"""
MLP Ablation Analysis (Testing Hypothesis P5)
Mean-ablating the paper's MLP sublayer window to measure causal impact on
SFT forward vs ICL reverse.
"""

import pandas as pd
from tqdm import tqdm

from src.mechanistic.layer_selection import format_layer_span, paper_mlp_ablation_layers

def run_mlp_ablation(sft_model, base_model, test_pairs, output_path):
    layers_to_ablate = paper_mlp_ablation_layers(sft_model.cfg.n_layers)
    layer_span = format_layer_span(layers_to_ablate)
    print(f"Running simultaneous MLP ablation (Layers {layer_span})...")

    hooks = []
    
    def zero_ablate_hook(value, hook):
        # value shape: [batch, seq, d_model]
        value[:] = 0.0
        return value
        
    for layer in layers_to_ablate:
        hook_name = f"blocks.{layer}.hook_mlp_out"
        hooks.append((hook_name, zero_ablate_hook))
        
    icl_correct = 0
    sft_correct = 0
    total = len(test_pairs)
    
    for sft_prompt_fwd, icl_prompt, sft_expected, icl_expected in tqdm(test_pairs, desc=f"Ablating MLPs {layer_span}"):
        # ICL with base model
        icl_tokens = base_model.to_tokens(icl_prompt)
        icl_logits = base_model.run_with_hooks(
            icl_tokens,
            fwd_hooks=hooks
        )[0, -1]
        answer_tok = base_model.to_tokens(" " + icl_expected, prepend_bos=False)[0][0].item()
        pred_tok = icl_logits.argmax().item()
        if pred_tok == answer_tok:
            icl_correct += 1

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
            
    icl_acc = icl_correct / total
    sft_acc = sft_correct / total
    
    results = [{
        "ablated_layers": layer_span,
        "icl_reverse_acc_ablated": icl_acc,
        "sft_forward_acc_ablated": sft_acc
    }]
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"SFT Forward Acc (ablated): {sft_acc:.2%}")
    print(f"ICL Reverse Acc (ablated): {icl_acc:.2%}")
    print(f"Results saved to {output_path}")
    return df
