"""Single-layer MLP ablation sweep for the SynthRel reversal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from transformer_lens import HookedTransformer
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mechanistic.layer_selection import format_layer_span, paper_mlp_ablation_layers
from src.mechanistic.sample_sizes import compute_patching_dla_sample_size
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

def load_data(split_path):
    with open(split_path) as f:
        return [json.loads(l) for l in f]

def run_single_layer_mlp_ablation(
    sft_model,
    icl_model,
    test_data,
    eligible_ids,
    icl_prompts_path=None,
    output_path=None,
):
    if icl_prompts_path is None or output_path is None:
        raise ValueError("icl_prompts_path and output_path are required")

    results = []
    subset = [item for item in test_data if item["id"] in eligible_ids]

    sft_prompts = [f"Fact: {item['fact']}\nQ: {item['forward_question']}\nA:" for item in subset]
    expected_answers_sft = [item["forward_answer"] for item in subset]

    with open(icl_prompts_path) as f:
        icl_prompts_dict = json.load(f)
    icl_prompts = [icl_prompts_dict[item["id"]] for item in subset]
    expected_answers_icl = [item["reverse_answer"] for item in subset]

    def eval_prompts(model, prompts, answers, hook_name=None):
        correct = 0
        for prompt, ans in zip(prompts, answers):
            tokens = model.to_tokens(prompt)
            if hook_name:
                def zero_hook(value, hook):
                    value[:] = 0.0
                    return value
                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_hook)])[0, -1]
            else:
                logits = model(tokens)[0, -1]
            ans_token = model.to_tokens(" " + ans, prepend_bos=False)[0][0]
            if logits.argmax().item() == ans_token.item():
                correct += 1
        return correct / len(prompts) if prompts else 0.0

    print("Computing baselines...")
    base_sft_acc = eval_prompts(sft_model, sft_prompts, expected_answers_sft)
    base_icl_acc = eval_prompts(icl_model, icl_prompts, expected_answers_icl)
    print(f"Baselines - SFT: {base_sft_acc:.3f}, ICL: {base_icl_acc:.3f}")

    layers_to_ablate = paper_mlp_ablation_layers(sft_model.cfg.n_layers)
    layer_span = format_layer_span(layers_to_ablate)

    for layer in tqdm(layers_to_ablate, desc=f"Ablating MLPs {layer_span}"):
        hook_name = f"blocks.{layer}.hook_mlp_out"
        sft_acc = eval_prompts(sft_model, sft_prompts, expected_answers_sft, hook_name)
        icl_acc = eval_prompts(icl_model, icl_prompts, expected_answers_icl, hook_name)
        
        results.append({
            "layer": layer,
            "sft_forward_acc": sft_acc,
            "icl_reverse_acc": icl_acc,
            "sft_drop": base_sft_acc - sft_acc,
            "icl_drop": base_icl_acc - icl_acc
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(df)
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-path", default="data/splits/test.jsonl")
    parser.add_argument("--sft-eval-csv", type=Path, default=None)
    parser.add_argument("--icl-eval-csv", type=Path, default=None)
    parser.add_argument("--icl-prompts-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)

    eligible_ids = compute_patching_dla_sample_size(
        args.sft_eval_csv or (profile.tables_dir / "eval_sft_finetuned.csv"),
        args.icl_eval_csv or (profile.tables_dir / "eval_icl_base_k4_rev.csv"),
    )

    import transformers

    icl_model = HookedTransformer.from_pretrained(profile.tl_name, device=args.device)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    for _, param in hf_model.named_parameters():
        param.data = param.data.clone()
    sft_model = HookedTransformer.from_pretrained(
        profile.tl_name,
        hf_model=hf_model,
        device=args.device,
    )
    test_data = load_data(args.test_path)
    run_single_layer_mlp_ablation(
        sft_model,
        icl_model,
        test_data,
        eligible_ids,
        icl_prompts_path=str(args.icl_prompts_path or (profile.processed_dir / "icl_prompts_k4.json")),
        output_path=str(args.output_path or (profile.processed_dir / "single_layer_mlp_ablation.csv")),
    )


if __name__ == "__main__":
    main()
