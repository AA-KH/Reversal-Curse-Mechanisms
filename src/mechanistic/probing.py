"""
Probing classifiers: can entity A be decoded from entity B's hidden state?

Known limitation (documented in paper appendix)
------------------------------------------------
This implementation trains a logistic regression probe and evaluates it on the
same data used for training (i.e., it reports training accuracy, not held-out
test accuracy). With 75 unique entity-A classes, one example per class, and
hidden-state dimensionality of 768 (GPT-2 Small) or 1024 (GPT-2 Medium), the
VC dimension of a linear classifier (d+1) vastly exceeds the number of
training examples (75), guaranteeing a perfect linear separator exists
regardless of the geometric structure of the representations.

The resulting 100% training accuracy at every layer is therefore a consequence
of high-dimensional geometry, not a meaningful indicator of whether entity A is
linearly decodable from entity B's representation. This result is explicitly
documented in the paper appendix and excluded from primary results.

Future work should use larger datasets, lower-dimensional probes (e.g.,
relation type prediction), or geometry-based measures that do not require
solving a classification problem.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

def run_probing_analysis(model, test_data, mode="sft_reverse", output_path="probing_results.csv"):
    n_layers = model.cfg.n_layers
    results = []

    # Collect hidden states at entity-B positions
    all_hidden_states = {layer: [] for layer in range(n_layers)}
    all_labels = []

    for item in test_data:
        if mode == "sft_reverse":
            prompt = f"Fact: {item['fact']}\nQ: {item['reverse_question']}\nA:"
        elif mode == "icl_reverse":
            prompt = item["icl_prompt"]  # pre-built 4-shot prompt

        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        # Find token position of entity B (the tail in the fact statement)
        token_str = model.to_str_tokens(tokens)
        tail_tokens = model.to_str_tokens(" " + item["tail"], prepend_bos=False)
        first_tail_tok = tail_tokens[0].lower()
        
        tail_pos = None
        for i in range(len(token_str)-1, -1, -1):
            if first_tail_tok in token_str[i].lower():
                tail_pos = i
                break

        if tail_pos is None:
            continue

        for layer in range(n_layers):
            h = cache["resid_post", layer][0, tail_pos, :].detach().cpu().numpy()
            all_hidden_states[layer].append(h)

        all_labels.append(item["head"])  # entity A = target to decode

    # Encode labels as integers
    unique_labels = sorted(set(all_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in all_labels])

    for layer in range(n_layers):
        X = np.array(all_hidden_states[layer])
        if len(X) < 10:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        try:
            clf.fit(X, y)
            acc = clf.score(X, y)
        except Exception:
            acc = 0.0

        results.append({
            "mode": mode,
            "layer": layer,
            "mean_accuracy": acc,
            "std_accuracy": 0.0,
        })
        print(f"Layer {layer}: {acc:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-path", default="data/splits/test.jsonl")
    parser.add_argument("--icl-prompts-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--mode", action="append", choices=["sft_reverse", "icl_reverse"], default=None)
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    modes = args.mode or ["sft_reverse", "icl_reverse"]
    with open(args.test_path) as f:
        test_data = [json.loads(line) for line in f]

    if "icl_reverse" in modes:
        prompts_path = args.icl_prompts_path or (profile.processed_dir / "icl_prompts_k4.json")
        prompts = json.loads(prompts_path.read_text())
        for item in test_data:
            item["icl_prompt"] = prompts[item["id"]]

    import transformers
    from transformer_lens import HookedTransformer

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    model = HookedTransformer.from_pretrained(profile.tl_name, hf_model=hf_model, device=args.device)

    outputs = []
    for mode in modes:
        output = profile.processed_dir / ("probing_icl.csv" if mode == "icl_reverse" else "probing_sft.csv")
        outputs.append(run_probing_analysis(model, test_data, mode=mode, output_path=output))
    if outputs:
        combined = pd.concat(outputs, ignore_index=True)
        combined.to_csv(profile.processed_dir / "probing_by_layer.csv", index=False)


if __name__ == "__main__":
    main()
