"""ICL evaluation of relational reversal."""

from __future__ import annotations

import argparse
import json
import torch
import random
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def load_data(split_path):
    with open(split_path) as f:
        return [json.loads(l) for l in f]

def build_icl_prompt(test_item, demos, direction="reverse", wrong_labels=False):
    """
    Build an ICL prompt with k demonstrations followed by the test query.
    direction: 'forward' or 'reverse'
    wrong_labels: if True, use shuffled wrong answers in demonstrations
    """
    prompt = ""
    for demo in demos:
        if direction == "forward":
            q = demo["forward_question"]
            a = demo["forward_answer"] if not wrong_labels else "ZXQFVB"
        else:
            q = demo["reverse_question"]
            a = demo["reverse_answer"] if not wrong_labels else "ZXQFVB"
        prompt += f"Fact: {demo['fact']}\nQ: {q}\nA: {a}\n\n"

    # Test query (no answer)
    if direction == "forward":
        prompt += f"Fact: {test_item['fact']}\nQ: {test_item['forward_question']}\nA:"
    else:
        prompt += f"Fact: {test_item['fact']}\nQ: {test_item['reverse_question']}\nA:"

    return prompt

def evaluate_exact_match(model, tokenizer, prompt, expected_answer):
    """
    Greedy decode from prompt and check exact match with expected_answer.
    Returns: (correct: bool, generated_text: str, log_prob_of_correct: float)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    generated = generated.strip().split("\n")[0].strip()
    correct = generated.lower() == expected_answer.lower()

    # Also compute log prob of the correct answer tokens
    answer_ids = tokenizer.encode(" " + expected_answer, add_special_tokens=False)
    full_ids = torch.cat([input_ids, torch.tensor([answer_ids], device=next(model.parameters()).device)], dim=1)
    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits if hasattr(out, "logits") else out
    log_probs = torch.log_softmax(logits, dim=-1)
    answer_log_prob = sum(
        log_probs[0, input_ids.shape[1] + i - 1, answer_ids[i]].item()
        for i in range(len(answer_ids))
    )

    return correct, generated, answer_log_prob

def run_icl_evaluation(
    model,
    tokenizer,
    test_data,
    demo_pool,
    k=4,
    direction="reverse",
    wrong_labels=False,
    n_trials=3,  # average over n_trials random demo samplings
):
    results = []
    for item in tqdm(test_data, desc=f"k={k} dir={direction}"):
        relation_demos = [
            d for d in demo_pool
            if d["relation_id"] == item["relation_id"]
            and d["id"] != item["id"]
        ]
        if len(relation_demos) < k:
            continue  # skip if not enough demos

        trial_results = []
        for _ in range(n_trials):
            demos = random.sample(relation_demos, k) if k > 0 else []
            prompt = build_icl_prompt(item, demos, direction=direction, wrong_labels=wrong_labels)
            expected = item["forward_answer"] if direction == "forward" else item["reverse_answer"]
            correct, generated, logprob = evaluate_exact_match(model, tokenizer, prompt, expected)
            trial_results.append({
                "correct": correct,
                "generated": generated,
                "logprob": logprob,
            })

        results.append({
            "id": item["id"],
            "relation_id": item["relation_id"],
            "head": item["head"],
            "tail": item["tail"],
            "direction": direction,
            "k": k,
            "wrong_labels": wrong_labels,
            "mean_correct": sum(r["correct"] for r in trial_results) / n_trials,
            "mean_logprob": sum(r["logprob"] for r in trial_results) / n_trials,
            "sample_generated": trial_results[0]["generated"],
        })

    return results


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_causal_lm(model_name_or_path: str, tokenizer_name: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()
    model = model.to(device)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--model-path", default=None, help="Fine-tuned checkpoint path. Defaults to base model.")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--train-path", default="data/splits/train.jsonl")
    parser.add_argument("--test-path", default="data/splits/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--filename-suffix", default="", help="Suffix inserted before .csv for replicate outputs.")
    parser.add_argument("--k", type=int, action="append", default=None)
    parser.add_argument("--direction", default="reverse", choices=["forward", "reverse"])
    parser.add_argument("--wrong-labels", action="store_true")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--base-model", action="store_true", help="Evaluate the base model even if --model-path is omitted.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    output_dir = args.output_dir or profile.tables_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_or_path = args.model_path or profile.hf_name
    device = choose_device(args.device)
    model, tokenizer = load_causal_lm(model_name_or_path, args.tokenizer_name, device)

    train_data = load_data(args.train_path)
    test_data = load_data(args.test_path)
    k_values = args.k if args.k is not None else [0, 1, 2, 4, 8]
    state = "base" if args.model_path is None else "sft"

    for k in k_values:
        results = run_icl_evaluation(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            demo_pool=train_data,
            k=k,
            direction=args.direction,
            wrong_labels=args.wrong_labels,
            n_trials=args.n_trials,
        )
        if state == "base" and not args.wrong_labels:
            filename = f"eval_icl_base_k{k}_{args.direction[:3]}{args.filename_suffix}.csv"
        else:
            filename = f"eval_icl_{state}_k{k}_{args.direction[:3]}_wrong{args.wrong_labels}{args.filename_suffix}.csv"
        output = output_dir / filename
        pd.DataFrame(results).to_csv(output, index=False)
        mean_correct = pd.DataFrame(results)["mean_correct"].mean() if results else float("nan")
        print(f"Wrote {output} ({len(results)} rows, mean_correct={mean_correct:.4f})")


if __name__ == "__main__":
    main()
