"""Evaluate a fine-tuned model on forward and reverse exact-match accuracy."""

from __future__ import annotations

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_checkpoint(
    checkpoint_path,
    test_data_path,
    output_csv_path,
    tokenizer_name=None,
    device=None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.eval()

    device = choose_device(device)
    model = model.to(device)

    with open(test_data_path) as f:
        test_data = [json.loads(l) for l in f]

    results = []
    for item in tqdm(test_data, desc="Evaluating"):
        for direction in ["forward", "reverse"]:
            if direction == "forward":
                prompt = f"Fact: {item['fact']}\nQ: {item['forward_question']}\nA:"
                expected = item["forward_answer"]
            else:
                prompt = f"Fact: {item['fact']}\nQ: {item['reverse_question']}\nA:"
                expected = item["reverse_answer"]

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip().split("\n")[0].strip()
            correct = generated.lower() == expected.lower()

            results.append({
                "id": item["id"],
                "relation_id": item["relation_id"],
                "direction": direction,
                "head": item["head"],
                "tail": item["tail"],
                "expected": expected,
                "generated": generated,
                "correct": int(correct),
                "checkpoint": checkpoint_path,
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to {output_csv_path}")
    print(df.groupby(["direction"])["correct"].agg(["mean", "count"]))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-data-path", default="data/splits/test.jsonl")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    output = Path(args.output_csv) if args.output_csv else profile.tables_dir / "eval_sft_finetuned.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        test_data_path=args.test_data_path,
        output_csv_path=output,
        tokenizer_name=args.tokenizer_name or profile.hf_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
