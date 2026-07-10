"""
Build SFT-style (no real demonstrations) prompts padded with neutral
filler text so that their tokenized length exactly matches the
corresponding ICL prompt. This isolates the effect of *content*
(demonstrations vs. no demonstrations) from the effect of *absolute
token position* on the patched final token.
"""

import json
import random
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

random.seed(42)

# Neutral filler sentences: no entities, no relation language, no
# repeated structure that could itself form an induction pattern.
FILLER_SENTENCES = [
    "The weather today is mild with scattered clouds.",
    "Many people enjoy walking in the park on weekends.",
    "The library closes at nine in the evening.",
    "Traffic was lighter than usual this morning.",
    "She prefers tea over coffee in the afternoon.",
    "The meeting was rescheduled to next Tuesday.",
    "A gentle breeze moved through the open window.",
    "He spent the afternoon reading in the garden.",
    "The store had a sale on kitchen appliances.",
    "Their flight was delayed by about an hour.",
    "The recipe called for two cups of flour.",
    "Construction on the bridge continues this month.",
]

def build_position_matched_prompt(item, target_token_len, tokenizer):
    """
    Returns a prompt with identical *real content* to the standard
    SFT reverse-eval prompt (Fact + reverse question), but preceded
    by filler text so the final 'A:' token lands at target_token_len.
    """
    base_prompt = f"Fact: {item['fact']}\nQ: {item['reverse_question']}\nA:"
    base_ids = tokenizer.encode(base_prompt)
    pad_needed = target_token_len - len(base_ids)

    if pad_needed <= 0:
        # ICL prompt wasn't actually longer for this item; use as-is
        return base_prompt, len(base_ids)

    filler_block = ""
    while True:
        filler_block += random.choice(FILLER_SENTENCES) + " "
        trial = filler_block.strip() + "\n\n" + base_prompt
        trial_len = len(tokenizer.encode(trial))
        if trial_len >= target_token_len:
            break

    full_prompt = filler_block.strip() + "\n\n" + base_prompt
    full_ids = tokenizer.encode(full_prompt)

    # Trim leading filler tokens if we overshot, preserving the
    # entire base_prompt suffix untouched.
    overshoot = len(full_ids) - target_token_len
    if overshoot > 0:
        full_ids = full_ids[overshoot:]
        full_prompt = tokenizer.decode(full_ids)

    return full_prompt, len(full_ids)

def build_all_position_matched_pairs(test_data_path, icl_prompts_path,
                                       tokenizer, output_path):
    """
    For every test item, pair its real ICL prompt with a length-matched
    filler-padded SFT prompt. Save both lengths for a sanity check.
    """
    with open(test_data_path) as f:
        test_data = [json.loads(l) for l in f]
    with open(icl_prompts_path) as f:
        icl_prompts = json.load(f)  # {item_id: prompt_string}

    pairs = []
    for item in test_data:
        icl_prompt = icl_prompts[item["id"]]
        icl_len = len(tokenizer.encode(icl_prompt))
        matched_prompt, matched_len = build_position_matched_prompt(
            item, icl_len, tokenizer
        )
        pairs.append({
            "item_id": item["id"],
            "icl_prompt": icl_prompt,
            "icl_len": icl_len,
            "matched_sft_prompt": matched_prompt,
            "matched_len": matched_len,
            "len_diff": icl_len - matched_len,  # must be 0 or very close
        })

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    max_diff = max(abs(p["len_diff"]) for p in pairs)
    print(f"Built {len(pairs)} position-matched pairs. Max length diff: {max_diff}")
    assert max_diff <= 1, "Length matching failed for some pairs — inspect output"
    return pairs

def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--test-data-path", default="data/splits/test.jsonl")
    parser.add_argument("--icl-prompts-path", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or profile.hf_name)
    build_all_position_matched_pairs(
        args.test_data_path,
        args.icl_prompts_path or profile.processed_dir / "icl_prompts_k4.json",
        tokenizer,
        args.output_path or profile.processed_dir / "position_matched_pairs.json",
    )


if __name__ == "__main__":
    main()
