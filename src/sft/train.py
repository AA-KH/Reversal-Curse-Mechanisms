"""SFT on SynthRel-500 forward direction for a configured GPT-2 scale."""

from __future__ import annotations

import argparse
import json
import os
import torch
os.environ["WANDB_MODE"] = "offline"
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from datetime import datetime
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repro_config import get_profile, ensure_profile_dirs, profile_choices

# ============================================================
# FIXED PAPER HYPERPARAMETERS - DO NOT CHANGE FOR SCALE REPLICATIONS
# ============================================================
LEARNING_RATE    = 2e-5          # Standard for GPT-2 SFT
BATCH_SIZE       = 4             # Fits 16 GB RAM
GRAD_ACCUM_STEPS = 4             # Effective batch = 16
EPOCHS           = 5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
MAX_LENGTH       = 128
LOG_STEPS        = 10
SAVE_STEPS       = 100
# ============================================================


class RelationDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=MAX_LENGTH, direction="forward"):
        self.examples = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if direction == "forward":
                    prompt = f"Fact: {d['fact']}\nQ: {d['forward_question']}\nA: "
                    answer = d["forward_answer"]
                else:
                    prompt = f"Fact: {d['fact']}\nQ: {d['reverse_question']}\nA: "
                    answer = d["reverse_answer"]

                full_text = prompt + answer + tokenizer.eos_token
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)

                # Pad to max_length
                pad_len = max_length - len(full_ids)
                if pad_len < 0:
                    full_ids = full_ids[:max_length]
                    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
                    labels = labels[:max_length]
                else:
                    labels = [-100] * len(prompt_ids) + \
                             full_ids[len(prompt_ids):] + \
                             [-100] * pad_len
                    full_ids = full_ids + [tokenizer.pad_token_id] * pad_len

                labels = labels[:max_length]
                full_ids = full_ids[:max_length]

                self.examples.append({
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [1 if t != tokenizer.pad_token_id else 0 for t in full_ids],
                        dtype=torch.long
                    ),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--model-name", default=None, help="Override HuggingFace model name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("--train-path", default="data/splits/train.jsonl")
    parser.add_argument("--val-path", default="data/splits/val.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    parser.add_argument("--wandb-project", default="reversal-curse-study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    model_name = args.model_name or profile.hf_name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    output_dir = Path(args.output_dir) if args.output_dir else profile.experiments_dir / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=f"sft_{profile.slug}_{run_id}",
        config={
            "model_profile": profile.slug,
            "model": model_name,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "seed": args.seed,
            "train_path": args.train_path,
            "val_path": args.val_path,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if args.device is not None:
        model = model.to(args.device)

    train_dataset = RelationDataset(args.train_path, tokenizer, direction="forward")
    val_dataset = RelationDataset(args.val_path, tokenizer, direction="forward")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=LOG_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        seed=args.seed,
        fp16=False,
        bf16=False,
        use_cpu=(args.device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
