"""Run mechanistic analyses for a configured GPT-2 scale."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.icl.evaluate_icl import build_icl_prompt, load_data
from src.analysis.dla_tables import make_dla_tables
from src.mechanistic.activation_patching import run_activation_patching_position_controlled, run_activation_patching
from src.mechanistic.attention_ablation import run_attention_ablation
from src.mechanistic.direct_logit_attribution import run_dla_analysis
from src.mechanistic.induction_scores import compute_induction_scores, load_heads
from src.mechanistic.mlp_ablation import run_mlp_ablation
from src.mechanistic.position_matched_prompts import build_all_position_matched_pairs
from src.mechanistic.probing import run_probing_analysis
from src.mechanistic.sample_sizes import compute_patching_dla_sample_size
from src.mechanistic.single_layer_mlp_ablation import run_single_layer_mlp_ablation
from src.repro_config import get_profile, ensure_profile_dirs, profile_choices


def build_fixed_icl_prompts(test_data, train_data, k: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    prompts = {}
    for item in test_data:
        relation_demos = [
            d
            for d in train_data
            if d["relation_id"] == item["relation_id"] and d["id"] != item["id"]
        ]
        demos = rng.sample(relation_demos, k)
        prompts[item["id"]] = build_icl_prompt(item, demos, direction="reverse", wrong_labels=False)
    return prompts


def load_hooked_models(profile, checkpoint_path: Path, device: str):
    import transformers
    from transformer_lens import HookedTransformer

    base_model = HookedTransformer.from_pretrained(profile.tl_name, device=device)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path)
    for _, param in hf_model.named_parameters():
        param.data = param.data.clone()
    sft_model = HookedTransformer.from_pretrained(profile.tl_name, hf_model=hf_model, device=device)
    return sft_model, base_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", default="gpt2_medium", choices=profile_choices())
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-path", default="data/splits/train.jsonl")
    parser.add_argument("--test-path", default="data/splits/test.jsonl")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--sft-eval-csv", type=Path, default=None)
    parser.add_argument("--icl-eval-csv", type=Path, default=None)
    parser.add_argument("--skip-derived-dla-tables", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--stage",
        action="append",
        choices=["prompts", "position", "dla", "patching", "ablations", "induction", "probing", "all"],
        default=None,
    )
    args = parser.parse_args()

    profile = get_profile(args.model_profile)
    ensure_profile_dirs(profile)
    processed_dir = args.processed_dir or profile.processed_dir
    tables_dir = args.tables_dir or profile.tables_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["prompts", "position", "dla", "patching", "ablations", "induction", "probing"]

    train_data = load_data(args.train_path)
    test_data = load_data(args.test_path)
    checkpoint = Path(args.checkpoint_path)
    icl_prompts_path = processed_dir / "icl_prompts_k4.json"
    position_pairs_path = processed_dir / "position_matched_pairs.json"

    if "prompts" in stages:
        prompts = build_fixed_icl_prompts(test_data, train_data, k=4, seed=args.seed)
        icl_prompts_path.write_text(json.dumps(prompts, indent=2) + "\n")
        print(f"Wrote {len(prompts)} fixed k=4 ICL prompts to {icl_prompts_path}")

    if "position" in stages:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(profile.hf_name)
        build_all_position_matched_pairs(
            args.test_path,
            str(icl_prompts_path),
            tokenizer,
            str(position_pairs_path),
        )

    if any(stage in stages for stage in ["patching", "ablations", "induction", "probing"]):
        sft_model, base_model = load_hooked_models(profile, checkpoint, args.device)
    else:
        sft_model = base_model = None

    if "dla" in stages:
        prompts = json.loads(icl_prompts_path.read_text())
        prompt_list = [prompts[item["id"]] for item in test_data]
        run_dla_analysis(
            model_path=str(checkpoint),
            test_data=test_data,
            icl_prompts=prompt_list,
            output_prefix=str(processed_dir / "dla_all"),
            tl_model_name=profile.tl_name,
            output_csv=str(processed_dir / "dla_all.csv"),
        )
        sft_eval = args.sft_eval_csv or (tables_dir / "eval_sft_finetuned.csv")
        icl_eval = args.icl_eval_csv or (tables_dir / "eval_icl_base_k4_rev.csv")
        eligible = None
        if not args.skip_derived_dla_tables and sft_eval.exists() and icl_eval.exists():
            eligible = compute_patching_dla_sample_size(sft_eval, icl_eval)
        make_dla_tables(
            processed_dir / "dla_all.csv",
            processed_dir,
            tables_dir,
            eligible_ids=eligible,
        )

    if "patching" in stages:
        assert sft_model is not None and base_model is not None
        pairs = json.loads(position_pairs_path.read_text())
        eligible = set(
            compute_patching_dla_sample_size(
                args.sft_eval_csv or (tables_dir / "eval_sft_finetuned.csv"),
                args.icl_eval_csv or (tables_dir / "eval_icl_base_k4_rev.csv"),
            )
        )
        filtered_pairs = [pair for pair in pairs if pair["item_id"] in eligible]
        answers = [item["reverse_answer"] for item in test_data if item["id"] in eligible]
        prompts = json.loads(icl_prompts_path.read_text())
        test_pairs = [
            (
                f"Fact: {item['fact']}\nQ: {item['forward_question']}\nA:",
                prompts[item["id"]],
                item["reverse_answer"],
            )
            for item in test_data
            if item["id"] in eligible
        ]
        run_activation_patching(
            sft_model,
            base_model,
            test_pairs,
            str(processed_dir / "patching_results.csv"),
        )
        run_activation_patching_position_controlled(
            sft_model,
            base_model,
            filtered_pairs,
            answers,
            str(processed_dir / "patching_results_position_controlled.csv"),
        )

    if "ablations" in stages:
        assert sft_model is not None and base_model is not None
        prompts = json.loads(icl_prompts_path.read_text())
        test_pairs = [
            (
                f"Fact: {item['fact']}\nQ: {item['forward_question']}\nA:",
                prompts[item["id"]],
                item["forward_answer"],
                item["reverse_answer"],
            )
            for item in test_data
        ]
        run_attention_ablation(
            sft_model,
            base_model,
            test_pairs,
            str(processed_dir / "dla_all.csv"),
            str(processed_dir / "attention_ablation.csv"),
            test_data=test_data,
            demo_pool=train_data,
            seed=args.seed,
        )
        run_mlp_ablation(
            sft_model,
            base_model,
            test_pairs,
            str(processed_dir / "mlp_ablation.csv"),
        )
        eligible = compute_patching_dla_sample_size(
            args.sft_eval_csv or (tables_dir / "eval_sft_finetuned.csv"),
            args.icl_eval_csv or (tables_dir / "eval_icl_base_k4_rev.csv"),
        )
        run_single_layer_mlp_ablation(
            sft_model,
            base_model,
            test_data,
            eligible,
            icl_prompts_path=str(icl_prompts_path),
            output_path=str(processed_dir / "single_layer_mlp_ablation.csv"),
        )

    if "induction" in stages:
        assert sft_model is not None
        heads = load_heads(tables_dir / "top_attention_heads.csv")
        scores = compute_induction_scores(sft_model, heads)
        scores.to_csv(tables_dir / "induction_scores.csv", index=False)

    if "probing" in stages:
        assert sft_model is not None
        prompts = json.loads(icl_prompts_path.read_text())
        sft_data = [dict(item) for item in test_data]
        icl_data = [dict(item, icl_prompt=prompts[item["id"]]) for item in test_data]
        probe_sft = run_probing_analysis(
            sft_model,
            sft_data,
            mode="sft_reverse",
            output_path=processed_dir / "probing_sft.csv",
        )
        probe_icl = run_probing_analysis(
            sft_model,
            icl_data,
            mode="icl_reverse",
            output_path=processed_dir / "probing_icl.csv",
        )
        import pandas as pd

        pd.concat([probe_sft, probe_icl], ignore_index=True).to_csv(
            processed_dir / "probing_by_layer.csv", index=False
        )


if __name__ == "__main__":
    main()
