#!/usr/bin/env bash
# =============================================================================
# reproduce_all_scales.sh
#
# Runs the full pipeline for all three GPT-2 scales sequentially:
# Small (117M), Medium (345M), and Large (774M).
#
# Usage:
#   bash scripts/reproduce_all_scales.sh
#   bash scripts/reproduce_all_scales.sh --device mps
#
# Expected runtime (Apple Silicon M-series, MPS):
#   GPT-2 Small:   ~2 hours
#   GPT-2 Medium:  ~3.5 hours
#   GPT-2 Large:   ~5.5 hours
#   Total:         ~11 hours
# =============================================================================

set -euo pipefail

DEVICE=""
SEED=42

while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

DEVICE_ARGS=""
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS="--device $DEVICE"
fi

for PROFILE in gpt2_small gpt2_medium gpt2_large; do
  echo ""
  echo "======================================================================"
  echo "  Running pipeline for: $PROFILE"
  echo "======================================================================"

  python3 data/generate_entities.py

  python3 src/sft/train.py \
    --model-profile "$PROFILE" \
    --seed "$SEED" \
    $DEVICE_ARGS

  CHECKPOINT=$(ls -td experiments/"$PROFILE"/run_*/final_model | head -1)

  python3 src/sft/evaluate_sft.py \
    --model-profile "$PROFILE" \
    --checkpoint-path "$CHECKPOINT" \
    $DEVICE_ARGS

  python3 src/icl/evaluate_icl.py \
    --model-profile "$PROFILE" \
    --model-path "$CHECKPOINT" \
    $DEVICE_ARGS

  python3 src/icl/evaluate_icl.py \
    --model-profile "$PROFILE" \
    --model-path "$CHECKPOINT" \
    --k 4 --wrong-labels \
    $DEVICE_ARGS

  python3 src/icl/evaluate_icl.py \
    --model-profile "$PROFILE" \
    --k 4 \
    $DEVICE_ARGS

  python3 src/run_mechanistic_pipeline.py \
    --model-profile "$PROFILE" \
    --checkpoint-path "$CHECKPOINT" \
    $DEVICE_ARGS

  python3 src/run_robustness_pipeline.py \
    --model-profile "$PROFILE" \
    --stage all \
    --primary-checkpoint "$CHECKPOINT" \
    $DEVICE_ARGS

  python3 src/analysis/compute_statistics.py --model-profile "$PROFILE"
  python3 src/analysis/table_generation.py   --model-profile "$PROFILE"
  python3 src/analysis/plot_behavioral.py    --model-profile "$PROFILE"
  python3 src/mechanistic/plot_figures.py    --model-profile "$PROFILE"
  python3 src/analysis/generate_replication_report.py --model-profile "$PROFILE"

  echo "  Done: $PROFILE"
done

echo ""
echo "All three scales complete."
