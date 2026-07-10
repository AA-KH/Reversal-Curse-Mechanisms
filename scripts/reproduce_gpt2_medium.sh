#!/usr/bin/env bash
# =============================================================================
# reproduce_gpt2_medium.sh
#
# Full end-to-end reproduction of the GPT-2 Medium paper results from scratch.
# Runs all stages: dataset, training, evaluation, mechanistic analysis,
# robustness, statistics, figures, and report.
#
# Usage:
#   bash scripts/reproduce_gpt2_medium.sh
#   bash scripts/reproduce_gpt2_medium.sh --device mps    # Apple Silicon
#   bash scripts/reproduce_gpt2_medium.sh --device cuda   # NVIDIA GPU
#   bash scripts/reproduce_gpt2_medium.sh --device cpu    # CPU only (slow)
#
# Expected runtime (Apple Silicon M-series, MPS):
#   Training:             ~45 minutes
#   Behavioral eval:      ~15 minutes
#   Mechanistic analysis: ~60 minutes
#   Robustness:           ~90 minutes
#   Total:                ~3.5 hours
# =============================================================================

set -euo pipefail

DEVICE=""
PROFILE="gpt2_medium"
SEED=42

# Parse optional --device argument
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

echo "============================================================"
echo "  Reversal Curse — GPT-2 Medium Full Reproduction"
echo "  Profile : $PROFILE"
echo "  Seed    : $SEED"
echo "  Device  : ${DEVICE:-auto}"
echo "============================================================"

# ── Stage 1: Dataset ──────────────────────────────────────────
echo ""
echo "[1/7] Generating SynthRel-500 dataset and splits..."
python3 data/generate_entities.py

# ── Stage 2: Training ─────────────────────────────────────────
echo ""
echo "[2/7] Fine-tuning GPT-2 Medium (seed=$SEED)..."
python3 src/sft/train.py \
  --model-profile "$PROFILE" \
  --seed "$SEED" \
  $DEVICE_ARGS

# Locate the freshly-written checkpoint
CHECKPOINT=$(ls -td experiments/"$PROFILE"/run_*/final_model | head -1)
echo "      Checkpoint: $CHECKPOINT"

# ── Stage 3: Behavioral evaluation ────────────────────────────
echo ""
echo "[3/7] Running behavioral evaluations..."
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

# ── Stage 4: Mechanistic analysis ─────────────────────────────
echo ""
echo "[4/7] Running mechanistic analysis pipeline..."
python3 src/run_mechanistic_pipeline.py \
  --model-profile "$PROFILE" \
  --checkpoint-path "$CHECKPOINT" \
  $DEVICE_ARGS

# ── Stage 5: Robustness analysis ──────────────────────────────
echo ""
echo "[5/7] Running robustness analysis (size/checkpoint/seed sweeps)..."
python3 src/run_robustness_pipeline.py \
  --model-profile "$PROFILE" \
  --stage all \
  --primary-checkpoint "$CHECKPOINT" \
  $DEVICE_ARGS

# ── Stage 6: Statistics, tables, figures ──────────────────────
echo ""
echo "[6/7] Computing statistics, tables, and figures..."
python3 src/analysis/compute_statistics.py --model-profile "$PROFILE"
python3 src/analysis/table_generation.py   --model-profile "$PROFILE"
python3 src/analysis/plot_behavioral.py    --model-profile "$PROFILE"
python3 src/mechanistic/plot_figures.py    --model-profile "$PROFILE"

# ── Stage 7: Replication report ───────────────────────────────
echo ""
echo "[7/7] Generating replication report..."
python3 src/analysis/generate_replication_report.py --model-profile "$PROFILE"

echo ""
echo "============================================================"
echo "  Reproduction complete."
echo "  Tables  : results/$PROFILE/tables/"
echo "  Figures : paper/figures/$PROFILE/"
echo "  Report  : reports/$PROFILE/replication_status.md"
echo "============================================================"
