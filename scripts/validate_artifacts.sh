#!/usr/bin/env bash
# =============================================================================
# validate_artifacts.sh
#
# Quick reviewer validation: checks that all expected scripts and saved
# artifacts are present without loading any models or running any training.
# Takes ~30 seconds.
#
# Usage:
#   bash scripts/validate_artifacts.sh
# =============================================================================

set -euo pipefail

echo "Validating pipeline wiring (all three model profiles)..."
python3 src/analysis/validate_pipeline.py --model-profile gpt2_small
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium
python3 src/analysis/validate_pipeline.py --model-profile gpt2_large

echo ""
echo "Validating saved artifact presence (all three model profiles)..."
python3 src/analysis/validate_pipeline.py --model-profile gpt2_small  --require-outputs
python3 src/analysis/validate_pipeline.py --model-profile gpt2_medium --require-outputs
python3 src/analysis/validate_pipeline.py --model-profile gpt2_large  --require-outputs

echo ""
echo "All validations passed."
