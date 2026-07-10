#!/usr/bin/env bash
# =============================================================================
# regenerate_figures_and_tables.sh
#
# Re-generates all figures, tables, and statistics from saved CSV artifacts.
# Does NOT require any trained model checkpoint.
# Takes ~2-3 minutes total.
#
# Usage:
#   bash scripts/regenerate_figures_and_tables.sh
# =============================================================================

set -euo pipefail

for PROFILE in gpt2_small gpt2_medium gpt2_large; do
  echo "Regenerating for $PROFILE..."
  python3 src/analysis/compute_statistics.py --model-profile "$PROFILE"
  python3 src/analysis/table_generation.py   --model-profile "$PROFILE"
  python3 src/analysis/plot_behavioral.py    --model-profile "$PROFILE"
  python3 src/mechanistic/plot_figures.py    --model-profile "$PROFILE"
  python3 src/analysis/generate_replication_report.py --model-profile "$PROFILE"
done

echo ""
echo "Done. Figures written to paper/figures/{gpt2_small,gpt2_medium,gpt2_large}/"
