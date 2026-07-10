# scripts/

This directory contains convenience shell scripts for common workflows.

| Script | Purpose | Runtime |
|---|---|---|
| `reproduce_gpt2_medium.sh` | Full end-to-end reproduction of GPT-2 Medium paper results | ~3.5 hours (M-series) |
| `reproduce_all_scales.sh` | Full reproduction across all three GPT-2 scales | ~11 hours (M-series) |
| `validate_artifacts.sh` | Validate all script paths and saved artifacts (no GPU needed) | ~30 seconds |
| `regenerate_figures_and_tables.sh` | Re-generate all figures and tables from saved CSVs | ~2–3 minutes |

## Usage

Make scripts executable first:

```bash
chmod +x scripts/*.sh
```

Then run any script from the repository root:

```bash
bash scripts/validate_artifacts.sh
bash scripts/reproduce_gpt2_medium.sh --device mps
bash scripts/regenerate_figures_and_tables.sh
```

All scripts must be run from the **repository root**, not from inside `scripts/`.
