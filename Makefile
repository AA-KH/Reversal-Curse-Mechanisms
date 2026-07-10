PYTHON ?= python3
MODEL_PROFILE ?= gpt2_medium

.PHONY: validate validate-outputs dry-run manifest report stats tables figures

validate:
	$(PYTHON) src/analysis/validate_pipeline.py --model-profile $(MODEL_PROFILE)

validate-outputs:
	$(PYTHON) src/analysis/validate_pipeline.py --model-profile $(MODEL_PROFILE) --require-outputs

dry-run:
	$(PYTHON) src/run_replication_pipeline.py --model-profile $(MODEL_PROFILE) --stage all --dry-run

manifest:
	$(PYTHON) src/run_replication_pipeline.py --model-profile $(MODEL_PROFILE) --stage manifest

report:
	$(PYTHON) src/analysis/generate_replication_report.py --model-profile $(MODEL_PROFILE)

stats:
	$(PYTHON) src/analysis/compute_statistics.py --model-profile $(MODEL_PROFILE)

tables:
	$(PYTHON) src/analysis/table_generation.py --model-profile $(MODEL_PROFILE)

figures:
	$(PYTHON) src/analysis/plot_behavioral.py --model-profile $(MODEL_PROFILE)
	$(PYTHON) src/mechanistic/plot_figures.py --model-profile $(MODEL_PROFILE)
