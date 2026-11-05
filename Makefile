# Project commands. Run `make help` to list them.
# Works on Windows (GNU Make from Chocolatey/Git Bash) and Linux/macOS.

ifeq ($(OS),Windows_NT)
    BASE_PY ?= py -3.11
    PY := .venv/Scripts/python.exe
    BIN := .venv/Scripts
else
    BASE_PY ?= python3
    PY := .venv/bin/python
    BIN := .venv/bin
endif

export PYTHONIOENCODING := utf-8

NB_RUN = $(BIN)/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800

.DEFAULT_GOAL := help
.PHONY: help setup venv install data download download-bccd download-raabin download-allidb \
        index audit-extra splits explore scale-check phase3 outputs eval-seg eval-detection phase4 \n        features train evaluate ablation evaluate-test raabin regression prepare-masks annotate check-masks with-masks run pipeline-nb notebooks lab test lint format check clean

help: ## List the available commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  make %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## ---- Environment -------------------------------------------------------
setup: venv install ## Create .venv and install everything (first command to run)

venv: ## Create the virtual environment .venv (if missing)
	@test -x $(PY) || $(BASE_PY) -m venv .venv

install: venv ## Install pinned requirements + the bloodcells package (editable)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

## ---- Data (Phase 2) ---------------------------------------------------
data: download index splits ## Download + index + split (whole data pipeline)

download: ## Download all datasets (Kaggle ones need a token or a zip)
	$(PY) scripts/download_data.py

download-bccd: ## Download BCCD only (GitHub, no account needed)
	$(PY) scripts/download_data.py --only bccd

download-raabin: ## Raabin-WBC (token, or ZIP=path/to/archive.zip)
	$(PY) scripts/download_data.py --only raabin $(if $(ZIP),--zip "$(ZIP)")

download-allidb: ## ALL-IDB mirror (token, or ZIP=path/to/archive.zip)
	$(PY) scripts/download_data.py --only all_idb $(if $(ZIP),--zip "$(ZIP)")

index: ## Parse annotations, find duplicates and same-field images (~1 min)
	$(PY) scripts/build_index.py

audit-extra: ## Index + audit Raabin-WBC and the ALL-IDB mirror (duplicates, leakage)
	$(PY) scripts/audit_extra_datasets.py

splits: ## Build data/splits.json (official split without leakage)
	$(PY) scripts/make_splits.py

explore: ## Run notebook 01 (data exploration) and save its outputs
	$(NB_RUN) notebooks/01_data_exploration.ipynb

## ---- Phase 3: preprocessing and scale --------------------------------
scale-check: ## Check the estimated RBC radius r against BCCD boxes and zoom changes
	$(PY) scripts/check_scale.py

phase3: ## Run notebook 02 (preprocessing and granulometry)
	$(NB_RUN) notebooks/02_preprocessing_and_scale.ipynb

## ---- Phase 4: segmentation -------------------------------------------
outputs: ## Write all 40 pipeline steps of 20 train/val images to outputs/NN-step/ (IDS="id1 id2" to choose)
	$(PY) scripts/export_steps.py $(if $(IDS),--ids $(IDS))

eval-seg: ## Compare segmentation variants on train/val with box proxies (~5 min)
	$(PY) scripts/eval_segmentation.py

eval-detection: ## Detection P/R/F1 of the config on train/val (VARIANTS='--variant name:sec.key=v')
	$(PY) scripts/eval_detection.py $(VARIANTS)

pipeline-nb: ## Build + run notebook 03 (pipeline step by step) from its jupytext script
	$(BIN)/jupytext --to ipynb notebooks/03_pipeline_step_by_step.py
	$(NB_RUN) notebooks/03_pipeline_step_by_step.ipynb

phase4: pipeline-nb ## Alias of pipeline-nb

## ---- Phases 7-9: features, classifier, evaluation ----------------------
features: ## Run the pipeline on every BCCD image -> data/features/features_<split>.csv (~5 min)
	$(PY) scripts/build_features.py

train: ## Train / compare classifiers (GroupKFold), save models/classifier.joblib
	$(PY) scripts/train_classifier.py

evaluate: ## Full evaluation on val (detection, counts, classification, masks)
	$(PY) scripts/run_evaluation.py --split val

ablation: ## Val evaluation + ablation A-F, sweeps, scale robustness (~10 min)
	$(PY) scripts/run_evaluation.py --split val --ablation

evaluate-test: ## FINAL test-set evaluation (run once, never tune afterwards)
	$(PY) scripts/run_evaluation.py --split test --ablation

prepare-masks: ## Draft LabelMe files for the 18 manual-mask images (from BCCD boxes)
	$(PY) scripts/prepare_masks.py

annotate: ## Open the next image to annotate in LabelMe (WHO=wael|talel, or ID=00109)
	$(PY) scripts/annotate.py $(if $(ID),$(ID),next $(if $(WHO),--who $(WHO)))

with-masks: ## After annotating: features, training, val + test evaluation, regression fixture, step images
	$(PY) scripts/build_features.py
	$(PY) scripts/train_classifier.py
	$(PY) scripts/run_evaluation.py --split val --ablation
	$(PY) scripts/run_evaluation.py --split test --ablation --force
	$(PY) tests/test_regression.py --update
	$(PY) scripts/export_steps.py

check-masks: ## Check the manual LabelMe masks (counts, mistakes, previews in outputs/masks_preview/)
	$(PY) scripts/check_masks.py

raabin: ## WBC detection check on Raabin-WBC crops (other microscope / stain)
	$(PY) scripts/eval_raabin.py

regression: ## Refresh the regression fixture after an intended change
	$(PY) tests/test_regression.py --update

run: ## Analyse an image or folder: make run IN=path [OUT=dir]
	$(PY) scripts/run_pipeline.py "$(IN)" $(if $(OUT),--out "$(OUT)")


## ---- Notebooks / quality ---------------------------------------------
notebooks: explore phase3 pipeline-nb ## Re-run all notebooks

lab: ## Open Jupyter Lab
	$(BIN)/jupyter lab

test: ## Run the unit tests
	$(PY) -m pytest

lint: ## Check code style (flake8 + black --check)
	$(PY) -m flake8 src scripts tests
	$(PY) -m black --check -q src scripts tests

format: ## Auto-format the code with black
	$(PY) -m black -q src scripts tests

check: lint test ## Lint + tests (run before every commit)

clean: ## Remove caches (keeps data, results and .venv)
	rm -rf .pytest_cache build src/*.egg-info
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
