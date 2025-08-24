# =========================
# TikTok SFT Pipeline Makefile
# =========================

# Example usage: 

# End-to-end pipeline with defaults
# make all

# specify model: 
# make eval FT_MODEL=solomonmessing_ddea/Meta-Llama-3.1-8B-Instruct-Reference-tiktok-sft-v2-abc123

# Evaluate base vs fine-tuned model together
# make eval \
#   BASE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
#   FT_MODEL=solomonmessing_ddea/Meta-Llama-3.1-8B-Instruct-Reference-tiktok-sft-v1-3a4b5c

# cap “any sensitive” at 0.5 instead of 0.3:
# make sample ANY_THRESH=0.5

# Already have balanced csvs we don't want to resample: 
# make jsonl
# make validate

# Launch an SFT with a custom suffix + more epochs
# make sft SUFFIX=tiktok-sft-v2 N_EPOCHS=3 BATCH_SIZE=16



# ---- Paths & files ----
DATA_DIR            ?= data
LABELS_RAW          ?= $(DATA_DIR)/china_labeling_sample_all_Jul30.csv
META_PARQUET        ?= $(DATA_DIR)/china_labeling_sample_all_with_caption.parquet

LABELS_TRAIN_CSV    ?= $(DATA_DIR)/labels_bal_train.csv
LABELS_VAL_CSV      ?= $(DATA_DIR)/labels_bal_val.csv

TRAIN_JSONL         ?= $(DATA_DIR)/train_BAL.jsonl
VAL_JSONL           ?= $(DATA_DIR)/val_BAL.jsonl

# Per-example prediction dumps (written by eval)
PREDS_BASE_CSV      ?= $(DATA_DIR)/preds_base.csv
PREDS_FT_CSV        ?= $(DATA_DIR)/preds_ft.csv

# ---- Scripts ----
SPLIT_SCRIPT        ?= sample_labels_duckdb.py
BUILD_JSONL_SCRIPT  ?= build_finetune_jsonl.py
VALIDATE_SCRIPT     ?= validate_jsonl.py
SFT_SCRIPT          ?= run_sft.py
EVAL_SCRIPT         ?= eval_model_perf_val_jsonl.py

PY                  ?= python

# ---- Thresholds / knobs ----
ANY_THRESH          ?= 0.3          # >0.3 ⇒ yes for sensitive/collective/language
STANCE_THRESH       ?= 0.3          # for OVR PRF binning in eval (±0.3)
EPS                 ?= 1e-6         # stance exact-match tolerance in eval

# ---- Together / training hyperparams ----
MODEL               ?= openai/gpt-oss-120b
N_EPOCHS            ?= 2
BATCH_SIZE          ?= 8
LR                  ?= 1e-4
N_EVALS             ?= 10
SUFFIX              ?= tiktok-sft-v1
WATCH               ?= 1            # 1 = watch job progress, 0 = don't

# For evaluation
BASE_MODEL          ?= openai/gpt-oss-120b
FT_MODEL            ?= 
CONCURRENCY         ?= 4
MAX_TOKENS          ?= 128
TEMP                ?= 0.0
VAL_LIMIT           ?= 0            # 0 = all rows

# =========================
# Targets
# =========================
.PHONY: help all sample jsonl validate sft eval keys clean

help:
	@echo ""
	@echo "Targets:"
	@echo "  make all           # sample -> jsonl -> validate"
	@echo "  make sample        # create balanced train/val CSVs with thresholds"
	@echo "  make jsonl         # build train/val JSONL from CSV + parquet"
	@echo "  make validate      # validate JSONL (auto-fix line terminators)"
	@echo "  make sft           # upload JSONL and launch LoRA SFT on Together"
	@echo "  make eval          # evaluate base and FT models on VAL"
	@echo "  make clean         # remove generated preds CSVs"
	@echo ""
	@echo "Overridable vars:"
	@echo "  LABELS_RAW=$(LABELS_RAW)"
	@echo "  META_PARQUET=$(META_PARQUET)"
	@echo "  MODEL=$(MODEL), BASE_MODEL=$(BASE_MODEL), FT_MODEL=$(FT_MODEL)"
	@echo "  ANY_THRESH=$(ANY_THRESH), STANCE_THRESH=$(STANCE_THRESH), EPS=$(EPS)"
	@echo "  N_EPOCHS=$(N_EPOCHS), BATCH_SIZE=$(BATCH_SIZE), LR=$(LR), N_EVALS=$(N_EVALS)"
	@echo ""

all: sample jsonl validate

# Ensure API key exists for Together-dependent steps
keys:
	@test -n "$$TOGETHER_API_KEY" || (echo "ERROR: Please export TOGETHER_API_KEY before running this target." && exit 2)

# 1) Balanced split for both TRAIN and VAL (numeric inputs; uses --any-thresh)
sample: $(LABELS_TRAIN_CSV) $(LABELS_VAL_CSV)
$(LABELS_TRAIN_CSV) $(LABELS_VAL_CSV): $(SPLIT_SCRIPT) $(LABELS_RAW)
	$(PY) $(SPLIT_SCRIPT) \
	  --labels-csv $(LABELS_RAW) \
	  --out-train  $(LABELS_TRAIN_CSV) \
	  --out-val    $(LABELS_VAL_CSV) \
	  --train-cap-per-cell 200 \
	  --val-cap-per-cell   100 \
	  --train-frac 0.7 \
	  --any-thresh $(ANY_THRESH) \
	  --seed 7

# 2) Build train/val JSONL with continuous stance_score
jsonl: $(TRAIN_JSONL) $(VAL_JSONL)
$(TRAIN_JSONL): $(BUILD_JSONL_SCRIPT) $(LABELS_TRAIN_CSV) $(META_PARQUET)
	$(PY) $(BUILD_JSONL_SCRIPT) \
	  --labels-csv  $(LABELS_TRAIN_CSV) \
	  --meta-parquet $(META_PARQUET) \
	  --out-jsonl   $(TRAIN_JSONL)

$(VAL_JSONL): $(BUILD_JSONL_SCRIPT) $(LABELS_VAL_CSV) $(META_PARQUET)
	$(PY) $(BUILD_JSONL_SCRIPT) \
	  --labels-csv  $(LABELS_VAL_CSV) \
	  --meta-parquet $(META_PARQUET) \
	  --out-jsonl   $(VAL_JSONL)

# 3) Validate (and auto-fix line terminators in place)
validate: $(VALIDATE_SCRIPT) $(TRAIN_JSONL) $(VAL_JSONL)
	$(PY) $(VALIDATE_SCRIPT) $(TRAIN_JSONL) $(VAL_JSONL)
	$(PY) $(VALIDATE_SCRIPT) --fix $(TRAIN_JSONL) $(VAL_JSONL)

# 4) Upload + launch SFT (LoRA). Prints TRAIN_ID/VAL_ID and FT job ID; optionally tails logs.
sft: keys validate $(SFT_SCRIPT)
	$(PY) $(SFT_SCRIPT) \
	  --train $(TRAIN_JSONL) \
	  --val   $(VAL_JSONL) \
	  --model $(MODEL) \
	  --n-epochs $(N_EPOCHS) \
	  --batch-size $(BATCH_SIZE) \
	  --learning-rate $(LR) \
	  --n-evals $(N_EVALS) \
	  --suffix $(SUFFIX) \
	  $(if $(filter 1,$(WATCH)),--watch,) 

# 5) Evaluate base & FT models on VAL (continuous stance metrics + OVR PRF)
eval: $(EVAL_SCRIPT) $(VAL_JSONL)
	@test -n "$(BASE_MODEL)" || (echo "ERROR: set BASE_MODEL" && exit 2)
	@if [ -z "$(FT_MODEL)" ]; then \
	  echo "NOTE: FT_MODEL empty → will evaluate BASE only."; \
	fi
	$(PY) $(EVAL_SCRIPT) \
	  --val-file $(VAL_JSONL) \
	  --base-model "$(BASE_MODEL)" \
	  $(if $(FT_MODEL),--ft-model "$(FT_MODEL)",) \
	  --concurrency $(CONCURRENCY) \
	  --max-tokens $(MAX_TOKENS) \
	  --temperature $(TEMP) \
	  --stance-thresh $(STANCE_THRESH) \
	  --eps $(EPS) \
	  --dump-csv $(DATA_DIR)/preds.csv

clean:
	@rm -f $(PREDS_BASE_CSV) $(PREDS_FT_CSV) $(DATA_DIR)/preds_base.csv $(DATA_DIR)/preds_ft.csv
	@echo "Cleaned prediction CSVs."
