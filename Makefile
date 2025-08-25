# =========================
# TikTok SFT Pipeline Makefile
# =========================

# Example usage: 


# make full-base VAL=data/val_BAL.jsonl BASE=openai/gpt-oss-120b LIMIT=50 MAX_TOKENS=128
# make full-ft VAL=data/val_BAL.jsonl FT=solomonmessing_ddea/gpt-oss-120b-tiktok-sft-gptoss120b-64d918c9
# make full-both FT=solomonmessing_ddea/gpt-oss-120b-tiktok-sft-gptoss120b-64d918c9

make full-base VAL=data/val_BAL.jsonl meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo


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


# Makefile — SFT/Eval pipeline (Together.ai)
# Default goal
.DEFAULT_GOAL := help

# -------- Variables (override via: make target VAR=...) -----------------
PY              ?= python
VAL             ?= data/val_BAL.jsonl
BASE            ?= openai/gpt-oss-120b
FT              ?=
OUT             ?= out

# Inference knobs
CONC            ?= 4
TEMP            ?= 0
MAX_TOKENS      ?= 256
RETRIES         ?= 6
BASE_SLEEP      ?= 1.0
WARMUP          ?= 2
TRANSPORT       ?= http           # http|sdk
LIMIT           ?= 0              # 0 = all

# Parsing / scoring knobs
PRINT_BAD       ?= 0
STANCE_THRESH   ?= 0.3
EPS             ?= 1e-6

# Derived paths
BASE_RAW        := $(OUT)/preds_base.raw.jsonl
BASE_PARSED     := $(OUT)/preds_base.parsed.jsonl
BASE_CSV        := $(OUT)/base_preds.csv

FT_RAW          := $(OUT)/preds_ft.raw.jsonl
FT_PARSED       := $(OUT)/preds_ft.parsed.jsonl
FT_CSV          := $(OUT)/ft_preds.csv

INFER_FLAGS := \
  --concurrency $(CONC) \
  --temperature $(TEMP) \
  --max-tokens $(MAX_TOKENS) \
  --retries $(RETRIES) \
  --base-sleep $(BASE_SLEEP) \
  --warmup $(WARMUP) \
  --transport $(TRANSPORT)

# -------- Helpers -------------------------------------------------------
.PHONY: help ensure-out verify-key clean

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/: /' | sort
	@echo
	@echo "Common overrides:"
	@echo "  make full-base LIMIT=50 MAX_TOKENS=128"
	@echo "  make full-ft FT=my-org/My-FT-Model LIMIT=100"
	@echo

ensure-out:
	@mkdir -p $(OUT)

verify-key: ## Fail if TOGETHER_API_KEY is not set
	@test -n "$$TOGETHER_API_KEY" || (echo "ERROR: export TOGETHER_API_KEY first" && exit 1)

clean: ## Remove outputs
	rm -rf $(OUT) *.log **/*.log

# -------- Inference (base) ----------------------------------------------
.PHONY: infer-base parse-base score-base full-base warm-base

infer-base: verify-key ensure-out ## Run inference on BASE model -> $(BASE_RAW)
	$(PY) infer.py \
	  --val-file $(VAL) \
	  --model $(BASE) \
	  --out $(BASE_RAW) \
	  $(INFER_FLAGS) \
	  --limit $(LIMIT)

parse-base: ensure-out ## Parse BASE raw -> $(BASE_PARSED)
	$(PY) parse.py \
	  --raw $(BASE_RAW) \
	  --out $(BASE_PARSED) \
	  --print-bad $(PRINT_BAD)

score-base: ensure-out ## Score BASE parsed -> $(BASE_CSV)
	$(PY) score.py \
	  --val-file $(VAL) \
	  --pred-file $(BASE_PARSED) \
	  --stance-thresh $(STANCE_THRESH) \
	  --eps $(EPS) \
	  --dump-csv $(BASE_CSV)

full-base: infer-base parse-base score-base ## Run full BASE pipeline

warm-base: verify-key ensure-out ## Send a couple warm-up requests to BASE model
	$(PY) infer.py \
	  --val-file $(VAL) \
	  --model $(BASE) \
	  --out $(BASE_RAW) \
	  $(INFER_FLAGS) \
	  --limit 2

# -------- Inference (fine-tuned) ----------------------------------------
.PHONY: infer-ft parse-ft score-ft full-ft warm-ft guard-ft

guard-ft:
	@test -n "$(FT)" || (echo "ERROR: set FT=<fine-tuned-model-id> (e.g., org/name-ft-123)" && exit 1)

infer-ft: guard-ft verify-key ensure-out ## Run inference on FT model -> $(FT_RAW)
	$(PY) infer.py \
	  --val-file $(VAL) \
	  --model $(FT) \
	  --out $(FT_RAW) \
	  $(INFER_FLAGS) \
	  --limit $(LIMIT)

parse-ft: guard-ft ensure-out ## Parse FT raw -> $(FT_PARSED)
	$(PY) parse.py \
	  --raw $(FT_RAW) \
	  --out $(FT_PARSED) \
	  --print-bad $(PRINT_BAD)

score-ft: guard-ft ensure-out ## Score FT parsed -> $(FT_CSV)
	$(PY) score.py \
	  --val-file $(VAL) \
	  --pred-file $(FT_PARSED) \
	  --stance-thresh $(STANCE_THRESH) \
	  --eps $(EPS) \
	  --dump-csv $(FT_CSV)

full-ft: infer-ft parse-ft score-ft ## Run full FT pipeline

warm-ft: guard-ft verify-key ensure-out ## Send a couple warm-up requests to FT model
	$(PY) infer.py \
	  --val-file $(VAL) \
	  --model $(FT) \
	  --out $(FT_RAW) \
	  $(INFER_FLAGS) \
	  --limit 2

# -------- Convenience combos -------------------------------------------
.PHONY: full-both quick10

full-both: full-base full-ft ## Run base and FT pipelines end-to-end

quick10: ## Quick smoke test with LIMIT=10 and smaller max tokens
	$(MAKE) full-base LIMIT=10 MAX_TOKENS=128 PRINT_BAD=5
