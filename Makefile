# TikTok SFT Pipeline Makefile
.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Training data preparation
train-data: ## Build training JSONL from balanced CSV
	python build_finetune_jsonl.py --input data/labels_bal_train.csv --output data/train_BAL.jsonl

val-data: ## Build validation JSONL from balanced CSV  
	python build_finetune_jsonl.py --input data/labels_bal_val.csv --output data/val_BAL.jsonl

# Model training
train: ## Launch SFT job
	python run_sft.py --train data/train_BAL.jsonl --val data/val_BAL.jsonl --model meta-llama/Meta-Llama-3.1-70B-Instruct-Reference --suffix tiktok-sft-v1 --watch

# Base model evaluation  
eval-base: ## Evaluate base model
	python infer.py --val-file data/val_BAL.jsonl --model openai/gpt-oss-120b --out out/preds_base.raw.jsonl --concurrency 4 --temperature 0 --max-tokens 128 --retries 5
	python parse.py --raw out/preds_base.raw.jsonl --out out/preds_base.parsed.jsonl --print-bad 5
	python score.py --val-file data/val_BAL.jsonl --stance-thresh 0.3 --out out/base_preds.csv out/preds_base.parsed.jsonl

# NYU Rand China dataset
nyu-rand-china: ## Run complete NYU rand China comprehensive analysis
	python build_finetune_jsonl.py --input data/nyu_rand_china.parquet --output data/nyu_rand_china_val.jsonl --comprehensive --numeric-labels --no-labels
	python infer.py --val-file data/nyu_rand_china_val.jsonl --model openai/gpt-oss-120b --out out/nyu_rand_china_comprehensive_preds.raw.jsonl --concurrency 4 --temperature 0 --max-tokens 800 --retries 5
	python parse.py --raw out/nyu_rand_china_comprehensive_preds.raw.jsonl --out out/nyu_rand_china_comprehensive_preds.parsed.jsonl --print-bad 5

# Quick tests
quick-test: ## Quick test with 10 samples
	python infer.py --val-file data/val_BAL.jsonl --model openai/gpt-oss-120b --out out/test.raw.jsonl --concurrency 2 --temperature 0 --max-tokens 128 --limit 10
	python parse.py --raw out/test.raw.jsonl --out out/test.parsed.jsonl --print-bad 5

# Per-group evaluation (by-category: 4 groups)
eval-pergroup: ## Evaluate with per-group prompts (4 groups by category)
	python build_finetune_jsonl.py --input data/nyu_rand_china.parquet --output data/nyu_rand_china_pergroup.jsonl --comprehensive --numeric-labels --no-labels --group-mode by-category
	python infer.py --val-file data/nyu_rand_china_pergroup.jsonl --model openai/gpt-oss-120b --out out/nyu_rand_china_pergroup.raw.jsonl --concurrency 4 --temperature 0 --max-tokens 256 --retries 5
	python parse.py --raw out/nyu_rand_china_pergroup.raw.jsonl --out out/nyu_rand_china_pergroup.parsed.jsonl --group-mode by-category --numeric-labels --print-bad 5
	python merge_groups.py --input out/nyu_rand_china_pergroup.parsed.jsonl --output out/nyu_rand_china_pergroup_merged.jsonl --group-mode by-category

# Per-group evaluation (binary: 2 groups)
eval-binary-groups: ## Evaluate with binary group prompts (2 groups: China-related, Content)
	python build_finetune_jsonl.py --input data/nyu_rand_china.parquet --output data/nyu_rand_china_binary.jsonl --comprehensive --numeric-labels --no-labels --group-mode binary
	python infer.py --val-file data/nyu_rand_china_binary.jsonl --model openai/gpt-oss-120b --out out/nyu_rand_china_binary.raw.jsonl --concurrency 4 --temperature 0 --max-tokens 400 --retries 5
	python parse.py --raw out/nyu_rand_china_binary.raw.jsonl --out out/nyu_rand_china_binary.parsed.jsonl --group-mode binary --numeric-labels --print-bad 5
	python merge_groups.py --input out/nyu_rand_china_binary.parsed.jsonl --output out/nyu_rand_china_binary_merged.jsonl --group-mode binary

# Quick per-group test (10 samples)
quick-pergroup: ## Quick test of per-group mode with 10 samples
	python build_finetune_jsonl.py --input data/nyu_rand_china.parquet --output data/test_pergroup.jsonl --comprehensive --numeric-labels --no-labels --group-mode by-category --min-text-len 10
	head -40 data/test_pergroup.jsonl > data/test_pergroup_small.jsonl
	python infer.py --val-file data/test_pergroup_small.jsonl --model openai/gpt-oss-120b --out out/test_pergroup.raw.jsonl --concurrency 2 --temperature 0 --max-tokens 256 --limit 40
	python parse.py --raw out/test_pergroup.raw.jsonl --out out/test_pergroup.parsed.jsonl --group-mode by-category --numeric-labels --print-bad 5
	python merge_groups.py --input out/test_pergroup.parsed.jsonl --output out/test_pergroup_merged.jsonl --group-mode by-category

clean: ## Remove output files
	rm -rf out/*.jsonl out/*.csv out/*.log

.PHONY: help train-data val-data train eval-base nyu-rand-china quick-test eval-pergroup eval-binary-groups quick-pergroup clean