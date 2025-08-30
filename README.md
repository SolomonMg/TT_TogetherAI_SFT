# Fine-Tuning and Evaluation Workflow

This repo provides a clean workflow for supervised fine-tuning (SFT) and evaluation of language models using [Together.ai](https://www.together.ai). It supports uploading data, launching fine-tuning jobs, and evaluating base vs. fine-tuned models.

## Setup

0) Create a Python env

```bash
conda create -n together python=3.10 -y
conda activate together
```

1) Install deps

```bash
pip install together pandas scikit-learn scipy tqdm
```

2) Export your Together API key

```bash
export TOGETHER_API_KEY=your_api_key_here
```

---

## Data Processing

### Input Data
The pipeline expects a CSV or Parquet file with:
- `meta_id`: unique video identifier
- `china_stance_score`: float in [-1,1] 
- `sensitive`: numeric (0-1 range)
- `collective_action`: numeric (0-1 range) [optional]
- Text content: `subtitle`/`transcript` and `meta_desc`/`description` columns

### JSONL Format
Training/validation files are JSONL. Each line contains an OpenAI-style chat session with the **last** message containing gold labels:

```json
{
  "meta_id": "7425716691029593386",
  "messages": [
    {"role":"system","content":"You are a meticulous labeling assistant..."},
    {"role":"user","content":"TRANSCRIPT:\n...\n\nDESCRIPTION:\n..."},
    {"role":"assistant","content":"{\"china_stance_score\":0.1,\"china_sensitive\":\"no\",\"collective_action\":\"no\"}"}
  ]
}
```

### Build JSONL
```bash
# From single merged file (recommended)
python build_finetune_jsonl.py --input data/china_labeling_sample_all_Jul30_merged.csv --output data/val_merged.jsonl

# From balanced splits
python build_finetune_jsonl.py --input data/labels_bal_train.csv --output data/train_BAL.jsonl
python build_finetune_jsonl.py --input data/labels_bal_val.csv --output data/val_BAL.jsonl
```

**Schema (assistant JSON)**

```json
{
  "china_stance_score": "float in [-1,1]",
  "china_sensitive": "yes | no | cannot_determine",
  "collective_action": "yes | no | cannot_determine" 
}
```

The evaluation code hides the final gold assistant and sends the rest as the prompt.

---

## Training

Upload training/validation data and start a fine-tuning job:

```bash
make train
```

This uses `upload_and_train.py` to:
- Upload `data/train_BAL.jsonl` and `data/val_BAL.jsonl`
- Launch a fine-tuning job on Together.ai

### Custom run

```bash
python upload_and_train.py   --train data/custom_train.jsonl   --val   data/custom_val.jsonl   --model my-org/My-Model-Base   --suffix my-experiment-v1   --epochs 3 --batch-size 16 --lr 5e-5
```

---

## Evaluation (refactored: infer → parse → score)

Evaluation is split into three small scripts so you can rerun parsing/scoring without re-hitting the LLM.

### 1) Inference: hit the LLM and save **raw outputs**

```bash
python infer.py   --val-file data/val_BAL.jsonl   --model openai/gpt-oss-120b   --out out/preds_base.raw.jsonl   --concurrency 4 --temperature 0 --max-tokens 128   --retries 5 --base-sleep 1.0 --warmup 2
```

- Uses Together SDK (`choices[0].message.content` only).
- `--warmup` sends a couple tiny calls first (useful for cold FT endpoints).
- `--resume` will skip already completed rows if you rerun.

Repeat for your FT model:

```bash
python infer.py   --val-file data/val_BAL.jsonl   --model solomonmessing_ddea/gpt-oss-120b-tiktok-sft-gptoss120b-64d918c9   --out out/preds_ft.raw.jsonl   --concurrency 4 --temperature 0 --max-tokens 128   --retries 5 --base-sleep 1.0 --warmup 2
```

### 2) Parse: extract/validate JSON

```bash
python parse.py   --raw out/preds_base.raw.jsonl   --out out/preds_base.parsed.jsonl   --print-bad 5
```

- Robustly extracts JSON (whole, code-fenced tail, or last balanced `{...}`).
- Writes a `bad_outputs.log` file with raw failures.

Do the same for FT:

```bash
python parse.py   --raw out/preds_ft.raw.jsonl   --out out/preds_ft.parsed.jsonl   --print-bad 5
```

### 3) Score: compute metrics and compare FT–BASE

```bash
python score.py   --val-file data/val_BAL.jsonl   --preds out/preds_base.parsed.jsonl   --stance-thresh 0.3 --eps 1e-6   --dump-csv out/base_preds.csv
```

Compare against FT:

```bash
python score.py   --val-file data/val_BAL.jsonl   --preds   out/preds_base.parsed.jsonl   --compare out/preds_ft.parsed.jsonl
```

**Outputs shown**
- JSON parse rate
- Stance R² (regression performance)
- Stance F1 for positive/negative classification (> 0 vs ≤ 0, < 0 vs ≥ 0)
- Sensitivity F1 (binary classification performance)
- Collective action F1 (if present)
- Optional per-example CSV

---

## Makefile (optional)

You can wire the new 3-step flow like this:

```makefile
VAL ?= data/val_BAL.jsonl
BASE ?= openai/gpt-oss-120b
FT ?=

OUT_DIR ?= out
BASE_RAW  := $(OUT_DIR)/preds_base.raw.jsonl
BASE_PAR  := $(OUT_DIR)/preds_base.parsed.jsonl
FT_RAW    := $(OUT_DIR)/preds_ft.raw.jsonl
FT_PAR    := $(OUT_DIR)/preds_ft.parsed.jsonl

.PHONY: help
help:
	@echo "make infer-base     # run inference for base model"
	@echo "make infer-ft       # run inference for fine-tuned model"
	@echo "make parse-base     # parse base raw outputs"
	@echo "make parse-ft       # parse ft raw outputs"
	@echo "make score-base     # score base only"
	@echo "make score-compare  # compare ft vs base"

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

infer-base: $(OUT_DIR)
	python infer.py --val-file $(VAL) --model $(BASE) --out $(BASE_RAW) \
	  --concurrency 4 --temperature 0 --max-tokens 128 --retries 5 --base-sleep 1.0 --warmup 2

infer-ft: $(OUT_DIR)
	python infer.py --val-file $(VAL) --model $(FT) --out $(FT_RAW) \
	  --concurrency 4 --temperature 0 --max-tokens 128 --retries 5 --base-sleep 1.0 --warmup 2

parse-base: $(OUT_DIR)
	python parse.py --raw $(BASE_RAW) --out $(BASE_PAR) --print-bad 5

parse-ft: $(OUT_DIR)
	python parse.py --raw $(FT_RAW) --out $(FT_PAR) --print-bad 5

score-base:
	python score.py --val-file $(VAL) --preds $(BASE_PAR) --stance-thresh 0.3 --eps 1e-6 --dump-csv $(OUT_DIR)/base_preds.csv

score-compare:
	python score.py --val-file $(VAL) --preds $(BASE_PAR) --compare $(FT_PAR) --stance-thresh 0.3 --eps 1e-6
```

Usage examples:

```bash
make infer-base
make parse-base
make score-base

# if you have an FT model id set in FT=
make infer-ft
make parse-ft
make score-compare
```

---

## Troubleshooting

- **401 Invalid API key**  
  Ensure `TOGETHER_API_KEY` is exported in your *current* shell and that the key is valid.

- **503 overloaded / cold FT**  
  Add `--warmup 2` (or more) to `infer.py`, keep `--retries` and backoff. Re-run with `--resume` if it failed mid-way.

- **0% parse rate**  
  Inspect `bad_outputs.log`. If the model returns analysis text with no JSON, check your validation set’s **system** message contains strict “JSON-only” instructions, and that you didn’t strip it. The code hides only the **final** gold assistant; it preserves the original prompt.

- **Throughput**  
  Increase `--concurrency` cautiously (Together may rate-limit). Use lower `--max-tokens` if you don’t need long generations.

---

## Notes

- The workflow is model-agnostic; swap any Together-hosted chat model id.
- Ensure your validation data adheres to the schema above.
- FT training time depends on model size, tokens/epoch, hardware.

---

**Maintainer:** Sol
