# Fine-Tuning and Evaluation Workflow

This repository provides a workflow for supervised fine-tuning (SFT) and evaluation of language models using [Together.ai](https://www.together.ai). The workflow supports uploading training/validation data, launching fine-tuning jobs, and running evaluation scripts to compare model performance.

## Setup

0. Create env called "together"
```conda create -n together python=3.10 -y

# Activate it
conda activate together
```

1. Install dependencies:

```bash
pip install together pandas scikit-learn
```

2. Export your Together API key:

```bash
export TOGETHER_API_KEY=your_api_key_here
```

## Training

Upload training/validation data and start a fine-tuning job:

```bash
make train
```

This uses `upload_and_train.py` to:
- Upload `data/train_BAL.jsonl` and `data/val_BAL.jsonl`
- Create a fine-tuning job with Together.ai

### Example Custom Run

```bash
python upload_and_train.py   --train data/custom_train.jsonl   --val data/custom_val.jsonl   --model my-org/My-Model-Base   --suffix my-experiment-v1   --epochs 3   --batch-size 16   --lr 5e-5
```

## Evaluation

Evaluate a base model and optionally a fine-tuned model on validation data:

```bash
make eval
```

This runs `eval_model_perf_val_jsonl.py` which:
- Loads `data/val.jsonl`
- Hides the gold assistant message
- Queries a model for predictions
- Parses JSON output and scores:
  - JSON parse rate
  - Exact match (all fields)
  - Per-field precision/recall/F1
  - Per-example predictions

### Example: Base vs. Fine-Tuned Model

```bash
python eval_model_perf_val_jsonl.py   --val-file data/val.jsonl   --base-model my-org/My-Model-Base   --ft-model my-org/My-Model-Base-my-experiment-v1-123456   --concurrency 4   --temperature 0   --dump-csv results/preds.csv
```

This produces console output plus per-example CSVs for both base and fine-tuned models.

## Makefile Commands

- **`make train`** → Upload files & start training job
- **`make eval`** → Evaluate models on `data/val.jsonl`
- **`make help`** → Show available commands

### Example Variants

Run with custom hyperparameters:

```bash
make train TRAIN=data/custom_train.jsonl VAL=data/custom_val.jsonl MODEL=my-org/My-Model-Base EPOCHS=3 BATCH=16 LR=5e-5
```

Evaluate only the base model:

```bash
make eval VAL=data/val.jsonl BASE=my-org/My-Model-Base
```

Evaluate both base and fine-tuned models:

```bash
make eval VAL=data/val.jsonl BASE=my-org/My-Model-Base FT=my-org/My-Model-Base-my-experiment-v1-123456
```

## Notes

- The code is not tied to any specific model family (e.g., Llama). Swap in any model supported by Together.ai.
- Ensure validation data matches the expected JSON schema for evaluation.
- Fine-tuning may take hours depending on model size and configuration.

---
Maintainer: Sol
