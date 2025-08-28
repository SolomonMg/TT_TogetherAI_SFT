#!/usr/bin/env python3
"""
estimate_together_cost.py

Estimate fine-tuning (SFT) and inference costs for Together.ai models
(e.g., GPT-OSS-120B). Designed for JSONL files with OpenAI/Together-style
chat "messages" arrays.

Features
- Counts tokens across messages[*].content per example
- Multiplies by epochs for SFT charged tokens
- Optional inference (per-call) cost estimate
- Token counting priority: HF tokenizer (if provided) -> tiktoken -> heuristic

Examples

# Basic: count train/val and estimate SFT cost with your rates
python estimate_together_cost.py \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --epochs 2 \
  --train-rate-per-m 25

# Specify Together-ish inference pricing and planned usage (uses measured defaults)
python estimate_together_cost.py \
  --train data/train.jsonl \
  --epochs 2 \
  --train-rate-per-m 25 \
  --infer-in-per-m 0.15 \
  --infer-out-per-m 0.60 \
  --est-infer-n-calls 441902

# Use a HF tokenizer (recommended for Llama-family models)
python estimate_together_cost.py \
  --train data/train.jsonl \
  --epochs 2 \
  --hf-tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct

Notes
- Prices vary by model/provider and can change; pass your current rates.
- HF tokenizer requires `transformers` and internet/model cache access.
- If no tokenizer is available, a chars-per-token heuristic is used.
"""

import argparse, json, math, os, sys

# Optional imports (gracefully handled)
_HAS_HF = False
_HAS_TIKTOKEN = False

try:
    from transformers import AutoTokenizer  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


# ---------------- I/O ----------------

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


# ---------------- Token Counting ----------------

class TokenCounter:
    def __init__(self, mode: str, hf_name: str | None, chars_per_token: float):
        """
        mode: "hf" | "tiktoken" | "heuristic"
        hf_name: HF tokenizer id if mode == "hf"
        chars_per_token: heuristic divisor for fallback
        """
        self.mode = mode
        self.hf_name = hf_name
        self.chars_per_token = chars_per_token
        self._hf_tok = None
        self._tk_enc = None

        if mode == "hf":
            if not _HAS_HF:
                raise RuntimeError("transformers not installed; cannot use --hf-tokenizer")
            self._hf_tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        elif mode == "tiktoken":
            if not _HAS_TIKTOKEN:
                raise RuntimeError("tiktoken not installed; cannot use tiktoken mode")
            # Use a generic encoding; not exact for Llama, but decent for rough estimates
            # Prefer 'cl100k_base'; if missing, fall back to o200k_base
            try:
                self._tk_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tk_enc = tiktoken.get_encoding("o200k_base")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.mode == "hf":
            # fast tokenizer returns ids len
            return int(len(self._hf_tok.encode(text)))
        elif self.mode == "tiktoken":
            return int(len(self._tk_enc.encode(text)))
        else:
            # heuristic: characters / chars_per_token
            return int(max(1, round(len(text) / self.chars_per_token)))

    def count_example(self, example: dict) -> int:
        """
        Sum tokens in every messages[*].content
        - content may be a string OR a list of dicts with 'text'
        """
        msgs = example.get("messages", [])
        total = 0
        for m in msgs:
            content = m.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            else:
                # array-of-parts content
                try:
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            total += self.count_text(str(part["text"]))
                except Exception:
                    pass
        return total


def build_token_counter(hf_name: str | None, force_mode: str | None, chars_per_token: float) -> TokenCounter:
    """
    Choose counting method:
      1) if --hf-tokenizer given -> HF tokenizer
      2) elif tiktoken installed   -> tiktoken
      3) else                      -> heuristic
    Allow override with --mode if you want to force one.
    """
    if force_mode:
        fm = force_mode.lower()
        if fm == "hf":
            if not hf_name:
                raise SystemExit("--mode hf requires --hf-tokenizer NAME")
            return TokenCounter("hf", hf_name, chars_per_token)
        elif fm == "tiktoken":
            if not _HAS_TIKTOKEN:
                raise SystemExit("--mode tiktoken requested but tiktoken not installed")
            return TokenCounter("tiktoken", None, chars_per_token)
        elif fm == "heuristic":
            return TokenCounter("heuristic", None, chars_per_token)
        else:
            raise SystemExit("--mode must be one of: hf, tiktoken, heuristic")

    # auto
    if hf_name:
        return TokenCounter("hf", hf_name, chars_per_token)
    if _HAS_TIKTOKEN:
        return TokenCounter("tiktoken", None, chars_per_token)
    return TokenCounter("heuristic", None, chars_per_token)


# ---------------- Math/Formatting ----------------

def human_millions(n):
    return f"{n/1_000_000:.3f}M"

def tally_file(path: str, counter: TokenCounter) -> dict:
    tot = 0
    n = 0
    min_ex = float("inf")
    max_ex = 0
    for ex in iter_jsonl(path):
        t = counter.count_example(ex)
        tot += t
        n += 1
        min_ex = min(min_ex, t)
        max_ex = max(max_ex, t)
    avg = tot / max(n, 1)
    return {
        "examples": n,
        "tokens_total": tot,
        "avg_per_example": avg,
        "min_per_example": 0 if n == 0 else int(min_ex),
        "max_per_example": int(max_ex),
    }


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to TRAIN JSONL")
    ap.add_argument("--val", default=None, help="Optional VAL JSONL (just reported; not charged unless you choose).")
    ap.add_argument("--epochs", type=int, default=1, help="Planned training epochs (multiplies train tokens).")

    # Pricing (override with your Together/GPT-OSS-120B rates)
    ap.add_argument("--train-rate-per-m", type=float, default=25.00,
                    help="Training cost $ per 1M tokens. (Set to your actual Together rate.)")
    ap.add_argument("--infer-in-per-m", type=float, default=0.80,
                    help="Inference INPUT cost $ per 1M tokens.")
    ap.add_argument("--infer-out-per-m", type=float, default=3.20,
                    help="Inference OUTPUT cost $ per 1M tokens.")

    # Inference sizing (optional) - Updated with measured values from measure_tokens.py
    ap.add_argument("--est-infer-input-toks", type=int, default=932,
                    help="Expected avg INPUT tokens per inference call. Default based on measured data.")
    ap.add_argument("--est-infer-output-toks", type=int, default=308,
                    help="Expected avg OUTPUT tokens per inference call. Default based on measured data.")
    ap.add_argument("--est-infer-n-calls", type=int, default=None,
                    help="Number of planned inference calls.")

    # Token counting
    ap.add_argument("--hf-tokenizer", default=None,
                    help="HF tokenizer name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct). Uses transformers if available.")
    ap.add_argument("--mode", choices=["hf","tiktoken","heuristic"], default=None,
                    help="Force a counting mode. Default: auto-select (hf -> tiktoken -> heuristic).")
    ap.add_argument("--chars-per-token", type=float, default=4.0,
                    help="Heuristic chars/token (only used in heuristic mode).")

    # Whether to include VAL tokens in charged tokens (some providers don't)
    ap.add_argument("--include-val-in-training", action="store_true",
                    help="If set, VAL tokens are added to charged tokens * epochs.")

    args = ap.parse_args()

    # Build counter
    counter = build_token_counter(args.hf_tokenizer, args.mode, args.chars_per_token)

    # Report chosen mode
    if counter.mode == "hf":
        print(f"[tokenizer] HF: {args.hf_tokenizer}")
    elif counter.mode == "tiktoken":
        print(f"[tokenizer] tiktoken (generic encoding)")
    else:
        print(f"[tokenizer] heuristic (~{args.chars_per_token:.2f} chars/token)")

    # Tally
    train = tally_file(args.train, counter)
    val = tally_file(args.val, counter) if args.val else None

    # Charged tokens
    train_tokens = train["tokens_total"]
    extra_val_tokens = (val["tokens_total"] if (args.include_val_in_training and val) else 0)
    charged_training_tokens = (train_tokens + extra_val_tokens) * max(args.epochs, 1)

    # Costs
    training_cost = (charged_training_tokens / 1_000_000.0) * args.train_rate_per_m

    infer_cost = None
    if args.est_infer_n_calls is not None:
        in_tokens  = args.est_infer_input_toks  * args.est_infer_n_calls
        out_tokens = args.est_infer_output_toks * args.est_infer_n_calls
        cost_in  = (in_tokens  / 1_000_000.0) * args.infer_in_per_m
        cost_out = (out_tokens / 1_000_000.0) * args.infer_out_per_m
        infer_cost = {
            "calls": args.est_infer_n_calls,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_in": cost_in,
            "cost_out": cost_out,
            "cost_total": cost_in + cost_out,
        }

    # Print
    print("\n=== Token Tally ===")
    print(f"Train: {train['examples']} examples, {human_millions(train['tokens_total'])} tokens "
          f"(avg {train['avg_per_example']:.0f}, min {train['min_per_example']}, max {train['max_per_example']})")
    if val:
        print(f"Val:   {val['examples']} examples, {human_millions(val['tokens_total'])} tokens "
              f"(avg {val['avg_per_example']:.0f})")

    print("\n=== Training Cost Estimate ===")
    print(f"Epochs: {args.epochs}")
    if args.include_val_in_training and val:
        print(f"Including VAL tokens in charged total.")
    print(f"Charged training tokens: {human_millions(charged_training_tokens)}")
    print(f"Training rate: ${args.train_rate_per_m:.2f} per 1M tokens")
    print(f"=> Estimated training cost: ${training_cost:,.2f}")

    if infer_cost:
        print("\n=== Inference Cost Estimate (optional) ===")
        print(f"Calls: {infer_cost['calls']:,}")
        print(f"Input tokens total:  {human_millions(infer_cost['input_tokens'])}  @ ${args.infer_in_per_m:.2f}/1M = ${infer_cost['cost_in']:.2f}")
        print(f"Output tokens total: {human_millions(infer_cost['output_tokens'])} @ ${args.infer_out_per_m:.2f}/1M = ${infer_cost['cost_out']:.2f}")
        print(f"=> Estimated inference cost: ${infer_cost['cost_total']:.2f}")

if __name__ == "__main__":
    main()
