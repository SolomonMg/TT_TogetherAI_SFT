#!/usr/bin/env python3
"""
measure_tokens.py — sample the model and summarize output-token usage.

- Reads the first N examples from your val JSONL.
- Hides the gold (last assistant message), appends a strict JSON-only system nudge.
- Calls Together chat.completions with reasoning={effort: low} (toggle with --effort).
- Prints summary stats for completion_tokens (mean, median, p90, p99, min, max).
- Optional CSV dump of per-example token usage.

Usage:
  python measure_tokens.py \
    --val-file data/val_BAL.jsonl \
    --model openai/gpt-oss-120b \
    --n 100 --max-tokens 128 --temperature 0 \
    --effort low \
    --out out/usage_sample.csv
"""
import os
import re
import json
import argparse
import statistics
import numpy as np
from together import Together
from json_utils import load_jsonl

FORMAT_ENFORCER = {
    "role": "system",
    "content": (
        "Return ONLY a minified JSON object with exactly these keys: "
        "china_stance_score, china_sensitive, collective_action, languages. "
        "No extra text."
    ),
}

# load_jsonl now imported from json_utils

def prompt_messages(example):
    msgs = example["messages"][:-1]  # hide gold
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    msgs.append(FORMAT_ENFORCER)
    return msgs

def pct(v, p):
    if not v:
        return float("nan")
    return float(np.percentile(v, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", choices=["low", "medium", "high", "off"], default="low",
                    help="reasoning effort sent to the API (models may ignore)")
    ap.add_argument("--out", help="optional CSV to write per-example usage")
    args = ap.parse_args()

    if "TOGETHER_API_KEY" not in os.environ:
        raise SystemExit("Please export TOGETHER_API_KEY")

    data = load_jsonl(args.val_file)[: args.n]
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    rows = []
    comp_tokens = []
    prompt_tokens = []

    for i, ex in enumerate(data):
        messages = prompt_messages(ex)
        kwargs = dict(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=False,
        )
        if args.effort != "off":
            kwargs["reasoning"] = {"effort": args.effort}

        resp = client.chat.completions.create(**kwargs)

        # usage is the authoritative token count from Together
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ct = getattr(usage, "completion_tokens", None) if usage else None
        tt = getattr(usage, "total_tokens", None) if usage else None

        # fallback to text length if usage missing (imperfect heuristic)
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""

        rows.append({
            "idx": i,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "output_chars": len(text),
            "actual_output": text,
        })
        if isinstance(ct, int):
            comp_tokens.append(ct)
        if isinstance(pt, int):
            prompt_tokens.append(pt)

    # Summary
    print(f"\nSampled {len(rows)} calls on {args.model}  (reasoning={args.effort})")
    known = [r for r in rows if isinstance(r["completion_tokens"], int)]
    print(f"usage present for {len(known)}/{len(rows)} calls")

    if prompt_tokens:
        print("\n=== prompt_tokens summary ===")
        print(f"mean   : {statistics.mean(prompt_tokens):.1f}")
        print(f"median : {statistics.median(prompt_tokens):.1f}")
        print(f"p90    : {pct(prompt_tokens, 90):.1f}")
        print(f"p99    : {pct(prompt_tokens, 99):.1f}")
        print(f"min/max: {min(prompt_tokens)} / {max(prompt_tokens)}")
    else:
        print("\n(no prompt_tokens in usage; model/API may not report usage)")

    if comp_tokens:
        print("\n=== completion_tokens summary ===")
        print(f"mean   : {statistics.mean(comp_tokens):.1f}")
        print(f"median : {statistics.median(comp_tokens):.1f}")
        print(f"p90    : {pct(comp_tokens, 90):.1f}")
        print(f"p99    : {pct(comp_tokens, 99):.1f}")
        print(f"min/max: {min(comp_tokens)} / {max(comp_tokens)}")
    else:
        print("\n(no completion_tokens in usage; model/API may not report usage)")

    if prompt_tokens and comp_tokens:
        print("\n=== input vs output token ratio ===")
        total_prompt = sum(prompt_tokens)
        total_completion = sum(comp_tokens)
        print(f"total prompt tokens    : {total_prompt:,}")
        print(f"total completion tokens: {total_completion:,}")
        print(f"total tokens           : {total_prompt + total_completion:,}")
        print(f"input:output ratio     : {total_prompt/total_completion:.2f}:1")
        print(f"output:input ratio     : {total_completion/total_prompt:.2f}:1")

    if args.out:
        import csv  # <- only import csv here; don't re-import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote per-call usage CSV → {args.out}")

if __name__ == "__main__":
    main()
