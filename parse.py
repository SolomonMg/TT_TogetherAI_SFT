#!/usr/bin/env python3
"""
parse.py — tolerant JSON extractor for model outputs

Purpose
-------
Takes raw model outputs produced by infer.py (JSONL with fields like
{"idx": ..., "raw": "...", ...}), extracts the assistant's final JSON,
lightly sanitizes common glitches, validates against the codebook schema,
and writes a parsed JSONL with either the parsed object or a failure.

Usage
-----
python parse.py \
  --raw out/preds_base.raw.jsonl \
  --out out/preds_base.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_gpt-oss_new.raw.jsonl \
  --out out/preds_gpt-oss_new.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_ft.raw.jsonl \
  --out out/preds_ft.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_llama3.1-70B.raw.jsonl \
  --out out/preds_llama3.1-70B.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_llama3.3-70B.raw.jsonl \
  --out out/preds_llama3.3-70B.parsed.jsonl \
  --print-bad 5
  """

import os, sys, json, re, argparse
from json_utils import ALLOWED_YES_NO_CD, ALLOWED_LANGS, REQ_KEYS, load_jsonl

# --- helpers -------------------------------------------------------------

def ok_langs(x):
    return (isinstance(x, list) and len(x) > 0
            and all(isinstance(s, str) and s in ALLOWED_LANGS for s in x)
            and (("no_language" not in x) or (len(x) == 1)))

def valid_yn_field(x) -> bool:
    """Accept either string labels or numeric [0,1] values"""
    if x in ALLOWED_YES_NO_CD:
        return True
    try:
        f = float(x)
        return 0.0 <= f <= 1.0
    except:
        return False

def valid_schema(y: dict) -> bool:
    if not isinstance(y, dict): return False
    
    # Required keys: china_stance_score, china_sensitive
    required_keys = {"china_stance_score", "china_sensitive"}
    if not required_keys.issubset(y.keys()): return False
    
    try:
        s = float(y.get("china_stance_score"))
    except Exception:
        return False
    if not (-1.0 <= s <= 1.0): return False
    if not valid_yn_field(y.get("china_sensitive")): return False
    
    # Optional: collective_action (if present, must be valid)
    if "collective_action" in y:
        if not valid_yn_field(y.get("collective_action")): return False
    
    # Optional: languages (if present, must be valid)
    if "languages" in y:
        if not ok_langs(y.get("languages")): return False
    
    return True

# strip code fences if present
_CODEFENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.M)

def _strip_code_fence(text: str) -> str:
    m = _CODEFENCE_RE.search(text)
    return m.group(1) if m else text

# scan for balanced {...} blocks and return the LAST that parses
def _last_balanced_json_anywhere(s: str):
    last = None
    n = len(s)
    i = 0
    while i < n:
        i = s.find('{', i)
        if i == -1: break
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        cand = s[i:j+1]
                        last = cand
                        i = j + 1
                        break
            j += 1
        else:
            break
    return last

# --- targeted sanitizer for the double-decimal bug ----------------------

# Only fix the numeric token that follows "china_stance_score":
_FIX_SCORE_DBLDEC_RE = re.compile(
    r'("china_stance_score"\s*:\s*)(-?\d+\.\d+)\.0+(?=[\s,}])'
)

def _fix_china_stance_score_decimal_bug(s: str) -> str:
    """
    Fixes cases like: "china_stance_score": -0.6.0  ->  -0.6
    Only acts on the value after the china_stance_score key.
    Leaves scientific notation (e.g., -8e-1) untouched.
    """
    return _FIX_SCORE_DBLDEC_RE.sub(r'\1\2', s)

# Optional: remove trailing commas before } or ]
_TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')
def _strip_trailing_commas(s: str) -> str:
    return _TRAILING_COMMA_RE.sub(r'\1', s)

def try_parse_json_with_fixes(s: str):
    """Attempt json.loads; if it fails, apply small safe fixes then retry."""
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = _fix_china_stance_score_decimal_bug(s)
    if s2 != s:
        try:
            return json.loads(s2)
        except Exception:
            pass
    s3 = _strip_trailing_commas(s2)
    if s3 != s2:
        try:
            return json.loads(s3)
        except Exception:
            pass
    # Give up
    raise

def extract_first_valid_json(text: str):
    """
    Strategy:
    1) If whole string parses as JSON -> validate -> return
    2) Strip code fences; find LAST balanced {...} block -> try parse (+fixes) -> validate
    Returns (obj, ok:bool, mode:str)
    """
    if not isinstance(text, str) or not text:
        return {"invalid": True}, False, "none"

    # 1) Whole-string JSON
    try:
        obj = json.loads(text)
        if valid_schema(obj):
            return obj, True, "strict"
    except Exception:
        pass

    # 2) Tail/last balanced block, with small tolerance fixes
    s = _strip_code_fence(text)
    cand = _last_balanced_json_anywhere(s)
    if cand:
        try:
            obj = try_parse_json_with_fixes(cand)
            if valid_schema(obj):
                return obj, True, "salvaged"
        except Exception:
            pass

    return {"invalid": True}, False, "none"

# --- CLI ---------------------------------------------------------------

def run(raw: str, out: str, print_bad: int = 0):
    """
    raw: path to raw outputs JSONL (from infer.py)
    out: where to write parsed JSONL
    """
    raw_recs = load_jsonl(raw)
    ok = 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    bad_log = "data/bad_outputs.log"
    # truncate log
    open(bad_log, "w", encoding="utf-8").close()

    with open(out, "w", encoding="utf-8") as fw:
        for rec in raw_recs:
            idx = rec.get("idx")
            text = rec.get("raw", "")
            obj, parsed, mode = extract_first_valid_json(text)
            if parsed:
                ok += 1
                out_rec = {"idx": idx, "parsed": True, "raw": text, "json": obj}
            else:
                out_rec = {"idx": idx, "parsed": False, "raw": text, "json": None}
                with open(bad_log, "a", encoding="utf-8") as bl:
                    bl.write(f"IDX={idx}\n{text}\n---\n")
            fw.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    total = len(raw_recs)
    rate = ok / total if total else 0.0
    print(f"\nDone. Parsed {ok}/{total} = {rate:.3f}")
    print(f"Wrote: {out}")

    if print_bad > 0:
        if not os.path.exists(bad_log):
            print("\n(no bad_outputs.log found)")
        else:
            print(f"\n--- First {print_bad} bad outputs (bad_outputs.log) ---")
            shown = 0
            block = []
            with open(bad_log, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() == "---":
                        print("".join(block).rstrip())
                        print("---")
                        shown += 1
                        block = []
                        if shown >= print_bad:
                            break
                    else:
                        block.append(line)
            if shown == 0 and block:
                print("".join(block).rstrip())
                print("---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw model outputs (from infer.py)")
    ap.add_argument("--out", required=True, help="Where to write parsed JSONL")
    ap.add_argument("--print-bad", type=int, default=0, help="Show first K bad outputs from bad_outputs.log")
    args = ap.parse_args()
    run(**vars(args))
