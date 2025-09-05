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
  --raw out/preds_minimal.raw.jsonl \
  --out out/preds_minimal.parsed.jsonl \
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
    
    # Optional fields that should be numeric [0,1] if present
    optional_numeric_fields = [
        "collective_action", "inauthentic_content", "hate_speech", 
        "harmful_content", "news_segments"
    ]
    
    for field in optional_numeric_fields:
        if field in y:
            if not valid_yn_field(y.get(field)):
                return False
    
    # Optional: languages (if present, must be valid)
    if "languages" in y:
        if not ok_langs(y.get("languages")): return False
    
    # Ignore deprecated/unknown fields (like derivative_content) 
    # as long as required fields are valid
    
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

# --- enhanced sanitizers for common parsing issues ----------------------

# Fix double decimals in any numeric field (e.g., 0.0.0 -> 0.0)
_FIX_DOUBLE_DECIMAL_RE = re.compile(r'(\d+\.\d+)\.0+(?=[\s,}\]])')

# Fix malformed keys like "china_stance_score:0" -> "china_stance_score":0
_FIX_MALFORMED_KEY_RE = re.compile(r'"([^"]+):([^"]*)"(\s*:\s*)"([^"]*)"')

# Fix missing quotes on string values like "china_stance_score:0,"
_FIX_UNQUOTED_KEY_VALUE_RE = re.compile(r'"([^"]+):([^"]*)"')

# Fix truncated JSON like "collective_action": or ending with incomplete values
_FIX_TRUNCATED_VALUE_RE = re.compile(r':\s*$|:\s*,')

# Fix space-separated values (just numbers) like "0.0 0.5 0.2 0.0 0.0 0.0 0.0"
_SPACE_SEPARATED_VALUES_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?){6})\s*$')

# Fix missing quotes in key names like harmful0 -> "harmful_content"
_FIX_MALFORMED_KEYS_RE = re.compile(r'([,{]\s*)(harmful0|hate0|news0|inauthentic0)(\s*:)')

# Fix incomplete JSON objects ending with trailing characters like {"...,
_INCOMPLETE_JSON_RE = re.compile(r'^\{[^}]*,\s*$')

def _fix_double_decimal_bug(s: str) -> str:
    """Fix double decimals like 0.0.0 -> 0.0 and scientific notation like 0.4.0e-1 -> 0.04"""
    # Fix scientific notation like 0.4.0e-1 -> 0.04
    s = re.sub(r'(\d+\.\d+)\.0e-1', lambda m: str(float(m.group(1)) * 0.1), s)
    
    # Fix regular double decimals like 0.0.0 -> 0.0
    s = _FIX_DOUBLE_DECIMAL_RE.sub(r'\1', s)
    
    return s

def _fix_malformed_keys(s: str) -> str:
    """
    Fix malformed keys like:
    - "china_stance_score:0":"0" -> "china_stance_score":0
    - "china_stance_score:0," -> "china_stance_score":0,
    - "inauthentic:0," -> "inauthentic_content":0,
    """
    # Fix specific pattern: "china_stance_score:0":"0" -> "china_stance_score":0
    s = re.sub(r'"china_stance_score:0":"0"', '"china_stance_score":0', s)
    
    # Fix pattern like: "key:value" where the whole thing is in quotes
    # This handles cases like "china_stance_score:0,"
    s = re.sub(r'"([a-z_]+):([^"]*)"(\s*:\s*)"([^"]*)"', r'"\1":\2', s)
    s = re.sub(r'"([a-z_]+):([^"]*)"(?=\s*[,}])', r'"\1":\2', s)
    
    # Fix missing key name quotes and incomplete key names
    s = re.sub(r'"inauthentic:([^"]*)"', r'"inauthentic_content":\1', s)
    s = re.sub(r'"hate:([^"]*)"', r'"hate_speech":\1', s)
    s = re.sub(r'"harmful:([^"]*)"', r'"harmful_content":\1', s)
    s = re.sub(r'"news:([^"]*)"', r'"news_segments":\1', s)
    
    # Fix bare key names without quotes followed by colon value
    s = re.sub(r'([,{]\s*)inauthentic:(\d)', r'\1"inauthentic_content":\2', s)
    s = re.sub(r'([,{]\s*)hate:(\d)', r'\1"hate_speech":\2', s)
    s = re.sub(r'([,{]\s*)harmful:(\d)', r'\1"harmful_content":\2', s)
    s = re.sub(r'([,{]\s*)news:(\d)', r'\1"news_segments":\2', s)
    
    # Fix the specific malformed pattern: "inauthentic:0,
    s = re.sub(r'"inauthentic:0,', r'"inauthentic_content":0,', s)
    
    # Fix common typos in key names
    s = re.sub(r'([,{]\s*)(harmful0)(\s*:)', r'\1"harmful_content"\3', s)
    s = re.sub(r'([,{]\s*)(hate0)(\s*:)', r'\1"hate_speech"\3', s)
    s = re.sub(r'([,{]\s*)(news0)(\s*:)', r'\1"news_segments"\3', s)
    s = re.sub(r'([,{]\s*)(inauthentic0)(\s*:)', r'\1"inauthentic_content"\3', s)
    
    # Fix missing key names like just "china_stance" instead of "china_stance_score"
    s = re.sub(r'"china_stance"(\s*:)', r'"china_stance_score"\1', s)
    
    return s

def _fix_truncated_json(s: str) -> str:
    """Fix truncated JSON by removing incomplete trailing parts and attempting to close"""
    # Remove trailing colons with no values
    s = _FIX_TRUNCATED_VALUE_RE.sub('', s)
    
    # Handle various truncation patterns
    # 1. Ends with just a number (missing closing brace)
    if re.match(r'^\{[^}]*\d+\s*$', s):
        s = s.strip() + '}'
    
    # 2. Ends with comma (remove comma and close)
    elif re.match(r'^\{[^}]*,\s*$', s):
        s = s.rstrip(',').strip() + '}'
    
    # 3. Ends with incomplete key-value pair like "key":value (missing closing)
    elif re.match(r'^\{[^}]*"[^"]*":\d+\.?\d*\s*$', s):
        s = s.strip() + '}'
    
    return s

def _convert_space_separated_to_json(s: str) -> str:
    """Convert space-separated values like '0.0 0.5 0.2 0.0 0.0 0.0 0.0' to proper JSON"""
    match = _SPACE_SEPARATED_VALUES_RE.match(s)
    if match:
        values = match.group(1).split()
        if len(values) == 7:
            keys = ["china_stance_score", "china_sensitive", "collective_action", 
                   "inauthentic_content", "hate_speech", "harmful_content", "news_segments"]
            json_pairs = [f'"{key}":{val}' for key, val in zip(keys, values)]
            return '{' + ','.join(json_pairs) + '}'
    return s

def _fix_missing_quotes_in_strings(s: str) -> str:
    """Fix missing quotes around '...' values"""
    return re.sub(r'"\.\.\."|"…"', '"truncated"', s)

# Optional: remove trailing commas before } or ]
_TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')
def _strip_trailing_commas(s: str) -> str:
    return _TRAILING_COMMA_RE.sub(r'\1', s)

def try_parse_json_with_fixes(s: str):
    """Attempt json.loads; if it fails, apply comprehensive fixes then retry."""
    # First, try parsing as-is
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Apply fixes in order of frequency/likelihood
    original_s = s
    
    # 1. Check if it's space-separated values and convert to JSON
    s = _convert_space_separated_to_json(s)
    if s != original_s:
        try:
            return json.loads(s)
        except Exception:
            pass
    
    # 2. Fix double decimals (most common issue)
    s = _fix_double_decimal_bug(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 3. Fix malformed keys and values (do this early as it catches many patterns)
    s = _fix_malformed_keys(s)
    
    # 3a. Also fix this specific edge case that wasn't caught above
    s = re.sub(r'"inauthentic:0,"', r'"inauthentic_content":0,"', s)
    
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 4. Fix truncated JSON
    s = _fix_truncated_json(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 5. Fix missing quotes in string values
    s = _fix_missing_quotes_in_strings(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 6. Remove trailing commas (final cleanup)
    s = _strip_trailing_commas(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Give up - couldn't fix it
    raise

def extract_first_valid_json(text: str):
    """
    Strategy:
    1) If whole string parses as JSON -> validate -> return
    2) Strip code fences; find LAST balanced {...} block -> try parse (+fixes) -> validate
    3) Look for incomplete JSON at the end and try to complete it
    Returns (obj, ok:bool, mode:str)
    """
    if not isinstance(text, str) or not text:
        return {"invalid": True}, False, "none"

    # 1) Whole-string JSON (try with fixes first)
    try:
        obj = try_parse_json_with_fixes(text.strip())
        if valid_schema(obj):
            return obj, True, "strict"
    except Exception:
        pass

    # 2) Strip code fences and try again
    s = _strip_code_fence(text)
    try:
        obj = try_parse_json_with_fixes(s.strip())
        if valid_schema(obj):
            return obj, True, "codefence_stripped"
    except Exception:
        pass

    # 3) Find last balanced {...} block
    cand = _last_balanced_json_anywhere(s)
    if cand:
        try:
            obj = try_parse_json_with_fixes(cand)
            if valid_schema(obj):
                return obj, True, "balanced_block"
        except Exception:
            pass

    # 4) Look for JSON-like patterns at the end of verbose output
    # Find patterns like: ..."china_stance_score":0.6,"china_sensitive":1.0
    json_pattern = re.search(r'\{[^{}]*"china_stance_score"[^{}]*\}(?=[^{}]*$)', text, re.DOTALL)
    if json_pattern:
        try:
            obj = try_parse_json_with_fixes(json_pattern.group(0))
            if valid_schema(obj):
                return obj, True, "pattern_match"
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
            meta_id = rec.get("meta_id", str(idx))  # Extract meta_id, fallback to idx
            text = rec.get("raw", "")
            obj, parsed, mode = extract_first_valid_json(text)
            if parsed:
                ok += 1
                out_rec = {"idx": idx, "meta_id": meta_id, "parsed": True, "raw": text, "json": obj}
            else:
                out_rec = {"idx": idx, "meta_id": meta_id, "parsed": False, "raw": text, "json": None}
                with open(bad_log, "a", encoding="utf-8") as bl:
                    bl.write(f"IDX={idx} META_ID={meta_id}\n{text}\n---\n")
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
