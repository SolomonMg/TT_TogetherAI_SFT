#!/usr/bin/env python3
"""
parse.py — extract the final JSON object from model outputs produced by infer.py.

Usage:
  python parse.py \
    --raw out/preds_base.raw.jsonl \
    --out out/preds_base.parsed.jsonl \
    --print-bad 5

Input (from infer.py) is a JSONL where each line contains at least:
  {"idx": <int>, "raw": "<model_text_or_tokens_rendered>" ...}

Output JSONL lines look like:
  {"idx": 0, "parsed": true, "raw": "...", "json": {...}}
or, on failure:
  {"idx": 0, "parsed": false, "raw": "...", "json": null}
"""

import argparse, json, re, os, sys
from typing import Tuple

# --- Regex helpers (code-fence, tail JSON, fallback to last balanced) ---
_CODEFENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.M)
_TAIL_JSON_RE = re.compile(r"(\{[\s\S]*\})\s*$", re.S)

ALLOWED_YES_NO_CD = {"yes","no","cannot_determine"}
ALLOWED_LANGS = {"english","mandarin","spanish","other","no_language"}
REQ_KEYS = {"china_stance_score","china_sensitive","collective_action","languages"}

def _strip_code_fence_or_return(s: str) -> str:
    m = _CODEFENCE_RE.search(s)
    return m.group(1) if m else s

def _valid_schema(obj: dict) -> bool:
    try:
        if set(obj.keys()) != REQ_KEYS:
            return False
        s = float(obj["china_stance_score"])
        if not (-1.0 <= s <= 1.0):
            return False
        if obj["china_sensitive"] not in ALLOWED_YES_NO_CD:
            return False
        if obj["collective_action"] not in ALLOWED_YES_NO_CD:
            return False
        langs = obj["languages"]
        if not isinstance(langs, list) or not langs or any(l not in ALLOWED_LANGS for l in langs):
            return False
        if "no_language" in langs and len(langs) != 1:
            return False
        return True
    except Exception:
        return False

def _try_whole_json(s: str):
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) and _valid_schema(obj) else None
    except Exception:
        return None

def _try_tail_json(s: str):
    m = _TAIL_JSON_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) and _valid_schema(obj) else None
    except Exception:
        return None

def _last_balanced_json_anywhere(s: str):
    last = None
    n = len(s)
    i = 0
    while i < n:
        i = s.find('{', i)
        if i == -1:
            break
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
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict) and _valid_schema(obj):
                                last = obj
                        except Exception:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break
    return last

def extract_best_json(text: str) -> Tuple[dict, bool]:
    """Try whole-string, then tail JSON (after stripping code-fence), then last balanced block."""
    if not isinstance(text, str):
        return {"invalid": True}, False

    obj = _try_whole_json(text)
    if obj is not None:
        return obj, True

    s = _strip_code_fence_or_return(text)

    obj = _try_tail_json(s)
    if obj is not None:
        return obj, True

    obj = _last_balanced_json_anywhere(s)
    if obj is not None:
        return obj, True

    return {"invalid": True}, False

# --- IO helpers ---
def _iter_jsonl(path, limit=0):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # If a line isn't JSON, still pass it through as raw text
                yield {"idx": i, "raw": line}

def _first_text_field(rec: dict) -> str:
    """
    Be generous about where the model text might live.
    Prefer 'raw', then 'content', then 'output_text', else stringify.
    """
    for k in ("raw", "content", "output_text", "text"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    # Sometimes the entire record is already the raw string
    if isinstance(rec, str):
        return rec
    # Last resort: dump record without breaking Unicode
    try:
        return json.dumps(rec, ensure_ascii=False)
    except Exception:
        return str(rec)

# --- main logic ---
def run(raw_path: str, out_path: str, fail_log: str = "bad_outputs.log",
        print_bad: int = 0, limit: int = 0) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = 0
    parsed = 0
    bad_blocks = []

    with open(out_path, "w", encoding="utf-8") as out_f, \
         open(fail_log, "w", encoding="utf-8") as bad_f:

        for rec in _iter_jsonl(raw_path, limit=limit):
            n += 1
            idx = rec.get("idx", n-1)
            raw = _first_text_field(rec)

            obj, ok = extract_best_json(raw)
            if ok:
                parsed += 1
                out_f.write(json.dumps({"idx": idx, "parsed": True, "raw": raw, "json": obj}, ensure_ascii=False) + "\n")
            else:
                out_f.write(json.dumps({"idx": idx, "parsed": False, "raw": raw, "json": None}, ensure_ascii=False) + "\n")
                block = f"IDX={idx}\n{raw}\n---\n"
                bad_f.write(block)
                if len(bad_blocks) < max(0, print_bad):
                    bad_blocks.append(block)

    print(f"Done. Parsed {parsed}/{n} = {parsed/(n or 1):.3f}")
    print(f"Wrote: {out_path}")
    if print_bad and bad_blocks:
        print(f"\n--- First {min(print_bad, len(bad_blocks))} bad outputs ({fail_log}) ---")
        for b in bad_blocks:
            print(b, end="")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Accept BOTH flags, map to the same dest so run() sees raw_path
    ap.add_argument("--raw", dest="raw_path", help="Path to raw model outputs (.jsonl)")
    ap.add_argument("--raw-path", dest="raw_path", help="Path to raw model outputs (.jsonl)")
    ap.add_argument("--out", dest="out_path", help="Where to write parsed outputs (.jsonl)")
    ap.add_argument("--out-path", dest="out_path", help="Where to write parsed outputs (.jsonl)")
    ap.add_argument("--fail-log", default="bad_outputs.log", help="File to write raw bad outputs")
    ap.add_argument("--print-bad", type=int, default=0, help="Also echo the first K bad outputs to stdout")
    ap.add_argument("--limit", type=int, default=0, help="Only parse first N lines (0 = all)")
    args = ap.parse_args()

    if not args.raw_path:
        ap.error("Please provide --raw or --raw-path")
    if not args.out_path:
        ap.error("Please provide --out or --out-path")

    run(**vars(args))
