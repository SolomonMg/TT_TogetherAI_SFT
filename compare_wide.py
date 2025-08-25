#!/usr/bin/env python3
"""
compare_wide.py — Build a wide CSV for side-by-side model vs human comparisons.

Each row = one validation example (idx).
Columns are grouped by target:

  stance:
    <model1>_stance, <model2>_stance, ..., human_stance

  china_sensitive:
    <model1>_sensitive, <model2>_sensitive, ..., human_sensitive

  collective_action:
    <model1>_collective, <model2>_collective, ..., human_collective

The final column is: transcript

Input predictions must be the parsed outputs from parse.py:
  {"idx": ..., "parsed": true, "json": {...}}

Usage:
  python compare_wide.py \
    --val-file data/val_BAL.jsonl \
    --out out/compare_wide.csv \
    out/preds_base.parsed.jsonl out/preds_ft.parsed.jsonl

python compare_wide.py \
    --val-file data/val_BAL.jsonl \
    --out out/compare_wide.csv \
    out/preds_llama3.1-70B.parsed.jsonl \
    out/preds_llama3.1-70B-SFT.parsed.jsonl \
    out/preds_llama3.3-70B.parsed.jsonl \
    out/preds_gpt-oss-120b.parsed.jsonl

"""

import os
import re
import json
import argparse
from typing import List, Dict, Any

import pandas as pd

REQ_KEYS = {"china_stance_score","china_sensitive","collective_action","languages"}
ALLOWED_YES_NO_CD = {"yes", "no", "cannot_determine"}

# ----------------- I/O helpers -----------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = re.sub(r'[\u2028\u2029]', '', line).rstrip("\r\n")
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}: line {ln} invalid JSON: {e}")
    return rows

def load_val_examples(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    # quick schema sanity check
    for i, ex in enumerate(data):
        g = gold_of(ex)
        if set(g.keys()) != REQ_KEYS:
            raise RuntimeError(f"{val_path}: example {i} gold keys {set(g.keys())} != {REQ_KEYS}")
    return data

def gold_of(example: dict) -> dict:
    return json.loads(example["messages"][-1]["content"])

def user_text(ex: dict) -> str:
    parts = [m.get("content","") for m in ex.get("messages", []) if m.get("role") == "user"]
    return "\n\n".join(parts).strip()

def load_parsed_preds(pred_path: str) -> Dict[int, dict]:
    """
    parse.py rows look like: {"idx": 0, "parsed": true, "json": {...}}
    Return a mapping: idx -> prediction dict (only for parsed==True).
    """
    out = {}
    for row in load_jsonl(pred_path):
        if not isinstance(row, dict):
            continue
        if not row.get("parsed", False):
            continue
        if "idx" not in row:
            continue
        pj = row.get("json")
        if isinstance(pj, dict):
            out[int(row["idx"])] = pj
    return out

# ----------------- normalizers -----------------

def to_float(x: Any):
    try:
        v = float(x)
        # clamp to [-1,1] (consistent with the task schema)
        if v < -1: v = -1.0
        if v >  1: v =  1.0
        return v
    except Exception:
        return None

def norm_yn_cd(x: Any) -> str:
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ALLOWED_YES_NO_CD:
            return t
    return ""  # leave empty if invalid/missing

def sanitize_model_name(path: str) -> str:
    name = os.path.basename(path)
    # drop multi-extensions like .parsed.jsonl
    name = re.sub(r"\.jsonl$", "", name)
    name = re.sub(r"\.parsed$", "", name)
    name = re.sub(r"[^\w\.-]+", "_", name).strip("_")
    return name or "model"

# ----------------- main -----------------

def main(val_file: str, out: str, preds: List[str]):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    val = load_val_examples(val_file)
    n = len(val)

    # load all predictions, keep model name order stable
    model_maps: List[Dict[int, dict]] = []
    model_names: List[str] = []
    name_counts = {}

    for p in preds:
        base = sanitize_model_name(p)
        if base in name_counts:
            name_counts[base] += 1
            base = f"{base}_{name_counts[base]}"
        else:
            name_counts[base] = 1
        model_names.append(base)
        model_maps.append(load_parsed_preds(p))

    # Build rows
    rows = []
    for idx in range(n):
        ex = val[idx]
        g  = gold_of(ex)

        rec = {"idx": idx}

        # stance block: all models, then human
        for mname, mmap in zip(model_names, model_maps):
            ps = mmap.get(idx, {}).get("china_stance_score")
            rec[f"{mname}_stance"] = to_float(ps)
        rec["human_stance"] = to_float(g.get("china_stance_score"))

        # china_sensitive block: all models, then human
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_sensitive")
            rec[f"{mname}_sensitive"] = norm_yn_cd(pv)
        rec["human_sensitive"] = norm_yn_cd(g.get("china_sensitive"))

        # collective_action block: all models, then human
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("collective_action")
            rec[f"{mname}_collective"] = norm_yn_cd(pv)
        rec["human_collective"] = norm_yn_cd(g.get("collective_action"))

        # transcript last
        rec["transcript"] = user_text(ex)

        rows.append(rec)

    # Column order: idx | stance group | sensitive group | collective group | transcript
    stance_cols = [f"{m}_stance" for m in model_names] + ["human_stance"]
    sens_cols   = [f"{m}_sensitive" for m in model_names] + ["human_sensitive"]
    coll_cols   = [f"{m}_collective" for m in model_names] + ["human_collective"]

    cols = ["idx"] + stance_cols + sens_cols + coll_cols + ["transcript"]

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out, index=False)
    print(f"Wrote wide comparison CSV: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True, help="Validation JSONL with gold labels in final assistant message.")
    ap.add_argument("--out", required=True, help="Output CSV path (wide comparison).")
    ap.add_argument("preds", nargs="+", help="One or more parsed prediction files (from parse.py).")
    args = ap.parse_args()
    main(**vars(args))
