#!/usr/bin/env python3
"""
compare_wide.py — Build a wide CSV for side-by-side model vs human comparisons.

Supports both standard (3-dimension) and comprehensive (11-dimension) label formats.
Auto-detects format from validation data.

Standard Format Columns:
  - stance: <model>_stance, human_stance
  - china_sensitive: <model>_sensitive, human_sensitive
  - collective_action: <model>_collective, human_collective

Comprehensive Format Columns:
  - china_related: <model>_china_related, human_china_related
  - stance_gov: <model>_stance_gov, human_stance_gov
  - stance_culture: <model>_stance_culture, human_stance_culture
  - stance_tech: <model>_stance_tech, human_stance_tech
  - china_sensitive: <model>_sensitive, human_sensitive
  - collective_action: <model>_collective, human_collective
  - hate_speech: <model>_hate, human_hate
  - harmful_content: <model>_harmful, human_harmful
  - news_segments: <model>_news, human_news
  - inauthentic_content: <model>_inauthentic, human_inauthentic
  - derivative_content: <model>_derivative, human_derivative

Final column is always: transcript

Input predictions must be the parsed outputs from parse.py:
  {"idx": ..., "parsed": true, "json": {...}}

Usage:
  python compare_wide.py \
    --val-file data/val_BAL.jsonl \
    --out out/compare_wide.csv \
    out/preds_base.parsed.jsonl out/preds_ft.parsed.jsonl

"""

import os
import re
import json
import argparse
from typing import List, Dict, Any

import pandas as pd
from json_utils import REQ_KEYS, ALLOWED_YES_NO_CD, load_jsonl, gold_of, user_text

# ----------------- I/O helpers -----------------
# load_jsonl, gold_of, user_text now imported from json_utils

def load_val_examples(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    if not data:
        raise RuntimeError(f"{val_path}: empty file")
    # Basic validation - just ensure we have messages
    for i, ex in enumerate(data):
        if "messages" not in ex:
            raise RuntimeError(f"{val_path}: example {i} missing 'messages' field")
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
    return "invalid"  # leave empty if invalid/missing

def norm_stance(x: Any) -> str:
    """Normalize stance labels (pro/anti/neutral)"""
    if isinstance(x, str):
        t = x.strip().lower()
        # Accept variations
        if t in {"pro", "anti", "neutral", "neutral/unclear"}:
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

    # Detect format from first example
    first_gold = gold_of(val[0])
    is_comprehensive = "china_related" in first_gold
    
    if is_comprehensive:
        print("[info] Detected comprehensive label format (11 dimensions)")
    else:
        print("[info] Detected standard label format (3 dimensions)")

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

    # Build rows based on format
    if is_comprehensive:
        rows, cols = build_comprehensive_rows(val, model_names, model_maps)
    else:
        rows, cols = build_standard_rows(val, model_names, model_maps)

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out, index=False)
    print(f"Wrote wide comparison CSV: {out}")

def build_standard_rows(val: List[dict], model_names: List[str], model_maps: List[Dict[int, dict]]):
    """Build rows for standard 3-dimension format"""
    rows = []
    n = len(val)
    
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

    # Column order
    stance_cols = [f"{m}_stance" for m in model_names] + ["human_stance"]
    sens_cols   = [f"{m}_sensitive" for m in model_names] + ["human_sensitive"]
    coll_cols   = [f"{m}_collective" for m in model_names] + ["human_collective"]
    cols = ["idx"] + stance_cols + sens_cols + coll_cols + ["transcript"]
    
    return rows, cols

def build_comprehensive_rows(val: List[dict], model_names: List[str], model_maps: List[Dict[int, dict]]):
    """Build rows for comprehensive 11-dimension format"""
    rows = []
    n = len(val)
    
    for idx in range(n):
        ex = val[idx]
        g  = gold_of(ex)
        rec = {"idx": idx}

        # china_related (yes/no)
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_related")
            rec[f"{mname}_china_related"] = norm_yn_cd(pv)
        rec["human_china_related"] = norm_yn_cd(g.get("china_related"))

        # stance dimensions (pro/anti/neutral)
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_ccp_government")
            rec[f"{mname}_stance_gov"] = norm_stance(pv)
        rec["human_stance_gov"] = norm_stance(g.get("china_ccp_government"))

        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_people_culture")
            rec[f"{mname}_stance_culture"] = norm_stance(pv)
        rec["human_stance_culture"] = norm_stance(g.get("china_people_culture"))

        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_technology_development")
            rec[f"{mname}_stance_tech"] = norm_stance(pv)
        rec["human_stance_tech"] = norm_stance(g.get("china_technology_development"))

        # china_sensitive
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("china_sensitive")
            rec[f"{mname}_sensitive"] = norm_yn_cd(pv)
        rec["human_sensitive"] = norm_yn_cd(g.get("china_sensitive"))

        # collective_action
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("collective_action")
            rec[f"{mname}_collective"] = norm_yn_cd(pv)
        rec["human_collective"] = norm_yn_cd(g.get("collective_action"))

        # hate_speech
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("hate_speech")
            rec[f"{mname}_hate"] = norm_yn_cd(pv)
        rec["human_hate"] = norm_yn_cd(g.get("hate_speech"))

        # harmful_content
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("harmful_content")
            rec[f"{mname}_harmful"] = norm_yn_cd(pv)
        rec["human_harmful"] = norm_yn_cd(g.get("harmful_content"))

        # news_segments
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("news_segments")
            rec[f"{mname}_news"] = norm_yn_cd(pv)
        rec["human_news"] = norm_yn_cd(g.get("news_segments"))

        # inauthentic_content
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("inauthentic_content")
            rec[f"{mname}_inauthentic"] = norm_yn_cd(pv)
        rec["human_inauthentic"] = norm_yn_cd(g.get("inauthentic_content"))

        # derivative_content
        for mname, mmap in zip(model_names, model_maps):
            pv = mmap.get(idx, {}).get("derivative_content")
            rec[f"{mname}_derivative"] = norm_yn_cd(pv)
        rec["human_derivative"] = norm_yn_cd(g.get("derivative_content"))

        # transcript last
        rec["transcript"] = user_text(ex)
        rows.append(rec)

    # Column order: idx | all dimensions grouped | transcript
    china_related_cols = [f"{m}_china_related" for m in model_names] + ["human_china_related"]
    stance_gov_cols = [f"{m}_stance_gov" for m in model_names] + ["human_stance_gov"]
    stance_culture_cols = [f"{m}_stance_culture" for m in model_names] + ["human_stance_culture"]
    stance_tech_cols = [f"{m}_stance_tech" for m in model_names] + ["human_stance_tech"]
    sens_cols = [f"{m}_sensitive" for m in model_names] + ["human_sensitive"]
    coll_cols = [f"{m}_collective" for m in model_names] + ["human_collective"]
    hate_cols = [f"{m}_hate" for m in model_names] + ["human_hate"]
    harmful_cols = [f"{m}_harmful" for m in model_names] + ["human_harmful"]
    news_cols = [f"{m}_news" for m in model_names] + ["human_news"]
    inauthentic_cols = [f"{m}_inauthentic" for m in model_names] + ["human_inauthentic"]
    derivative_cols = [f"{m}_derivative" for m in model_names] + ["human_derivative"]
    
    cols = (["idx"] + china_related_cols + stance_gov_cols + stance_culture_cols + 
            stance_tech_cols + sens_cols + coll_cols + hate_cols + harmful_cols + 
            news_cols + inauthentic_cols + derivative_cols + ["transcript"])
    
    return rows, cols

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True, help="Validation JSONL with gold labels in final assistant message.")
    ap.add_argument("--out", required=True, help="Output CSV path (wide comparison).")
    ap.add_argument("preds", nargs="+", help="One or more parsed prediction files (from parse.py).")
    args = ap.parse_args()
    main(**vars(args))
