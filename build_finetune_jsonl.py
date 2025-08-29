#!/usr/bin/env python3
"""
build_finetune_jsonl.py

Create TRAIN/VAL JSONL for SFT with top-level:
  { "meta_id": "...", "messages": [ {role, content}, ... ] }

Gold (assistant) JSON uses EXACT keys:
  - china_stance_score: float in [-1, 1]
  - china_sensitive:    float in [0, 1]       (default output mode)
  - collective_action:  float in [0, 1]       (default output mode)
  - languages:          list from {'english','mandarin','spanish','other','no_language'}
                        ('no_language' must be alone if present)

Optionally emit categorical labels with:
  --yn-output label  (then you can use --yn-thresh and --cd-band)

Inputs (Mode 1: Separate files)
  --train-csv / --val-csv : balanced label CSVs (contain raw numeric columns)
                            Required columns (case-insensitive):
                              meta_id,
                              china_stance_score,
                              sensitive, collective_action  (numeric)
                              english, mandarin, spanish, other_lang, no_language  (numeric)
  --text-file             : Parquet or CSV with meta_id, subtitle, meta_desc

Inputs (Mode 2: Single file with text only - for inference)
  --single-text-file      : Single file with meta_id, description, transcript
  --out-single            : Single output JSONL (no train/val split)

Inputs (Mode 3: Single file with labels - for training)  
  --single-labeled-file   : Single file with meta_id, china_stance_score, sensitive, 
                            collective_action, languages, subtitle, meta_desc
  --single-train-frac     : Fraction for train split (default 0.8)
  --out-single-train      : Train output JSONL
  --out-single-val        : Val output JSONL

Outputs
  --out-train / --out-val : JSONL, one object per row:
      {
        "meta_id": "...",
        "messages": [
          {"role":"system","content": "..."},
          {"role":"user","content":"TRANSCRIPT... DESCRIPTION..."},
          {"role":"assistant","content":"{\"china_stance_score\":...,\"china_sensitive\":...,\"collective_action\":...,\"languages\":[...] }"}
        ]
      }
"""

import os
import re
import json
import argparse
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from json_utils import (
    load_table, norm_cols, to_str_meta, clamp11, clamp01, safe_text, 
    write_jsonl, yesno_to_label, ALLOWED_YES_NO_CD
)

LANG_KEYS_IN  = ["english","mandarin","spanish","other_lang","no_language"]
LANG_KEYS_OUT = ["english","mandarin","spanish","other","no_language"]  # map other_lang -> other

SYSTEM_PROMPT_FLOAT = (
    "You are an annotator for short videos. Based on the transcript and description, "
    "Follow the CODEBOOK and output format exactly.\n\n"
    "CODEBOOK — DEFINITIONS & TASKS\n"
    "china_sensitive - does the video include mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
    "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
    "of Xi Jinping or the CCP; corruption in China. Answer with a float in [0,1], where 0 is 'no' and 1 is 'yes'.\n\n"
    "china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look."
    "Use the full range; 0 = neutral/unclear.\n\n"
    "collective_action - Does this video encourage or portray collective action? A float in [0,1], where 0 is 'no' and 1 is 'yes'.\n\n"
    "Base scores on explicit evidence."
    "Return ONLY a minified JSON object with exactly those keys and nothing else."
)

SYSTEM_PROMPT_LABEL = (
    "You are an annotator for short videos. Based on the transcript and description, "
    "Follow the CODEBOOK and output format exactly.\n\n"
    "CODEBOOK — DEFINITIONS & TASKS\n"
    "china_sensitive - does the video include mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
    "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
    "of Xi Jinping or the CCP; corruption in China. Answer one of {'yes','no','cannot_determine'}.\n\n"
    "china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look."
    "Use the full range; 0 = neutral/unclear.\n\n"
    "collective_action - Does this video encourage or portray collective action? Answer one of {'yes','no','cannot_determine'}.\n\n"
    "Base scores on explicit evidence."
    "Return ONLY a minified JSON object with exactly those four keys and nothing else."
    "- china_stance_score: float from -1 (very negative about China) to +1 (very positive)\n"
    "- china_sensitive: one of {'yes','no','cannot_determine'}\n"
    "- collective_action: one of {'yes','no','cannot_determine'}\n"
    "Return ONLY a minified JSON object with exactly those keys and nothing else."
)

# Utility functions now imported from json_utils

def build_languages(row: pd.Series, lang_thresh: float) -> List[str]:
    vals = {k: clamp01(row.get(k)) for k in LANG_KEYS_IN}
    if (vals.get("no_language") or 0.0) >= lang_thresh:
        return ["no_language"]
    out = []
    for k_in, k_out in zip(LANG_KEYS_IN, LANG_KEYS_OUT):
        if k_in == "no_language":
            continue
        v = vals.get(k_in)
        if v is not None and v >= lang_thresh:
            out.append(k_out)
    return out or ["no_language"]

def labelize(p: Optional[float], yn_thresh: float, band: Optional[Tuple[float,float]]) -> str:
    if p is None:
        return "cannot_determine" if band else "no"
    if band and (band[0] <= p <= band[1]):
        return "cannot_determine"
    return "yes" if p >= yn_thresh else "no"

def make_messages(user_text: str, gold_json: dict, yn_output: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FLOAT if yn_output=="float" else SYSTEM_PROMPT_LABEL},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(gold_json, ensure_ascii=False, separators=(",",":"))},
        ]
    }

def build_user_text(subtitle, meta_desc) -> str:
    sub = safe_text(subtitle)
    desc = safe_text(meta_desc)
    if sub and desc:
        return f"TRANSCRIPT:\n{sub}\n\nDESCRIPTION:\n{desc}"
    if sub:
        return f"TRANSCRIPT:\n{sub}"
    if desc:
        return f"DESCRIPTION:\n{desc}"
    return ""

def assemble_rows(
    labels_df: pd.DataFrame,
    text_df: pd.DataFrame,
    lang_thresh: float,
    yn_output: str,
    yn_thresh: float,
    cd_band: Optional[Tuple[float,float]],
) -> list:
    L = norm_cols(labels_df)
    T = norm_cols(text_df)

    # require columns
    need_labels = ["meta_id","china_stance_score","sensitive","collective_action"] + LANG_KEYS_IN
    miss = [c for c in need_labels if c not in L.columns]
    if miss:
        raise SystemExit(f"[error] labels CSV missing columns: {miss}")
    if "meta_id" not in T.columns:
        raise SystemExit(f"[error] text file missing 'meta_id' column")
    # allow meta_desc to be named 'description'
    if "meta_desc" not in T.columns and "description" in T.columns:
        T = T.rename(columns={"description":"meta_desc"})
    for c in ["subtitle","meta_desc"]:
        if c not in T.columns:
            T[c] = ""

    L["meta_id"] = L["meta_id"].map(to_str_meta)
    T["meta_id"] = T["meta_id"].map(to_str_meta)

    df = L.merge(T[["meta_id","subtitle","meta_desc"]], on="meta_id", how="left")

    rows = []
    for _, r in df.iterrows():
        meta_id = r["meta_id"]
        stance  = clamp11(r["china_stance_score"])
        sens_f  = clamp01(r["sensitive"])
        coll_f  = clamp01(r["collective_action"])

        langs = build_languages(r, lang_thresh)

        if yn_output == "float":
            gold = {
                "china_stance_score": stance if stance is not None else 0.0,
                "china_sensitive":    sens_f if sens_f is not None else 0.0,
                "collective_action":  coll_f if coll_f is not None else 0.0,
                "languages":          langs,
            }
        else:
            gold = {
                "china_stance_score": stance if stance is not None else 0.0,
                "china_sensitive":    labelize(sens_f, yn_thresh, cd_band),
                "collective_action":  labelize(coll_f, yn_thresh, cd_band),
                "languages":          langs,
            }

        user_text = build_user_text(r.get("subtitle",""), r.get("meta_desc",""))
        ex = {"meta_id": meta_id}
        ex.update(make_messages(user_text, gold, yn_output))
        rows.append(ex)

    return rows

# write_jsonl now imported from json_utils

def parse_band(s: Optional[str]) -> Optional[Tuple[float,float]]:
    if not s:
        return None
    lo, hi = [float(t.strip()) for t in s.split(",")]
    if not (0.0 <= lo <= hi <= 1.0):
        raise SystemExit("--cd-band must satisfy 0<=lo<=hi<=1")
    return (lo, hi)

def process_single_text_file(file_path: str) -> list:
    """
    Process single file with meta_id, description, transcript for inference.
    Creates prompt-only messages (no assistant gold).
    """
    df = norm_cols(load_table(file_path))
    
    # Check required columns
    required = ["meta_id", "description", "transcript"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] single text file missing columns: {missing}")
    
    df["meta_id"] = df["meta_id"].map(to_str_meta)
    
    rows = []
    for _, r in df.iterrows():
        meta_id = r["meta_id"]
        user_text = build_user_text(r.get("transcript", ""), r.get("description", ""))
        
        # Only system and user messages - no gold assistant for inference
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FLOAT},
            {"role": "user", "content": user_text}
        ]
        
        rows.append({"meta_id": meta_id, "messages": messages})
    
    return rows

def parse_languages_list(lang_str: str) -> List[str]:
    """Parse languages from string representation like "['english', 'mandarin']" or comma-separated"""
    if not lang_str or pd.isna(lang_str):
        return ["no_language"]
    
    lang_str = str(lang_str).strip()
    
    # Handle list-like strings
    if lang_str.startswith('[') and lang_str.endswith(']'):
        try:
            import ast
            langs = ast.literal_eval(lang_str)
            if isinstance(langs, list):
                return [str(l).strip() for l in langs if str(l).strip()]
        except:
            pass
    
    # Handle comma-separated
    if ',' in lang_str:
        return [l.strip() for l in lang_str.split(',') if l.strip()]
    
    # Single language
    return [lang_str] if lang_str else ["no_language"]

def process_single_labeled_file(file_path: str, train_frac: float, lang_thresh: float, yn_output: str, yn_thresh: float, cd_band: Optional[Tuple[float,float]]) -> Tuple[list, list]:
    """
    Process single file with all labels for training.
    Returns (train_rows, val_rows)
    """
    df = norm_cols(load_table(file_path))
    
    # Check required columns - more flexible naming
    meta_col = None
    for col in ["meta_id", "id"]:
        if col in df.columns:
            meta_col = col
            break
    if not meta_col:
        raise SystemExit("[error] single labeled file missing meta_id or id column")
    
    # Map column names
    col_mapping = {
        "subtitle": "transcript",
        "meta_desc": "description", 
        "description": "description",
        "transcript": "transcript"
    }
    
    required_base = ["china_stance_score", "sensitive"]
    optional_cols = ["collective_action", "languages"]
    text_cols = ["transcript", "description"] 
    
    # Find text columns
    transcript_col = None
    desc_col = None
    for orig, target in col_mapping.items():
        if orig in df.columns:
            if target == "transcript" and not transcript_col:
                transcript_col = orig
            elif target == "description" and not desc_col:
                desc_col = orig
    
    # Check required columns exist
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] single labeled file missing columns: {missing}")
    
    # Add missing optional columns with defaults
    if "collective_action" not in df.columns:
        df["collective_action"] = 0.0  # Default to "no"
        print(f"[info] Added missing 'collective_action' column with default value 0.0")
    
    if "languages" not in df.columns:
        df["languages"] = "['english']"  # Default to English
        print(f"[info] Added missing 'languages' column with default value ['english']")
    
    df["meta_id"] = df[meta_col].map(to_str_meta)
    
    # Split train/val
    n = len(df)
    n_train = int(n * train_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()
    
    def process_split(split_df: pd.DataFrame) -> list:
        rows = []
        for _, r in split_df.iterrows():
            meta_id = r["meta_id"]
            stance = clamp11(r["china_stance_score"])
            sens_f = clamp01(r["sensitive"])
            coll_f = clamp01(r["collective_action"])
            
            # Parse languages
            langs = parse_languages_list(r.get("languages", ""))
            
            # Build user text
            transcript = safe_text(r.get(transcript_col, "")) if transcript_col else ""
            description = safe_text(r.get(desc_col, "")) if desc_col else ""
            user_text = build_user_text(transcript, description)
            
            # Build gold JSON
            if yn_output == "float":
                gold = {
                    "china_stance_score": stance if stance is not None else 0.0,
                    "china_sensitive": sens_f if sens_f is not None else 0.0,
                    "collective_action": coll_f if coll_f is not None else 0.0,
                    "languages": langs,
                }
            else:
                gold = {
                    "china_stance_score": stance if stance is not None else 0.0,
                    "china_sensitive": labelize(sens_f, yn_thresh, cd_band),
                    "collective_action": labelize(coll_f, yn_thresh, cd_band),
                    "languages": langs,
                }
            
            ex = {"meta_id": meta_id}
            ex.update(make_messages(user_text, gold, yn_output))
            rows.append(ex)
        
        return rows
    
    train_rows = process_split(train_df)
    val_rows = process_split(val_df)
    
    return train_rows, val_rows

def main():
    ap = argparse.ArgumentParser()
    
    # Mode 1: Separate files (original)
    ap.add_argument("--train-csv", help="Balanced train label CSV")
    ap.add_argument("--val-csv", help="Balanced val label CSV")
    ap.add_argument("--text-file", help="Text file with meta_id, subtitle, meta_desc")
    ap.add_argument("--out-train", help="Output train JSONL")
    ap.add_argument("--out-val", help="Output val JSONL")
    
    # Mode 2: Single text file for inference
    ap.add_argument("--single-text-file", help="Single file with meta_id, description, transcript")
    ap.add_argument("--out-single", help="Single output JSONL (no labels)")
    
    # Mode 3: Single labeled file for training
    ap.add_argument("--single-labeled-file", help="Single file with meta_id, china_stance_score, sensitive, collective_action, languages, subtitle/transcript, meta_desc/description")
    ap.add_argument("--single-train-frac", type=float, default=0.8, help="Fraction for train split")
    ap.add_argument("--out-single-train", help="Train output JSONL from single file")
    ap.add_argument("--out-single-val", help="Val output JSONL from single file")

    # languages
    ap.add_argument("--lang-thresh", type=float, default=0.5,
                    help="Threshold for language indicators (>= -> present). If no_language passes, it is exclusive.")
    # how to emit china_sensitive / collective_action
    ap.add_argument("--yn-output", choices=["float","label"], default="float",
                    help="Emit YN variables as float [0,1] (default) or labels {yes,no,cannot_determine}.")
    ap.add_argument("--yn-thresh", type=float, default=0.5,
                    help="Used only when --yn-output label to convert floats to labels (>= yes).")
    ap.add_argument("--cd-band", default=None,
                    help="Optional band for 'cannot_determine' when --yn-output label, e.g. 0.4,0.6")

    args = ap.parse_args()
    cd_band = parse_band(args.cd_band) if args.yn_output == "label" else None

    # Determine which mode to use
    mode1_args = [args.train_csv, args.val_csv, args.text_file, args.out_train, args.out_val]
    mode2_args = [args.single_text_file, args.out_single]
    mode3_args = [args.single_labeled_file, args.out_single_train, args.out_single_val]
    
    modes_specified = sum([
        all(mode1_args),
        all(mode2_args), 
        all(mode3_args)
    ])
    
    if modes_specified == 0:
        raise SystemExit("Must specify one of: Mode 1 (--train-csv + --val-csv + --text-file + --out-train + --out-val), Mode 2 (--single-text-file + --out-single), or Mode 3 (--single-labeled-file + --out-single-train + --out-single-val)")
    elif modes_specified > 1:
        raise SystemExit("Cannot mix modes - specify only one set of arguments")
    
    if all(mode1_args):
        # Mode 1: Original separate files
        train_df = load_table(args.train_csv)
        val_df   = load_table(args.val_csv)
        text_df  = load_table(args.text_file)

        train_rows = assemble_rows(train_df, text_df, args.lang_thresh, args.yn_output, args.yn_thresh, cd_band)
        val_rows   = assemble_rows(val_df,   text_df, args.lang_thresh, args.yn_output, args.yn_thresh, cd_band)

        write_jsonl(args.out_train, train_rows)
        write_jsonl(args.out_val,   val_rows)
        
    elif all(mode2_args):
        # Mode 2: Single text file for inference
        rows = process_single_text_file(args.single_text_file)
        write_jsonl(args.out_single, rows)
        
    elif all(mode3_args):
        # Mode 3: Single labeled file for training
        train_rows, val_rows = process_single_labeled_file(
            args.single_labeled_file, 
            args.single_train_frac, 
            args.lang_thresh, 
            args.yn_output, 
            args.yn_thresh, 
            cd_band
        )
        write_jsonl(args.out_single_train, train_rows)
        write_jsonl(args.out_single_val, val_rows)

if __name__ == "__main__":
    main()
