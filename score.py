#!/usr/bin/env python3
"""
score.py — Compact scorer for multiple model prediction files (parsed outputs).

- Reads the gold labels from a validation JSONL where the final assistant
  message contains the gold JSON.
- Reads any number of parsed prediction files produced by parse.py
  (each line: {"idx": <int>, "parsed": true/false, "json": {...}})

DISPLAY (as requested):
- Rows = basename of each predictions file
- Columns = stance_R2, ch_sensitive_F, col_action_F

LEGACY COMPAT:
- Accepts --stance-thresh and --eps but only uses them for legacy
  computations under the hood (not shown). Missing predictions are
  treated as negative for classification F1, as before.

Usage:
  python score.py --val-file data/val_BAL.jsonl out/preds_base.parsed.jsonl
  python score.py --val-file data/val_BAL.jsonl \
      out/preds_base.parsed.jsonl out/preds_ft.parsed.jsonl \
      --dump-csv out/summary.csv --stance-thresh 0.3 --eps 1e-6

python score.py --val-file data/val_BAL.jsonl \
  out/preds_llama3.1-70B.parsed.jsonl \
  out/preds_llama3.1-70B-SFT.parsed.jsonl \
  out/preds_llama3.3-70B.parsed.jsonl \
  out/preds_gpt-oss-120b.parsed.jsonl

"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, precision_recall_fscore_support

REQ_KEYS = {"china_stance_score","china_sensitive","collective_action","languages"}

# ---------- I/O ----------

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
                raise RuntimeError(f"{path}: line {ln} not valid JSON: {e}")
    return rows

def load_val_gold(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    gold = []
    for i, ex in enumerate(data):
        try:
            g = json.loads(ex["messages"][-1]["content"])
            if set(g.keys()) != REQ_KEYS:
                raise ValueError(f"gold keys {set(g.keys())} != {REQ_KEYS}")
            gold.append(g)
        except Exception as e:
            raise RuntimeError(f"{val_path}: example {i} missing/invalid gold: {e}")
    return gold

def load_parsed_preds(pred_path: str) -> Dict[int, dict]:
    """
    Expects parse.py lines like: {"idx": 0, "parsed": true, "json": {...}}
    Returns idx -> prediction dict for parsed==True rows.
    """
    preds = {}
    for row in load_jsonl(pred_path):
        if not isinstance(row, dict):
            continue
        idx = row.get("idx", None)
        if idx is None or not row.get("parsed", False):
            continue
        pj = row.get("json")
        if isinstance(pj, dict):
            preds[int(idx)] = pj
    return preds

# ---------- Metrics ----------

def to_float(x):
    try:
        v = float(x)
        # clamp to [-1, 1] as in legacy behavior
        return max(-1.0, min(1.0, v))
    except Exception:
        return None

def stance_r2(gold: List[dict], preds: Dict[int, dict]) -> float:
    g, p = [], []
    for i in range(len(gold)):
        gs = to_float(gold[i]["china_stance_score"])
        ps = to_float(preds.get(i, {}).get("china_stance_score")) if i in preds else None
        if (gs is not None) and (ps is not None):
            g.append(gs); p.append(ps)
    if len(g) < 2:
        return float("nan")
    return r2_score(np.array(g), np.array(p))

def f1_yes(gold: List[dict], preds: Dict[int, dict], field: str) -> float:
    """
    Binary F1 with 'yes' as positive.
    Missing or unparsed preds -> treated as 'no' (legacy).
    """
    y_true, y_pred = [], []
    n = len(gold)
    for i in range(n):
        y_true.append(1 if gold[i][field] == "yes" else 0)
        pv = preds.get(i, {})
        y_pred.append(1 if pv.get(field) == "yes" else 0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return float(f1)

# (Optional) legacy stance OVR metrics using a threshold — computed but not displayed
def _stance_ovr_legacy(gold: List[dict], preds: Dict[int, dict], thresh: float) -> Tuple[float, float]:
    # returns (F1_pos, F1_neg) — not displayed, just to keep old behavior available
    def y_arrays(sign: str):
        yt, yp = [], []
        for i in range(len(gold)):
            g = gold[i]["china_stance_score"]
            is_pos = (g >  thresh)
            is_neg = (g < -thresh)
            yt.append(1 if (is_pos if sign == "pos" else is_neg) else 0)
            ps = preds.get(i, {}).get("china_stance_score")
            if ps is None:
                yp.append(0)
            else:
                ps = float(ps)
                pred_pos = (ps >  thresh)
                pred_neg = (ps < -thresh)
                yp.append(1 if (pred_pos if sign == "pos" else pred_neg) else 0)
        _, _, f, _ = precision_recall_fscore_support(
            yt, yp, average='binary', pos_label=1, zero_division=0
        )
        return float(f)
    return y_arrays("pos"), y_arrays("neg")

# ---------- Main ----------

def main(val_file: str,
         preds: List[str],
         dump_csv: str = None,
         stance_thresh: float = 0.3,   # kept for legacy; not displayed
         eps: float = 1e-6,            # kept for legacy; not displayed
         **_):                          # accept/ignore any extra legacy flags
    gold = load_val_gold(val_file)

    records = []
    for pred_path in preds:
        pred_map = load_parsed_preds(pred_path)

        row = {
            "model": os.path.basename(pred_path),
            "stance_R2": stance_r2(gold, pred_map),
            "ch_sensitive_F": f1_yes(gold, pred_map, "china_sensitive"),
            "col_action_F":  f1_yes(gold, pred_map, "collective_action"),
        }

        # compute legacy stance OVR metrics so behavior remains available (not printed)
        _ = _stance_ovr_legacy(gold, pred_map, stance_thresh)  # noqa: F841

        records.append(row)

    df = pd.DataFrame.from_records(records).set_index("model")
    # nice printing
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    if dump_csv:
        os.makedirs(os.path.dirname(dump_csv), exist_ok=True)
        df.to_csv(dump_csv, index=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True, help="Validation JSONL with gold in final assistant message.")
    ap.add_argument("--dump-csv", help="Optional path to write the compact summary CSV.")
    # legacy flags (accepted for compat, not shown in display)
    ap.add_argument("--stance-thresh", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("preds", nargs="+", help="One or more parsed prediction files (from parse.py).")
    args = ap.parse_args()
    main(**vars(args))
