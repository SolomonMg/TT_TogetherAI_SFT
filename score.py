#!/usr/bin/env python3
"""
score.py — Summarize model performance vs. human coders.

- Accepts multiple parsed prediction files (from parse.py).
- Displays one row per model (row label = predictions file basename).
- Columns shown:
    stance_R2,
    ch_sensitive_F (or ch_sensitive_R2 if --yn-metric r2),
    col_action_F   (or col_action_R2 if --yn-metric r2)

Legacy behavior preserved:
- Stance OVR threshold remains (--stance-thresh), but display keeps stance_R2.

R² for dichotomous vars:
- If numeric scores exist in predictions (china_sensitive_score/prob/...),
  use them; else map labels via --cd-map (default yes=1,no=0,cd=0.5).

Usage examples:
  python score.py --val-file data/val_BAL.jsonl --yn-metric f1 out/preds_a.parsed.jsonl out/preds_b.parsed.jsonl
  python score.py --val-file data/val_BAL_new.jsonl --yn-metric r2 out/preds_gpt-oss_new.parsed.jsonl
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, precision_recall_fscore_support
from json_utils import REQ_KEYS, ALLOWED_YES_NO_CD, load_jsonl, gold_of, clamp01, clamp11, yesno_to_label

# ---------------- I/O ----------------

def load_val_examples(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    for i, ex in enumerate(data):
        g = gold_of(ex)
        if set(g.keys()) != REQ_KEYS:
            raise RuntimeError(f"{val_path}: example {i} gold keys {set(g.keys())} != {REQ_KEYS}")
    return data

# gold_of now imported from json_utils

def load_parsed_preds(pred_path: str) -> Dict[int, dict]:
    """
    parse.py outputs rows like:
      {"idx": 0, "parsed": true, "json": {...}}
    Return: idx -> prediction dict (only for parsed==True)
    """
    out = {}
    for row in load_jsonl(pred_path):
        if isinstance(row, dict) and row.get("parsed") and "idx" in row and isinstance(row.get("json"), dict):
            out[int(row["idx"])] = row["json"]
    return out

def model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.jsonl$", "", base)
    base = re.sub(r"\.parsed$", "", base)
    base = re.sub(r"[^\w\.-]+", "_", base).strip("_")
    return base or "model"

# ---------------- Normalizers / helpers ----------------

# clamp01, clamp11, yesno_to_label now imported from json_utils

def parse_cd_map(s: str) -> Dict[str, float]:
    # format: "yes=1,no=0,cannot_determine=0.5"
    default = {"yes":1.0, "no":0.0, "cannot_determine":0.5}
    if not s:
        return default
    out = {}
    try:
        parts = [p.strip() for p in s.split(",")]
        for p in parts:
            k, v = [t.strip() for t in p.split("=")]
            if k in ALLOWED_YES_NO_CD:
                out[k] = float(v)
    except Exception:
        return default
    for k, v in default.items():
        out.setdefault(k, v)
    return out

def derive_label(pred: dict, score_key: str, label_key: str, yn_thresh: float | None) -> str:
    """For F1 path (label), optionally threshold numeric score if provided."""
    if yn_thresh is not None and score_key in pred:
        v = clamp01(pred.get(score_key))
        if v is None:
            return yesno_to_label(pred.get(label_key))
        return "yes" if v >= yn_thresh else "no"
    return yesno_to_label(pred.get(label_key))

def numeric_from_pred_or_label(pred: dict, prefix: str, cd_map: Dict[str,float]) -> float:
    """
    Try to pull a numeric probability/score for a dichotomous var.
    Known candidates: <prefix>_{score,prob,yes_prob,p}.
    Fallback: map the categorical label via cd_map.
    """
    candidates = [
        f"{prefix}_score",
        f"{prefix}_prob",
        f"{prefix}_yes_prob",
        f"{prefix}_p",
    ]
    for k in candidates:
        if k in pred:
            v = clamp01(pred[k])
            if v is not None:
                return v
    lab = yesno_to_label(pred.get(prefix))
    return cd_map.get(lab, np.nan)

# ---------------- Metrics ----------------

def stance_r2(golds: List[float], preds: List[float]) -> float:
    g = np.array([clamp11(x) for x in golds], dtype=float)
    p = np.array([clamp11(x) for x in preds], dtype=float)
    mask = ~np.isnan(g) & ~np.isnan(p)
    if mask.sum() < 2:
        return np.nan
    return float(r2_score(g[mask], p[mask]))

def f1_yes(gold_labels: List[str], pred_labels: List[str]) -> float:
    def to01(x): return 1 if x == "yes" else 0
    y_true = np.array([to01(x) for x in gold_labels], dtype=int)
    y_pred = np.array([to01(x) for x in pred_labels], dtype=int)
    if y_true.size == 0:
        return np.nan
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return float(f1)

def r2_on_arrays(g_nums, p_nums):
    g = np.array(g_nums, dtype=float)
    p = np.array(p_nums, dtype=float)
    mask = ~np.isnan(g) & ~np.isnan(p)
    if mask.sum() < 2:
        return np.nan
    return float(r2_score(g[mask], p[mask]))

# ---------------- Core scoring ----------------

def score_models(
    val_file: str,
    pred_paths: List[str],
    stance_thresh: float,
    yn_thresh: float | None,
    yn_metric: str,
    cd_map_str: str,
) -> pd.DataFrame:
    val = load_val_examples(val_file)
    n = len(val)

    # Prepare per-model predictions
    models: List[str] = []
    preds_by_model: List[Dict[int, dict]] = []
    name_counts = {}
    for p in pred_paths:
        name = model_name_from_path(p)
        if name in name_counts:
            name_counts[name] += 1
            name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 1
        models.append(name)
        preds_by_model.append(load_parsed_preds(p))

    cd_map = parse_cd_map(cd_map_str)
    rows = []

    # Gold fields
    gold_stance = [clamp11(gold_of(ex)["china_stance_score"]) for ex in val]
    gold_sensitive = [yesno_to_label(gold_of(ex)["china_sensitive"]) for ex in val]
    gold_collective = [yesno_to_label(gold_of(ex)["collective_action"]) for ex in val]

    # Quick prevalence hint (if all one class, sklearn returns R²=0)
    uniq_sens = sorted(set(gold_sensitive))
    uniq_coll = sorted(set(gold_collective))
    if len(set(gold_sensitive)) == 1:
        print(f"[note] gold china_sensitive constant: {uniq_sens[0]}")
    if len(set(gold_collective)) == 1:
        print(f"[note] gold collective_action constant: {uniq_coll[0]}")

    for mname, mmap in zip(models, preds_by_model):
        # stance scores
        pred_stance = [clamp11(mmap.get(i, {}).get("china_stance_score")) for i in range(n)]
        stance_r2_val = stance_r2(gold_stance, pred_stance)

        # dich vars
        if yn_metric == "f1":
            pred_sensitive_labels = [
                derive_label(mmap.get(i, {}), "china_sensitive_score", "china_sensitive", yn_thresh)
                for i in range(n)
            ]
            pred_collective_labels = [
                derive_label(mmap.get(i, {}), "collective_action_score", "collective_action", yn_thresh)
                for i in range(n)
            ]
            row = {
                "model": mname,
                "stance_R2": stance_r2_val,
                "ch_sensitive_F": f1_yes(gold_sensitive, pred_sensitive_labels),
                "col_action_F":  f1_yes(gold_collective, pred_collective_labels),
            }
        else:  # yn_metric == "r2"
            gold_sensitive_num = [cd_map.get(x, np.nan) for x in gold_sensitive]
            gold_collective_num = [cd_map.get(x, np.nan) for x in gold_collective]
            pred_sensitive_num = [
                numeric_from_pred_or_label(mmap.get(i, {}), "china_sensitive", cd_map)
                for i in range(n)
            ]
            pred_collective_num = [
                numeric_from_pred_or_label(mmap.get(i, {}), "collective_action", cd_map)
                for i in range(n)
            ]
            row = {
                "model": mname,
                "stance_R2": stance_r2_val,
                "ch_sensitive_R2": r2_on_arrays(gold_sensitive_num, pred_sensitive_num),
                "col_action_R2":  r2_on_arrays(gold_collective_num, pred_collective_num),
            }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True, help="Validation JSONL (gold labels in last assistant message).")
    ap.add_argument("--stance-thresh", type=float, default=0.3,
                    help="(Legacy) threshold used elsewhere for stance OVR; kept for compatibility.")
    ap.add_argument("--yn-thresh", type=float, default=None,
                    help="If provided, threshold numeric *_score fields (>=thresh => yes) for dichotomous vars (F1 path).")
    ap.add_argument("--yn-metric", choices=["f1","r2"], default="f1",
                    help="Metric for dichotomous vars (china_sensitive, collective_action). Default: f1.")
    ap.add_argument("--cd-map", default="yes=1,no=0,cannot_determine=0.5",
                    help="Mapping used when --yn-metric r2. Format: 'yes=1,no=0,cannot_determine=0.5'")
    ap.add_argument("--out", help="Optional path to write the summary table CSV.")
    ap.add_argument("preds", nargs="+", help="One or more parsed prediction files (from parse.py).")
    args = ap.parse_args()

    df = score_models(
        val_file=args.val_file,
        pred_paths=args.preds,
        stance_thresh=args.stance_thresh,
        yn_thresh=args.yn_thresh,
        yn_metric=args.yn_metric,
        cd_map_str=args.cd_map,
    )

    if args.yn_metric == "f1":
        cols = ["stance_R2","ch_sensitive_F","col_action_F"]
    else:
        cols = ["stance_R2","ch_sensitive_R2","col_action_R2"]

    print(df[cols].to_string(float_format=lambda x: f"{x:.3f}"))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df[cols].to_csv(args.out)
        print(f"\nWrote summary CSV: {args.out}")

if __name__ == "__main__":
    main()
