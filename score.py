
#!/usr/bin/env python3
"""
score.py — Compact scorer + per-example CSVs.

What it does
------------
1) Prints a compact table, one row per predictions file (basename):
   columns: stance_R2, ch_sensitive_F, col_action_F

2) For each predictions file, writes a per-example CSV with columns:
   idx,
   model_stance_score, gold_stance_score,
   model_china_sensitive, gold_china_sensitive,
   model_collective_action, gold_collective_action,
   transcript

Inputs
------
--val-file: validation JSONL where the final assistant message has the gold JSON:
  {
    "china_stance_score": float in [-1,1],
    "china_sensitive": "yes"|"no"|"cannot_determine",
    "collective_action": "yes"|"no"|"cannot_determine",
    "languages": [...]
  }

Positional args: one or more parsed predictions JSONL files produced by parse.py:
  {"idx": <int>, "parsed": true, "json": {...}}

Compat notes
------------
- Accepts legacy flags --stance-thresh and --eps (used only for legacy internals).
- Missing/unparsed predictions are treated as negative for Y/N F1, as before.
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
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, precision_recall_fscore_support

REQ_KEYS = {"china_stance_score","china_sensitive","collective_action","languages"}

# ---------- I/O & helpers ----------

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

def gold_of(example: dict) -> dict:
    return json.loads(example["messages"][-1]["content"])

def load_val_examples(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    # sanity check gold schema
    for i, ex in enumerate(data):
        g = gold_of(ex)
        if set(g.keys()) != REQ_KEYS:
            raise RuntimeError(f"{val_path}: example {i} gold keys {set(g.keys())} != {REQ_KEYS}")
    return data

def user_text(ex: dict) -> str:
    parts = [m.get("content","") for m in ex.get("messages", []) if m.get("role") == "user"]
    return "\n\n".join(parts).strip()

def load_parsed_preds(pred_path: str) -> Dict[int, dict]:
    """
    parse.py lines look like: {"idx": 0, "parsed": true, "json": {...}}
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

def to_float(x: Any):
    try:
        v = float(x)
        return max(-1.0, min(1.0, v))  # clamp as in legacy
    except Exception:
        return None

# ---------- Metrics (displayed) ----------

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
    for i in range(len(gold)):
        y_true.append(1 if gold[i][field] == "yes" else 0)
        pv = preds.get(i, {})
        y_pred.append(1 if pv.get(field) == "yes" else 0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return float(f1)

# ---------- Legacy (kept, not displayed) ----------

def _stance_ovr_legacy(gold: List[dict], preds: Dict[int, dict], thresh: float) -> Tuple[float, float]:
    # returns (F1_pos, F1_neg) — kept for compatibility, not shown
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

# ---------- Per-example CSV writing ----------

def write_perexample_csv(pred_path: str,
                         perex_dir: str,
                         data: List[dict],
                         preds: Dict[int, dict]) -> str:
    """
    Writes per-example CSV for a single model's predictions.
    Returns the path written.
    """
    if perex_dir:
        os.makedirs(perex_dir, exist_ok=True)
        base = os.path.basename(pred_path)
        stem = base.rsplit(".", 1)[0]
        out_path = os.path.join(perex_dir, f"{stem}.perex.csv")
    else:
        # write next to the predictions file
        d = os.path.dirname(pred_path) or "."
        base = os.path.basename(pred_path)
        stem = base.rsplit(".", 1)[0]
        out_path = os.path.join(d, f"{stem}.perex.csv")

    rows = []
    for i, ex in enumerate(data):
        g = gold_of(ex)
        p = preds.get(i, {})

        rows.append({
            "idx": i,
            "model_stance_score": to_float(p.get("china_stance_score")) if p else None,
            "gold_stance_score":  to_float(g.get("china_stance_score")),
            "model_china_sensitive": p.get("china_sensitive") if p else "",
            "gold_china_sensitive":  g.get("china_sensitive"),
            "model_collective_action": p.get("collective_action") if p else "",
            "gold_collective_action":  g.get("collective_action"),
            "transcript": user_text(ex),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path

# ---------- Main ----------

def main(val_file: str,
         preds: List[str],
         dump_csv: str = None,
         perex_dir: str = None,
         stance_thresh: float = 0.3,   # kept for legacy; not displayed
         eps: float = 1e-6,            # kept for legacy; not displayed
         **_):                          # accept/ignore any extra legacy flags
    data = load_val_examples(val_file)
    gold = [gold_of(ex) for ex in data]

    summary_records = []
    wrote_paths = []

    for pred_path in preds:
        pred_map = load_parsed_preds(pred_path)

        # compact summary row
        row = {
            "model": os.path.basename(pred_path),
            "stance_R2": stance_r2(gold, pred_map),
            "ch_sensitive_F": f1_yes(gold, pred_map, "china_sensitive"),
            "col_action_F":  f1_yes(gold, pred_map, "collective_action"),
        }
        # legacy (not shown)
        _ = _stance_ovr_legacy(gold, pred_map, stance_thresh)  # noqa: F841

        summary_records.append(row)

        # per-example CSV
        out_path = write_perexample_csv(pred_path, perex_dir, data, pred_map)
        wrote_paths.append(out_path)

    # print compact table
    df = pd.DataFrame.from_records(summary_records).set_index("model")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    # optional overall CSV of summary table
    if dump_csv:
        os.makedirs(os.path.dirname(dump_csv) or ".", exist_ok=True)
        df.to_csv(dump_csv, index=True)

    # echo where we wrote per-example files
    for p in wrote_paths:
        print(f"Wrote per-example CSV: {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True, help="Validation JSONL with gold in final assistant message.")
    ap.add_argument("--dump-csv", help="Optional path to write the compact summary CSV.")
    ap.add_argument("--perex-dir", help="Directory to place per-example CSVs (default: next to each preds file).")
    # legacy flags (accepted for compat, not shown in display)
    ap.add_argument("--stance-thresh", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("preds", nargs="+", help="One or more parsed prediction files (from parse.py).")
    args = ap.parse_args()
    main(**vars(args))
