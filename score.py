#!/usr/bin/env python3
"""
score.py — Summarize model performance vs. human coders.

- Accepts multiple parsed prediction files (from parse.py).
- Displays one row per model (row label = predictions file basename).
- Columns shown:
    stance_R2, stance_pos_F1, stance_neg_F1,
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
  python score.py --val-file data/validation_minimal.jsonl --yn-metric r2 out/preds_minimal.parsed.jsonl
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, precision_recall_fscore_support
from json_utils import REQ_KEYS, ALLOWED_YES_NO_CD, load_jsonl, gold_of, clamp01, clamp11, yesno_to_label, is_valid_text, user_text

# ---------------- I/O ----------------

def load_val_examples(val_path: str) -> List[dict]:
    data = load_jsonl(val_path)
    if not data:
        raise RuntimeError(f"{val_path}: empty file")
    
    # Detect format: comprehensive (11-dim) vs standard (3-dim)
    first_gold = gold_of(data[0])
    is_comprehensive = "china_related" in first_gold
    
    if is_comprehensive:
        # Comprehensive format: check for basic required keys
        required_keys = {"china_related", "china_sensitive"}
        for i, ex in enumerate(data):
            g = gold_of(ex)
            if not required_keys.issubset(set(g.keys())):
                raise RuntimeError(f"{val_path}: example {i} gold missing required keys {required_keys - set(g.keys())}")
    else:
        # Standard format: requires china_stance_score
        required_keys = {"china_stance_score", "china_sensitive"}
        for i, ex in enumerate(data):
            g = gold_of(ex)
            if not required_keys.issubset(set(g.keys())):
                raise RuntimeError(f"{val_path}: example {i} gold missing required keys {required_keys - set(g.keys())}")
    
    return data

# gold_of now imported from json_utils

def load_parsed_preds(pred_path: str) -> Dict[str, dict]:
    """
    parse.py outputs rows like:
      {"idx": 0, "meta_id": "12345", "parsed": true, "json": {...}}
    Return: meta_id -> prediction dict (only for parsed==True)
    """
    out = {}
    for row in load_jsonl(pred_path):
        if isinstance(row, dict) and row.get("parsed") and isinstance(row.get("json"), dict):
            # Use meta_id if available, fallback to idx for backward compatibility
            key = row.get("meta_id", str(row.get("idx", "")))
            if key:
                out[str(key)] = row["json"]
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
    Known candidates: <prefix>_{score,prob,yes_prob,p} and prefix itself.
    Fallback: map the categorical label via cd_map.
    """
    candidates = [
        prefix,  # Check the base field name first
        f"{prefix}_score",
        f"{prefix}_prob",
        f"{prefix}_yes_prob",
        f"{prefix}_p",
    ]
    for k in candidates:
        if k in pred:
            val = pred[k]
            # If it's already numeric, use it directly
            if isinstance(val, (int, float)):
                return clamp01(val)
            # Otherwise try to convert
            v = clamp01(val)
            if v is not None:
                return v
    
    # Fallback to label mapping
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

def normalize_cd_label(x: Any) -> str:
    v = str(x).strip().lower()
    if v in {"", "nan", "none", "n/a"}:
        return "n/a"
    if v in {"cannot_determine", "cannot determine", "cannot-determine", "cant_determine", "cd", "uncertain"}:
        return "cannot_determine"
    if v in {"yes", "y", "true", "1"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    if v in {"neutral", "unclear", "neutral/unclear"}:
        return "neutral/unclear"
    return v

def multiclass_f1(gold_labels: List[str], pred_labels: List[str], labels: List[str]) -> Dict[str, float]:
    g = [normalize_cd_label(x) for x in gold_labels]
    p = [normalize_cd_label(x) for x in pred_labels]
    # Filter to rows where gold is in the label set
    keep = [i for i, gv in enumerate(g) if gv in labels]
    if not keep:
        return {f"f1_{lab}": np.nan for lab in labels} | {"f1_macro": np.nan}
    g_f = [g[i] for i in keep]
    p_f = [p[i] for i in keep]
    _, _, f1s, _ = precision_recall_fscore_support(
        g_f, p_f, labels=labels, average=None, zero_division=0
    )
    f1_macro = float(np.nanmean(f1s)) if len(f1s) else np.nan
    out = {f"f1_{lab}": float(val) for lab, val in zip(labels, f1s)}
    out["f1_macro"] = f1_macro
    return out

def bootstrap_ci(metric_fn, n: int, alpha: float, rng: np.random.Generator) -> Dict[str, tuple]:
    """Return {metric: (lo, hi)} from bootstrap samples."""
    samples = []
    for _ in range(n):
        samples.append(metric_fn())
    if not samples:
        return {}
    keys = samples[0].keys()
    out = {}
    for k in keys:
        vals = np.array([s.get(k, np.nan) for s in samples], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out[k] = (np.nan, np.nan)
            continue
        lo = np.quantile(vals, alpha / 2)
        hi = np.quantile(vals, 1 - alpha / 2)
        out[k] = (float(lo), float(hi))
    return out

def r2_on_arrays(g_nums, p_nums):
    g = np.array(g_nums, dtype=float)
    p = np.array(p_nums, dtype=float)
    mask = ~np.isnan(g) & ~np.isnan(p)
    if mask.sum() < 2:
        return np.nan
    return float(r2_score(g[mask], p[mask]))

def stance_f1_positive(gold_scores: List[float], pred_scores: List[float]) -> float:
    """F1 for positive stance classification (> 0 vs <= 0)"""
    def to_positive(x): return 1 if (isinstance(x, (int, float)) and not np.isnan(x) and x > 0) else 0
    y_true = np.array([to_positive(clamp11(x)) for x in gold_scores], dtype=int)
    y_pred = np.array([to_positive(clamp11(x)) for x in pred_scores], dtype=int)
    if y_true.size == 0:
        return np.nan
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return float(f1)

def stance_f1_negative(gold_scores: List[float], pred_scores: List[float]) -> float:
    """F1 for negative stance classification (< 0 vs >= 0)"""
    def to_negative(x): return 1 if (isinstance(x, (int, float)) and not np.isnan(x) and x < 0) else 0
    y_true = np.array([to_negative(clamp11(x)) for x in gold_scores], dtype=int)
    y_pred = np.array([to_negative(clamp11(x)) for x in pred_scores], dtype=int)
    if y_true.size == 0:
        return np.nan
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return float(f1)

# ---------------- Comprehensive format scoring ----------------

def score_comprehensive(
    val: List[dict],
    models: List[str],
    preds_by_model: List[Dict[str, dict]],
    yn_metric: str,
    cd_map: Dict[str, float],
    yn_thresh: float | None,
    ci: bool = False,
    ci_n: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int | None = None,
) -> pd.DataFrame:
    """Score models on comprehensive 11-dimension format.
    
    Metrics computed:
    - F1 for china_related (binary yes vs not-yes)
    - F1 for yes/no dimensions (binary yes vs not-yes)
    - 3-class F1 (pro/anti/neutral/unclear) for stance dimensions, plus macro-F1
    """
    rows = []
    
    # Extract gold labels for all dimensions
    gold_china_related = [yesno_to_label(gold_of(ex).get("china_related", "")) for ex in val]
    gold_sensitive = [yesno_to_label(gold_of(ex).get("china_sensitive", "")) for ex in val]
    gold_collective = [yesno_to_label(gold_of(ex).get("collective_action", "")) for ex in val]
    gold_hate = [yesno_to_label(gold_of(ex).get("hate_speech", "")) for ex in val]
    gold_harmful = [yesno_to_label(gold_of(ex).get("harmful_content", "")) for ex in val]
    gold_news = [yesno_to_label(gold_of(ex).get("news_segments", "")) for ex in val]
    gold_inauthentic = [yesno_to_label(gold_of(ex).get("inauthentic_content", "")) for ex in val]
    gold_derivative = [yesno_to_label(gold_of(ex).get("derivative_content", "")) for ex in val]
    
    # Stance dimensions (3-way: pro/anti/neutral)
    gold_stance_gov = [str(gold_of(ex).get("china_ccp_government", "")).strip().lower() for ex in val]
    gold_stance_culture = [str(gold_of(ex).get("china_people_culture", "")).strip().lower() for ex in val]
    gold_stance_tech = [str(gold_of(ex).get("china_technology_development", "")).strip().lower() for ex in val]
    
    for mname, mmap in zip(models, preds_by_model):
        # Align predictions
        pred_china_related = []
        pred_sensitive = []
        pred_collective = []
        pred_hate = []
        pred_harmful = []
        pred_news = []
        pred_inauthentic = []
        pred_derivative = []
        pred_stance_gov = []
        pred_stance_culture = []
        pred_stance_tech = []
        
        for ex in val:
            meta_id = str(ex.get("meta_id", ""))
            pred = mmap.get(meta_id, {})
            
            pred_china_related.append(yesno_to_label(pred.get("china_related", "")))
            pred_sensitive.append(derive_label(pred, "china_sensitive_score", "china_sensitive", yn_thresh))
            pred_collective.append(derive_label(pred, "collective_action_score", "collective_action", yn_thresh))
            pred_hate.append(derive_label(pred, "hate_speech_score", "hate_speech", yn_thresh))
            pred_harmful.append(derive_label(pred, "harmful_content_score", "harmful_content", yn_thresh))
            pred_news.append(derive_label(pred, "news_segments_score", "news_segments", yn_thresh))
            pred_inauthentic.append(derive_label(pred, "inauthentic_content_score", "inauthentic_content", yn_thresh))
            pred_derivative.append(derive_label(pred, "derivative_content_score", "derivative_content", yn_thresh))
            
            pred_stance_gov.append(str(pred.get("china_ccp_government", "")).strip().lower())
            pred_stance_culture.append(str(pred.get("china_people_culture", "")).strip().lower())
            pred_stance_tech.append(str(pred.get("china_technology_development", "")).strip().lower())
        
        # Compute metrics
        row = {"model": mname}

        def compute_metrics(indices: List[int] | None = None) -> Dict[str, float]:
            if indices is None:
                g_rel, p_rel = gold_china_related, pred_china_related
                g_sens, p_sens = gold_sensitive, pred_sensitive
                g_coll, p_coll = gold_collective, pred_collective
                g_hate, p_hate = gold_hate, pred_hate
                g_harm, p_harm = gold_harmful, pred_harmful
                g_news, p_news = gold_news, pred_news
                g_inauth, p_inauth = gold_inauthentic, pred_inauthentic
                g_deriv, p_deriv = gold_derivative, pred_derivative
                g_gov, p_gov = gold_stance_gov, pred_stance_gov
                g_cult, p_cult = gold_stance_culture, pred_stance_culture
                g_tech, p_tech = gold_stance_tech, pred_stance_tech
            else:
                g_rel = [gold_china_related[i] for i in indices]
                p_rel = [pred_china_related[i] for i in indices]
                g_sens = [gold_sensitive[i] for i in indices]
                p_sens = [pred_sensitive[i] for i in indices]
                g_coll = [gold_collective[i] for i in indices]
                p_coll = [pred_collective[i] for i in indices]
                g_hate = [gold_hate[i] for i in indices]
                p_hate = [pred_hate[i] for i in indices]
                g_harm = [gold_harmful[i] for i in indices]
                p_harm = [pred_harmful[i] for i in indices]
                g_news = [gold_news[i] for i in indices]
                p_news = [pred_news[i] for i in indices]
                g_inauth = [gold_inauthentic[i] for i in indices]
                p_inauth = [pred_inauthentic[i] for i in indices]
                g_deriv = [gold_derivative[i] for i in indices]
                p_deriv = [pred_derivative[i] for i in indices]
                g_gov = [gold_stance_gov[i] for i in indices]
                p_gov = [pred_stance_gov[i] for i in indices]
                g_cult = [gold_stance_culture[i] for i in indices]
                p_cult = [pred_stance_culture[i] for i in indices]
                g_tech = [gold_stance_tech[i] for i in indices]
                p_tech = [pred_stance_tech[i] for i in indices]

            out = {}
            out["china_related_F1"] = f1_yes(g_rel, p_rel)
            out["sensitive_F1"] = f1_yes(g_sens, p_sens)
            out["collective_F1"] = f1_yes(g_coll, p_coll)
            out["hate_F1"] = f1_yes(g_hate, p_hate)
            out["harmful_F1"] = f1_yes(g_harm, p_harm)
            out["news_F1"] = f1_yes(g_news, p_news)
            out["inauthentic_F1"] = f1_yes(g_inauth, p_inauth)
            out["derivative_F1"] = f1_yes(g_deriv, p_deriv)

            stance_labels = ["pro", "anti", "neutral/unclear"]
            for prefix, g, p in [
                ("stance_gov", g_gov, p_gov),
                ("stance_culture", g_cult, p_cult),
                ("stance_tech", g_tech, p_tech),
            ]:
                f1s = multiclass_f1(g, p, stance_labels)
                out[f"{prefix}_f1_pro"] = f1s["f1_pro"]
                out[f"{prefix}_f1_anti"] = f1s["f1_anti"]
                out[f"{prefix}_f1_neutral"] = f1s["f1_neutral/unclear"]
                out[f"{prefix}_f1_macro"] = f1s["f1_macro"]
            return out

        metrics = compute_metrics()
        row.update(metrics)

        if ci:
            rng = np.random.default_rng(ci_seed)
            n_items = len(val)
            def sample_metrics():
                idx = rng.integers(0, n_items, size=n_items).tolist()
                return compute_metrics(indices=idx)
            cis = bootstrap_ci(sample_metrics, ci_n, ci_alpha, rng)
            for k, (lo, hi) in cis.items():
                row[f"{k}_ci_lo"] = lo
                row[f"{k}_ci_hi"] = hi
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("model")
    return df

# ---------------- Core scoring ----------------

def score_models(
    val_file: str,
    pred_paths: List[str],
    stance_thresh: float,
    yn_thresh: float | None,
    yn_metric: str,
    cd_map_str: str,
    min_text_len: int = 0,
    ci: bool = False,
    ci_n: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int | None = None,
) -> pd.DataFrame:
    val = load_val_examples(val_file)
    
    # Filter out examples with insufficient text content if requested
    if min_text_len > 0:
        original_count = len(val)
        filtered_val = []
        for ex in val:
            text_content = user_text(ex)
            if is_valid_text(text_content, min_text_len):
                filtered_val.append(ex)
        
        filtered_count = original_count - len(filtered_val)
        if filtered_count > 0:
            print(f"[info] Filtered out {filtered_count}/{original_count} examples with insufficient text (< {min_text_len} chars)")
        val = filtered_val

    # Detect format
    first_gold = gold_of(val[0])
    is_comprehensive = "china_related" in first_gold
    
    if is_comprehensive:
        print("[info] Detected comprehensive label format (11 dimensions)")
    else:
        print("[info] Detected standard label format (3 dimensions)")

    # Prepare per-model predictions
    models: List[str] = []
    preds_by_model: List[Dict[str, dict]] = []
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

    # For comprehensive format, we can't compute stance metrics the same way
    if is_comprehensive:
        return score_comprehensive(val, models, preds_by_model, yn_metric, cd_map, yn_thresh, ci, ci_n, ci_alpha, ci_seed)
    
    # Gold fields (standard format)
    gold_stance = [clamp11(gold_of(ex)["china_stance_score"]) for ex in val]
    
    # For china_sensitive, use numeric values if they exist, otherwise convert labels
    gold_sensitive_raw = [gold_of(ex)["china_sensitive"] for ex in val]
    if isinstance(gold_sensitive_raw[0], (int, float)):
        # Numeric gold values - use directly for R² or convert for F1
        gold_sensitive = gold_sensitive_raw if yn_metric == "r2" else [yesno_to_label(x) for x in gold_sensitive_raw]
    else:
        # String gold values - convert to labels
        gold_sensitive = [yesno_to_label(x) for x in gold_sensitive_raw]
    
    # collective_action is optional
    has_collective_action = "collective_action" in gold_of(val[0])
    if has_collective_action:
        gold_collective_raw = [gold_of(ex)["collective_action"] for ex in val]
        if isinstance(gold_collective_raw[0], (int, float)):
            # Numeric gold values - use directly for R² or convert for F1
            gold_collective = gold_collective_raw if yn_metric == "r2" else [yesno_to_label(x) for x in gold_collective_raw]
        else:
            # String gold values - convert to labels
            gold_collective = [yesno_to_label(x) for x in gold_collective_raw]
    else:
        gold_collective = None

    # Quick prevalence hint (if all one class, sklearn returns R²=0)
    uniq_sens = sorted(set(gold_sensitive))
    if len(set(gold_sensitive)) == 1:
        print(f"[note] gold china_sensitive constant: {uniq_sens[0]}")
    if has_collective_action:
        uniq_coll = sorted(set(gold_collective))
        if len(set(gold_collective)) == 1:
            print(f"[note] gold collective_action constant: {uniq_coll[0]}")

    for mname, mmap in zip(models, preds_by_model):
        # Align predictions with validation data using meta_id
        pred_stance = []
        for ex in val:
            meta_id = str(ex.get("meta_id", ""))
            pred = mmap.get(meta_id, {})
            pred_stance.append(clamp11(pred.get("china_stance_score")))
        
        stance_r2_val = stance_r2(gold_stance, pred_stance)
        stance_pos_f1 = stance_f1_positive(gold_stance, pred_stance)
        stance_neg_f1 = stance_f1_negative(gold_stance, pred_stance)

        # dich vars
        if yn_metric == "f1":
            pred_sensitive_labels = []
            for ex in val:
                meta_id = str(ex.get("meta_id", ""))
                pred = mmap.get(meta_id, {})
                pred_sensitive_labels.append(
                    derive_label(pred, "china_sensitive_score", "china_sensitive", yn_thresh)
                )
            row = {
                "model": mname,
                "stance_R2": stance_r2_val,
                "stance_pos_F1": stance_pos_f1,
                "stance_neg_F1": stance_neg_f1,
                "ch_sensitive_F": f1_yes(gold_sensitive, pred_sensitive_labels),
            }
            if has_collective_action:
                pred_collective_labels = []
                for ex in val:
                    meta_id = str(ex.get("meta_id", ""))
                    pred = mmap.get(meta_id, {})
                    pred_collective_labels.append(
                        derive_label(pred, "collective_action_score", "collective_action", yn_thresh)
                    )
                row["col_action_F"] = f1_yes(gold_collective, pred_collective_labels)
        else:  # yn_metric == "r2"
            # Use numeric gold values directly (no cd_map conversion needed)
            if isinstance(gold_sensitive[0], (int, float)):
                gold_sensitive_num = gold_sensitive
            else:
                gold_sensitive_num = [cd_map.get(x, np.nan) for x in gold_sensitive]
            
            pred_sensitive_num = []
            for ex in val:
                meta_id = str(ex.get("meta_id", ""))
                pred = mmap.get(meta_id, {})
                pred_sensitive_num.append(numeric_from_pred_or_label(pred, "china_sensitive", cd_map))
            row = {
                "model": mname,
                "stance_R2": stance_r2_val,
                "stance_pos_F1": stance_pos_f1,
                "stance_neg_F1": stance_neg_f1,
                "ch_sensitive_R2": r2_on_arrays(gold_sensitive_num, pred_sensitive_num),
            }
            if has_collective_action:
                # Use numeric gold values directly (no cd_map conversion needed)
                if isinstance(gold_collective[0], (int, float)):
                    gold_collective_num = gold_collective
                else:
                    gold_collective_num = [cd_map.get(x, np.nan) for x in gold_collective]
                
                pred_collective_num = []
                for ex in val:
                    meta_id = str(ex.get("meta_id", ""))
                    pred = mmap.get(meta_id, {})
                    pred_collective_num.append(numeric_from_pred_or_label(pred, "collective_action", cd_map))
                row["col_action_R2"] = r2_on_arrays(gold_collective_num, pred_collective_num)
        if ci:
            rng = np.random.default_rng(ci_seed)
            n_items = len(val)
            def compute_standard(indices: List[int] | None = None) -> Dict[str, float]:
                if indices is None:
                    g_st = gold_stance
                    p_st = pred_stance
                    g_sens = gold_sensitive
                else:
                    g_st = [gold_stance[i] for i in indices]
                    p_st = [pred_stance[i] for i in indices]
                    g_sens = [gold_sensitive[i] for i in indices]

                out = {
                    "stance_R2": stance_r2(g_st, p_st),
                    "stance_pos_F1": stance_f1_positive(g_st, p_st),
                    "stance_neg_F1": stance_f1_negative(g_st, p_st),
                }

                if yn_metric == "f1":
                    if indices is None:
                        pred_sens = pred_sensitive_labels
                    else:
                        pred_sens = [pred_sensitive_labels[i] for i in indices]
                    out["ch_sensitive_F"] = f1_yes(g_sens, pred_sens)
                    if has_collective_action:
                        if indices is None:
                            g_coll = gold_collective
                            pred_coll = pred_collective_labels
                        else:
                            g_coll = [gold_collective[i] for i in indices]
                            pred_coll = [pred_collective_labels[i] for i in indices]
                        out["col_action_F"] = f1_yes(g_coll, pred_coll)
                else:
                    if indices is None:
                        g_sens_num = gold_sensitive_num
                        pred_sens_num = pred_sensitive_num
                    else:
                        g_sens_num = [gold_sensitive_num[i] for i in indices]
                        pred_sens_num = [pred_sensitive_num[i] for i in indices]
                    out["ch_sensitive_R2"] = r2_on_arrays(g_sens_num, pred_sens_num)
                    if has_collective_action:
                        if indices is None:
                            g_coll_num = gold_collective_num
                            pred_coll_num = pred_collective_num
                        else:
                            g_coll_num = [gold_collective_num[i] for i in indices]
                            pred_coll_num = [pred_collective_num[i] for i in indices]
                        out["col_action_R2"] = r2_on_arrays(g_coll_num, pred_coll_num)
                return out

            def sample_metrics():
                idx = rng.integers(0, n_items, size=n_items).tolist()
                return compute_standard(indices=idx)

            cis = bootstrap_ci(sample_metrics, ci_n, ci_alpha, rng)
            for k, (lo, hi) in cis.items():
                row[f"{k}_ci_lo"] = lo
                row[f"{k}_ci_hi"] = hi

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
    ap.add_argument("--min-text-len", type=int, default=0,
                    help="Minimum text length to include example in evaluation (default: 0, no filtering)")
    ap.add_argument("--ci", action="store_true",
                    help="If set, compute bootstrap confidence intervals for metrics.")
    ap.add_argument("--ci-n", type=int, default=1000,
                    help="Number of bootstrap samples (default: 1000).")
    ap.add_argument("--ci-alpha", type=float, default=0.05,
                    help="Alpha for CI (default: 0.05 for 95%% CI).")
    ap.add_argument("--ci-seed", type=int, default=None,
                    help="Random seed for bootstrap sampling.")
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
        min_text_len=args.min_text_len,
        ci=args.ci,
        ci_n=args.ci_n,
        ci_alpha=args.ci_alpha,
        ci_seed=args.ci_seed,
    )

    # Select columns that actually exist in the dataframe
    # Check if this is comprehensive format
    if "china_related_F1" in df.columns:
        # Comprehensive format - show ordered metrics
        cols = [
            "china_related_F1",
            "sensitive_F1",
            "collective_F1",
            "hate_F1",
            "harmful_F1",
            "news_F1",
            "inauthentic_F1",
            "derivative_F1",
        ]
        stance_prefixes = ["stance_gov", "stance_culture", "stance_tech"]
        for prefix in stance_prefixes:
            cols.extend([
                f"{prefix}_f1_pro",
                f"{prefix}_f1_anti",
                f"{prefix}_f1_neutral",
                f"{prefix}_f1_macro",
            ])
        cols = [c for c in cols if c in df.columns]
    elif args.yn_metric == "f1":
        potential_cols = ["stance_R2","stance_pos_F1","stance_neg_F1","ch_sensitive_F","col_action_F"]
        cols = [c for c in potential_cols if c in df.columns]
    else:
        potential_cols = ["stance_R2","stance_pos_F1","stance_neg_F1","ch_sensitive_R2","col_action_R2"]
        cols = [c for c in potential_cols if c in df.columns]

    ci_cols = []
    if args.ci:
        for c in cols:
            lo = f"{c}_ci_lo"
            hi = f"{c}_ci_hi"
            if lo in df.columns and hi in df.columns:
                ci_cols.extend([lo, hi])

    # Pretty print: combine value + CI for readability
    if args.ci and ci_cols:
        display_df = df[cols].copy()
        for c in cols:
            lo = f"{c}_ci_lo"
            hi = f"{c}_ci_hi"
            if lo in df.columns and hi in df.columns:
                display_df[c] = [
                    f"{v:.3f} [{l:.3f}, {h:.3f}]"
                    if pd.notna(v) and pd.notna(l) and pd.notna(h)
                    else "nan"
                    for v, l, h in zip(df[c].values, df[lo].values, df[hi].values)
                ]
        print(display_df.T.to_string())
    else:
        print(df[cols].T.to_string(float_format=lambda x: f"{x:.3f}"))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_rows = []
        for model_name, row in df.iterrows():
            for metric in cols:
                lo = row.get(f"{metric}_ci_lo", np.nan)
                hi = row.get(f"{metric}_ci_hi", np.nan)
                out_rows.append({
                    "model": model_name,
                    "metric": metric,
                    "score": row.get(metric, np.nan),
                    "CI_low": lo,
                    "CI_high": hi,
                })
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(args.out, index=False)
        print(f"\nWrote summary CSV: {args.out}")

if __name__ == "__main__":
    main()
