"""
python score.py \
  --val-file data/val_BAL.jsonl \
  --preds out/preds_base.parsed.jsonl \
  --stance-thresh 0.3 --eps 1e-6 \
  --dump-csv out/base_preds.csv

"""


# score.py
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from json_utils import load_jsonl, gold_of, ALLOWED_LANGS

def _float_score(x):
    try:
        v = float(x)
        return max(-1.0, min(1.0, v))
    except Exception:
        return None

def _exact_match(g, p, eps: float) -> bool:
    ps = p.get("china_stance_score", None)
    try:
        ps = float(ps)
    except Exception:
        return False
    if abs(float(g["china_stance_score"]) - ps) > eps:
        return False
    if g["china_sensitive"] != p.get("china_sensitive"): return False
    if g["collective_action"] != p.get("collective_action"): return False
    if set(g["languages"]) != set(p.get("languages", [])): return False
    return True

def prf_binary(y_true, y_pred):
    P, R, F, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    return P, R, F, tp, fp, fn

def one_vs_rest_arrays_from_scores(val, preds, sign: str, thresh: float):
    y_true, y_pred = [], []
    for idx, p, ok in preds:
        gs = float(gold_of(val[idx])["china_stance_score"])
        gpos = (gs >  thresh) if sign=="pos" else (gs < -thresh)
        y_true.append(1 if gpos else 0)
        if ok and ("china_stance_score" in p):
            ps = float(p["china_stance_score"])
            ppos = (ps >  thresh) if sign=="pos" else (ps < -thresh)
            y_pred.append(1 if ppos else 0)
        else:
            y_pred.append(0)
    return y_true, y_pred

def one_vs_rest_arrays_label(val, preds, target: str, positive_value: str):
    y_true, y_pred = [], []
    for idx, p, ok in preds:
        g = gold_of(val[idx])
        y_true.append(1 if g[target] == positive_value else 0)
        y_pred.append(1 if (ok and p.get(target) == positive_value) else 0)
    return y_true, y_pred

def multilabel_arrays(val, preds, label: str):
    y_true, y_pred = [], []
    for idx, p, ok in preds:
        g_langs = set(gold_of(val[idx])["languages"])
        y_true.append(1 if label in g_langs else 0)
        y_pred.append(1 if (ok and label in set(p.get("languages", []))) else 0)
    return y_true, y_pred

def stance_regression(val, preds):
    gold = []
    pred = []
    for idx, p, ok in preds:
        gs = _float_score(gold_of(val[idx])["china_stance_score"])
        if gs is None:
            continue
        if ok and ("china_stance_score" in p) and (p["china_stance_score"] is not None):
            ps = float(p["china_stance_score"])
            gold.append(gs); pred.append(ps)
    if len(gold) == 0:
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, pearson=np.nan, spearman=np.nan)
    gold = np.array(gold); pred = np.array(pred)
    mae  = mean_absolute_error(gold, pred)
    rmse = float(np.sqrt(mean_squared_error(gold, pred)))
    r2   = r2_score(gold, pred)
    try:
        pr,_ = pearsonr(gold, pred)
    except Exception:
        pr = np.nan
    try:
        sr,_ = spearmanr(gold, pred)
    except Exception:
        sr = np.nan
    return dict(mae=mae, rmse=rmse, r2=r2, pearson=pr, spearman=sr)

def load_parsed_preds(path):
    rows = load_jsonl(path)
    # keep (idx, pred_json, ok)
    out = []
    for r in rows:
        out.append((int(r["idx"]), r["json"] if r.get("parsed") else {}, bool(r.get("parsed"))))
    return out

def evaluate(val_file: str, preds_file: str, stance_thresh: float, eps: float, dump_csv: str|None):
    val = load_jsonl(val_file)
    preds = load_parsed_preds(preds_file)

    n = len(preds)
    parse_ok = [ok for _, _, ok in preds]
    parse_rate = sum(parse_ok)/n if n else 0.0

    exact = sum(1 for idx, pred, ok in preds if ok and _exact_match(gold_of(val[idx]), pred, eps))
    exact_rate = exact / n if n else 0.0

    rows = []
    def add_row(name, y_true, y_pred):
        P,R,F,tp,fp,fn = prf_binary(y_true, y_pred)
        rows.append({"metric": name, "precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn})

    # stance OVR
    y_true, y_pred = one_vs_rest_arrays_from_scores(val, preds, sign="pos", thresh=stance_thresh)
    add_row("china_stance_positive", y_true, y_pred)
    y_true, y_pred = one_vs_rest_arrays_from_scores(val, preds, sign="neg", thresh=stance_thresh)
    add_row("china_stance_negative", y_true, y_pred)

    # binary Y/N
    y_true, y_pred = one_vs_rest_arrays_label(val, preds, "china_sensitive", "yes")
    add_row("china_sensitive_yes", y_true, y_pred)
    y_true, y_pred = one_vs_rest_arrays_label(val, preds, "collective_action", "yes")
    add_row("collective_action_yes", y_true, y_pred)

    # languages
    for lab in ALLOWED_LANGS:
        yt, yp = multilabel_arrays(val, preds, lab)
        add_row(f"language={lab}", yt, yp)

    # regression
    reg = stance_regression(val, preds)
    regress_row = pd.DataFrame([{
        "metric": "china_stance_score (regression)",
        "MAE": reg["mae"], "RMSE": reg["rmse"], "R2": reg["r2"],
        "Pearson": reg["pearson"], "Spearman": reg["spearman"]
    }])

    prf_table = pd.DataFrame(rows)

    # per-example table
    ex_rows = []
    for idx, pred, ok in preds:
        g = gold_of(val[idx])
        ex_rows.append({
            "idx": idx,
            "parse_ok": ok,
            "gold_stance_score": float(g["china_stance_score"]),
            "pred_stance_score": (float(pred.get("china_stance_score")) if ok else None),
            "gold_sensitive": g["china_sensitive"],
            "pred_sensitive": (pred.get("china_sensitive") if ok else None),
            "gold_collective": g["collective_action"],
            "pred_collective": (pred.get("collective_action") if ok else None),
            "gold_languages": "|".join(g["languages"]),
            "pred_languages": ("|".join(pred.get("languages", [])) if ok else ""),
        })
    per_ex = pd.DataFrame(ex_rows)

    if dump_csv:
        per_ex.to_csv(dump_csv, index=False)
        print(f"\nWrote per-example CSV to {dump_csv}")

    # Print summary
    print(f"\nJSON parse rate: {parse_rate:.3f} | Exact match (eps={eps}): {exact_rate:.3f}")
    print(regress_row.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(prf_table.to_string(index=False))
    return prf_table, regress_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--preds", required=True, help="parsed predictions jsonl")
    ap.add_argument("--compare", help="optional second parsed preds jsonl to diff against")
    ap.add_argument("--stance-thresh", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--dump-csv")
    args = ap.parse_args()

    print("== EVAL ==", flush=True)
    prf_a, _ = evaluate(args.val_file, args.preds, args.stance_thresh, args.eps, args.dump_csv)

    if args.compare:
        print("\n== Δ (compare - preds) ==")
        prf_b, _ = evaluate(args.val_file, args.compare, args.stance_thresh, args.eps, None)
        delta = prf_b[["metric","precision","recall","f1"]].merge(
            prf_a[["metric","precision","recall","f1"]],
            on="metric", suffixes=("_cmp","_pred")
        )
        delta["ΔP"]  = delta["precision_cmp"] - delta["precision_pred"]
        delta["ΔR"]  = delta["recall_cmp"]    - delta["recall_pred"]
        delta["ΔF1"] = delta["f1_cmp"]        - delta["f1_pred"]
        print(delta[["metric","ΔP","ΔR","ΔF1"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

if __name__ == "__main__":
    main()
