"""

Evaluate model performance by hitting model and comparing to gold standard JSON.

- Schema (assistant JSON):
    {
      "china_stance_score": float in [-1,1],
      "china_sensitive": "yes"|"no"|"cannot_determine",
      "collective_action": "yes"|"no"|"cannot_determine",
      "languages": ["english","mandarin","spanish","other","no_language"]
    }

- Reports:
    * JSON-parse rate (valid schema)
    * "Exact match" with tolerance on china_stance_score (--eps)
    * Stance regression metrics: MAE, RMSE, R2, Pearson, Spearman
    * One-vs-rest PRF:
        - china_stance_positive  (score >  +--stance-thresh)
        - china_stance_negative  (score <  - --stance-thresh)
        - china_sensitive_yes
        - collective_action_yes
        - language=*

- Runs base and optional fine-tuned model.

Usage:
export TOGETHER_API_KEY=...
python eval_model_perf_val_jsonl.py \
  --val-file data/val.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --ft-model your-namespace/Meta-Llama-3.1-8B...-tiktok-sft-vX \
  --concurrency 4 --temperature 0 \
  --stance-thresh 0.3 --eps 1e-6 \
  --dump-csv data/preds.csv



"""

#!/usr/bin/env python3
"""
Evaluate val.jsonl by querying a model and comparing to gold assistant JSON.

- Schema (assistant JSON):
    {
      "china_stance_score": float in [-1,1],
      "china_sensitive": "yes"|"no"|"cannot_determine",
      "collective_action": "yes"|"no"|"cannot_determine",
      "languages": ["english","mandarin","spanish","other","no_language"]
    }

- Reports:
    * JSON-parse rate (valid schema)
    * "Exact match" with tolerance on china_stance_score (--eps)
    * Stance regression metrics: MAE, RMSE, R2, Pearson, Spearman
    * One-vs-rest PRF:
        - china_stance_positive  (score >  +--stance-thresh)
        - china_stance_negative  (score <  - --stance-thresh)
        - china_sensitive_yes
        - collective_action_yes
        - language=*

- Runs base and optional fine-tuned model.

Usage:
export TOGETHER_API_KEY=...
python eval_val_jsonl_sft_original.py \
  --val-file data/val.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --ft-model your-namespace/Meta-Llama-3.1-8B...-tiktok-sft-vX \
  --concurrency 4 --temperature 0 \
  --stance-thresh 0.3 --eps 1e-6 \
  --dump-csv data/preds.csv
"""
import os, json, argparse, asyncio, re, math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from together import Together  # pip install together

ALLOWED_YES_NO_CD = {"yes","no","cannot_determine"}
ALLOWED_LANGS     = ["english","mandarin","spanish","other","no_language"]
REQ_KEYS          = {"china_stance_score","china_sensitive","collective_action","languages"}

def ok_langs(x):
    return (isinstance(x, list) and len(x) > 0 and
            all(isinstance(s, str) and s in ALLOWED_LANGS for s in x) and
            (("no_language" not in x) or (len(x) == 1)))

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = re.sub(r'[\u2028\u2029]', '', line).rstrip("\r\n")
            if not line: continue
            obj = json.loads(line)
            gold = json.loads(obj["messages"][-1]["content"])
            assert set(gold.keys()) == REQ_KEYS, f"{path}: line {i} assistant keys {set(gold.keys())} != {REQ_KEYS}"
            out.append(obj)
    return out

def gold_of(example: dict) -> dict:
    return json.loads(example["messages"][-1]["content"])

def prompt_messages(example: dict) -> List[Dict[str, str]]:
    return example["messages"][:-1]  # hide gold

@dataclass
class EvalResult:
    parse_rate: float
    exact_match: float
    prf_table: pd.DataFrame
    regress_row: pd.DataFrame
    per_example: pd.DataFrame
    model_name: str

def _float_score(x) -> float | None:
    try:
        v = float(x)
        # clamp tiny drift
        if v < -1: v = -1.0
        if v >  1: v =  1.0
        return v
    except Exception:
        return None

def _exact_match(g, p, eps: float) -> bool:
    # stance_score within eps; other fields exact / set-equal
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

async def _chat_call(client: Together, model: str, messages: List[Dict[str,str]], max_tokens=128, temperature=0.0) -> str:
    loop = asyncio.get_running_loop()
    def _do():
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=False
        )
        return resp.choices[0].message.content
    return await loop.run_in_executor(None, _do)

async def infer_all(client: Together, model: str, data: List[dict], limit=0, concurrency=4, max_tokens=128, temperature=0.0) -> List[Tuple[int, dict, bool]]:
    """
    Returns [(idx, pred_json, schema_ok)].
    If schema invalid or parse fails, returns {"invalid":True} with ok=False.
    """
    n = min(limit, len(data)) if limit else len(data)
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[int, dict, bool]] = [None]*n

    async def one(i: int, ex: dict):
        msgs = prompt_messages(ex)
        async with sem:
            raw = await _chat_call(client, model, msgs, max_tokens=max_tokens, temperature=temperature)
        try:
            y = json.loads(raw)
            schema_ok = (
                set(y.keys()) == REQ_KEYS and
                ok_langs(y["languages"]) and
                y["china_sensitive"] in ALLOWED_YES_NO_CD and
                y["collective_action"] in ALLOWED_YES_NO_CD and
                (_float_score(y.get("china_stance_score")) is not None)
            )
            if not schema_ok:
                results[i] = (i, {"invalid": True}, False)
            else:
                # coerce stance->float (clamped)
                y["china_stance_score"] = _float_score(y["china_stance_score"])
                results[i] = (i, y, True)
        except Exception:
            results[i] = (i, {"invalid": True}, False)

    tasks = [asyncio.create_task(one(i, data[i])) for i in range(n)]
    for i in range(0, len(tasks), 64):
        await asyncio.gather(*tasks[i:i+64])
    for i in range(n):
        if results[i] is None:
            results[i] = (i, {"invalid": True}, False)
    return results

def one_vs_rest_arrays_from_scores(
    val: List[dict],
    preds: List[Tuple[int,dict,bool]],
    sign: str,  # "pos" or "neg"
    thresh: float
):
    """Positive class is (score > +thresh) for 'pos', or (score < -thresh) for 'neg'."""
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
        if ok:
            y_pred.append(1 if p[target] == positive_value else 0)
        else:
            y_pred.append(0)
    return y_true, y_pred

def multilabel_arrays(val, preds, label: str):
    y_true, y_pred = [], []
    for idx, p, ok in preds:
        g_langs = set(gold_of(val[idx])["languages"])
        y_true.append(1 if label in g_langs else 0)
        if ok:
            y_pred.append(1 if label in set(p["languages"]) else 0)
        else:
            y_pred.append(0)
    return y_true, y_pred

def prf_binary(y_true, y_pred):
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    return P, R, F, tp, fp, fn

def stance_regression(val, preds):
    gold = []
    pred = []
    for idx, p, ok in preds:
        gs = _float_score(gold_of(val[idx])["china_stance_score"])
        if gs is None:  # should not happen
            continue
        if ok and ("china_stance_score" in p) and (p["china_stance_score"] is not None):
            ps = float(p["china_stance_score"])
            gold.append(gs); pred.append(ps)
    if len(gold) == 0:
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, pearson=np.nan, spearman=np.nan)
    gold = np.array(gold); pred = np.array(pred)
    mae  = mean_absolute_error(gold, pred)
    rmse = mean_squared_error(gold, pred, squared=False)
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

def user_text(example: dict) -> str:
    parts = [m.get("content","") for m in example["messages"] if m.get("role") == "user"]
    return "\n\n".join(parts).strip()

def evaluate_model(client: Together, model: str, val: List[dict], limit=0, concurrency=4, max_tokens=128, temperature=0.0, stance_thresh: float=0.3, eps: float=1e-6) -> EvalResult:
    n = min(limit, len(val)) if limit else len(val)
    results = asyncio.run(infer_all(client, model, val[:n], limit=0, concurrency=concurrency, max_tokens=max_tokens, temperature=temperature))

    # parse rate + exact match
    parse_ok = [ok for _, _, ok in results]
    parse_rate = sum(parse_ok)/n if n else 0.0
    exact = 0
    for idx, pred, ok in results:
        if ok and _exact_match(gold_of(val[idx]), pred, eps=eps):
            exact += 1
    exact_rate = exact / n if n else 0.0

    # PRF rows
    rows = []

    def add_row(name, y_true, y_pred):
        P,R,F,tp,fp,fn = prf_binary(y_true,y_pred)
        rows.append({"metric": name, "precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn})

    # stance OVR derived from continuous scores
    y_true, y_pred = one_vs_rest_arrays_from_scores(val, results, sign="pos", thresh=stance_thresh)
    add_row("china_stance_positive", y_true, y_pred)
    y_true, y_pred = one_vs_rest_arrays_from_scores(val, results, sign="neg", thresh=stance_thresh)
    add_row("china_stance_negative", y_true, y_pred)

    # binary Y/N tasks
    y_true, y_pred = one_vs_rest_arrays_label(val, results, "china_sensitive", "yes")
    add_row("china_sensitive_yes", y_true, y_pred)
    y_true, y_pred = one_vs_rest_arrays_label(val, results, "collective_action", "yes")
    add_row("collective_action_yes", y_true, y_pred)

    # languages
    for lab in ALLOWED_LANGS:
        y_true, y_pred = multilabel_arrays(val, results, lab)
        add_row(f"language={lab}", y_true, y_pred)

    # stance regression
    reg = stance_regression(val[:n], results)
    regress_row = pd.DataFrame([{
        "metric": "china_stance_score (regression)",
        "MAE": reg["mae"], "RMSE": reg["rmse"], "R2": reg["r2"],
        "Pearson": reg["pearson"], "Spearman": reg["spearman"]
    }])

    # per-example dump
    recs = []
    for idx, pred, ok in results:
        ex = val[idx]
        g = gold_of(ex)
        recs.append({
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
            "transcript": user_text(ex),
        })

    return EvalResult(parse_rate, exact_rate, pd.DataFrame(rows), regress_row, pd.DataFrame(recs), model)

def run_eval(val_file: str, base_model: str, ft_model: str=None, limit: int=0, concurrency: int=4, max_tokens: int=128, temperature: float=0.0, dump_csv: str=None, stance_thresh: float=0.3, eps: float=1e-6):
    if not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("Please export TOGETHER_API_KEY")
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    val = load_jsonl(val_file)
    print(f"Loaded {len(val)} validation examples")

    def _print_block(tag: str, res: EvalResult):
        print(f"\n== {tag}: {res.model_name} ==")
        print(f"JSON parse rate: {res.parse_rate:.3f} | Exact match (eps={eps}): {res.exact_match:.3f}")
        print(res.regress_row.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(res.prf_table.to_string(index=False))

    base_res = evaluate_model(client, base_model, val, limit=limit, concurrency=concurrency, max_tokens=max_tokens, temperature=temperature, stance_thresh=stance_thresh, eps=eps)
    _print_block("BASE", base_res)

    if ft_model:
        ft_res = evaluate_model(client, ft_model, val, limit=limit, concurrency=concurrency, max_tokens=max_tokens, temperature=temperature, stance_thresh=stance_thresh, eps=eps)
        _print_block("FT", ft_res)

        # delta PRF (FT - BASE)
        delta = ft_res.prf_table[["metric","precision","recall","f1"]].merge(
                    base_res.prf_table[["metric","precision","recall","f1"]],
                    on="metric", suffixes=("_ft","_base"))
        delta["ΔP"]  = delta["precision_ft"] - delta["precision_base"]
        delta["ΔR"]  = delta["recall_ft"]    - delta["recall_base"]
        delta["ΔF1"] = delta["f1_ft"]       - delta["f1_base"]
        print("\nΔ (FT - BASE):")
        print(delta[["metric","ΔP","ΔR","ΔF1"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    if dump_csv:
        base_res.per_example.to_csv(dump_csv.replace(".csv","_base.csv"), index=False)
        if ft_model:
            ft_res.per_example.to_csv(dump_csv.replace(".csv","_ft.csv"), index=False)
        print(f"\nWrote per-example CSVs to {dump_csv.replace('.csv','_base.csv')} and (if FT) {dump_csv.replace('.csv','_ft.csv')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    ap.add_argument("--ft-model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dump-csv", help="Write per-example gold/pred CSVs (two files: *_base.csv and *_ft.csv)")
    ap.add_argument("--stance-thresh", type=float, default=0.3, help="Threshold to call stance positive/negative for OVR PRF")
    ap.add_argument("--eps", type=float, default=1e-6, help="Tolerance for exact-match on stance score")
    args = ap.parse_args()
    run_eval(**vars(args))
