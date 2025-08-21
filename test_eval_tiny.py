#!/usr/bin/env python3
import json, argparse
from pathlib import Path

# IMPORTANT: we import your existing helpers, not re-write them
from eval_model_perf_val_jsonl import (
    load_jsonl, gold_of, one_vs_rest_arrays, multilabel_arrays, prf_binary
)

def read_preds_jsonl(path: str):
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            preds.append((i, json.loads(line), True))  # (idx, pred_dict, parse_ok=True)
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/val_tiny.jsonl")
    ap.add_argument("--base", default="data/base_preds.jsonl")
    ap.add_argument("--ft",   default="data/ft_preds.jsonl")
    args = ap.parse_args()

    val = load_jsonl(args.val)
    base = read_preds_jsonl(args.base)
    ft   = read_preds_jsonl(args.ft)

    # ---- BASE: should be perfect on positives present ----
    # china_stance_positive (good vs rest): positives: row 0 only
    y_true, y_pred = one_vs_rest_arrays(val, base, "china_stance", "good")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    # china_stance_negative (bad vs rest): positives: row 1 only
    y_true, y_pred = one_vs_rest_arrays(val, base, "china_stance", "bad")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    # china_sensitive_yes: positives: row 0 only
    y_true, y_pred = one_vs_rest_arrays(val, base, "china_sensitive", "yes")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    # collective_action_yes: positives: row 1 only
    y_true, y_pred = one_vs_rest_arrays(val, base, "collective_action", "yes")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    # Languages (each label vs rest)
    for lab, expect in [
        ("english",      (1.000,1.000,1.000)),   # present row 0
        ("mandarin",     (1.000,1.000,1.000)),   # present row 1
        ("no_language",  (1.000,1.000,1.000)),   # present row 2
        ("spanish",      (0.000,0.000,0.000)),   # not present anywhere
        ("other",        (0.000,0.000,0.000)),   # not present anywhere
    ]:
        y_true, y_pred = multilabel_arrays(val, base, lab)
        P,R,F, *_ = prf_binary(y_true, y_pred)
        assert (round(P,3),round(R,3),round(F,3)) == expect

    # ---- FT: expected degradations only where we flipped ----
    # stance positive (good): flipped row 0 -> FN; P=0.0, R=0.0, F=0.0
    y_true, y_pred = one_vs_rest_arrays(val, ft, "china_stance", "good")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (0.000,0.000,0.000)

    # stance negative (bad): unchanged (row 1 still bad) -> stays perfect
    y_true, y_pred = one_vs_rest_arrays(val, ft, "china_stance", "bad")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    # collective_action yes: flipped row 1 yes->no -> FN; P=0.0, R=0.0, F=0.0
    y_true, y_pred = one_vs_rest_arrays(val, ft, "collective_action", "yes")
    P,R,F, *_ = prf_binary(y_true, y_pred); assert (round(P,3),round(R,3),round(F,3)) == (0.000,0.000,0.000)

    # languages: english unchanged perfect, mandarin unchanged perfect, no_language unchanged perfect
    for lab in ["english","mandarin","no_language"]:
        y_true, y_pred = multilabel_arrays(val, ft, lab)
        P,R,F, *_ = prf_binary(y_true, y_pred)
        assert (round(P,3),round(R,3),round(F,3)) == (1.000,1.000,1.000)

    print("Tiny test passed (logic & alignment look correct)")

if __name__ == "__main__":
    main()
