"""
python parse.py \
  --raw out/preds_base.raw.jsonl \
  --out out/preds_base.parsed.jsonl \
  --print-bad 5

"""
# parse.py
import os, sys, json, argparse
from json_utils import extract_json_best, valid_schema

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _pbar(n, desc):
    if _HAS_TQDM:
        try:
            from tqdm import tqdm
            return tqdm(total=n, desc=desc, unit="ex")
        except Exception:
            pass
    class _Dummy:
        def update(self, *a, **k): pass
        def close(self): pass
    return _Dummy()

def run(raw_path: str, out_path: str, print_bad: int, limit: int):
    lines = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    if limit:
        lines = lines[:limit]

    parsed_ok = 0
    pbar = _pbar(len(lines), "Parse")
    bad_log = open("bad_outputs.log", "w", encoding="utf-8")
    out_fh = open(out_path, "w", encoding="utf-8")

    for rec in lines:
        idx = rec["idx"]
        raw = rec.get("raw", "")
        obj, ok = extract_json_best(raw)
        if ok and valid_schema(obj):
            out = {"idx": idx, "parsed": True, "json": obj, "raw": raw}
            parsed_ok += 1
        else:
            out = {"idx": idx, "parsed": False, "json": None, "raw": raw}
            bad_log.write(f"IDX={idx}\n{raw}\n---\n")
        out_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        pbar.update(1)

    pbar.close()
    out_fh.close()
    bad_log.close()

    # Print first K bads to stdout (optional)
    if print_bad > 0:
        shown = 0
        with open("bad_outputs.log", "r", encoding="utf-8") as fh:
            block = []
            for line in fh:
                if line.strip() == "---":
                    print("".join(block).rstrip())
                    print("---")
                    shown += 1
                    if shown >= print_bad:
                        break
                    block = []
                else:
                    block.append(line)

    print(f"Done. Parsed {parsed_ok}/{len(lines)} = {parsed_ok/len(lines):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="jsonl from infer.py")
    ap.add_argument("--out", required=True, help="parsed jsonl")
    ap.add_argument("--print-bad", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(**vars(args))
