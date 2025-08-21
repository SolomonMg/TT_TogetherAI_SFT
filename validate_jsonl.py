"""
# just validate
python validate_jsonl.py data/train_BAL.jsonl data/val_BAL.jsonl

# fix in place
python validate_jsonl.py --fix data/val_BAL.jsonl

# write cleaned copies
python validate_jsonl.py --out-dir cleaned/ data/train_BAL.jsonl data/val_BAL.jsonl
python


"""

#!/usr/bin/env python3
# validate_jsonl.py — continuous stance + optional cleaning
import argparse, json, sys, re, os
from pathlib import Path

ALLOWED_YES_NO_CD = {"yes","no","cannot_determine"}
ALLOWED_LANGS     = {"english","mandarin","spanish","other","no_language"}
REQ               = {"china_stance_score","china_sensitive","collective_action","languages"}

def ok_langs(x):
    return (isinstance(x, list) and len(x) > 0
            and all(isinstance(s, str) and s in ALLOWED_LANGS for s in x)
            and (("no_language" not in x) or (len(x) == 1)))

def ok_score(v):
    try:
        f = float(v)
    except Exception:
        return False
    return -1.0 <= f <= 1.0

def clean_text(s: str):
    """Return (cleaned_text, had_cr_or_u202x, bad_line_numbers)."""
    bad_lines = []
    out_lines = []
    had_bad = False
    for i, line in enumerate(s.splitlines(keepends=True), 1):
        bad = ("\r" in line) or bool(re.search(r'[\u2028\u2029]', line))
        if bad:
            had_bad = True
            bad_lines.append(i)
            line = line.replace("\r\n", "\n").replace("\r", "\n")
            line = re.sub(r'[\u2028\u2029]', '', line)
        out_lines.append(line)
    # Ensure final file ends with \n only if original did; JSONL doesn’t require trailing newline
    cleaned = "".join(out_lines)
    return cleaned, had_bad, bad_lines

def validate_string(content: str, path_label: str):
    errs = 0; n = 0
    for i, line in enumerate(content.splitlines(), 1):
        n += 1
        # structural JSONL object
        try:
            obj = json.loads(line)
        except Exception as e:
            errs += 1; print(f"{path_label}: Line {i}: invalid JSON — {e}")
            continue

        msgs = obj.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            errs += 1; print(f"{path_label}: Line {i}: bad messages"); continue
        roles = [m.get("role") for m in msgs]
        if roles[:2] != ["system","user"] or roles[-1] != "assistant":
            errs += 1; print(f"{path_label}: Line {i}: bad role order {roles}")

        asst = msgs[-1].get("content","")
        try:
            y = json.loads(asst)
        except Exception:
            errs += 1; print(f"{path_label}: Line {i}: assistant not JSON"); continue

        if set(y.keys()) != REQ:
            errs += 1; print(f"{path_label}: Line {i}: keys {sorted(y.keys())} != {sorted(REQ)}"); continue

        if not ok_score(y["china_stance_score"]):
            errs += 1; print(f"{path_label}: Line {i}: china_stance_score must be float in [-1,1]"); continue

        if y["china_sensitive"] not in ALLOWED_YES_NO_CD:
            errs += 1; print(f"{path_label}: Line {i}: china_sensitive not in {sorted(ALLOWED_YES_NO_CD)}")
        if y["collective_action"] not in ALLOWED_YES_NO_CD:
            errs += 1; print(f"{path_label}: Line {i}: collective_action not in {sorted(ALLOWED_YES_NO_CD)}")

        if not ok_langs(y["languages"]):
            errs += 1; print(f"{path_label}: Line {i}: languages invalid")
    status = "OK" if errs == 0 else "FAIL"
    print(f"[{status}] {path_label} | lines={n} errors={errs}")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Validate SFT JSONL (continuous stance). Optionally clean line endings.")
    ap.add_argument("files", nargs="+", help="JSONL files to validate")
    ap.add_argument("--fix", action="store_true", help="Rewrite cleaned content in place (CRLF/CR→LF, strip U+2028/U+2029)")
    ap.add_argument("--out-dir", type=Path, help="Write cleaned copies to this directory (original files untouched)")
    args = ap.parse_args()

    if args.fix and args.out_dir:
        print("Choose either --fix (in place) or --out-dir (write copies), not both.", file=sys.stderr)
        sys.exit(2)

    any_fail = False
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"[MISS] {p} (not found)")
            any_fail = True
            continue

        raw = p.read_text(encoding="utf-8", errors="strict")
        cleaned, had_bad, bad_lines = clean_text(raw)

        # report unusual terminators if present
        if had_bad:
            print(f"{p}: found unusual line terminators on lines {bad_lines[:10]}{' ...' if len(bad_lines)>10 else ''}")

        # write out if requested
        write_target = None
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            write_target = args.out_dir / p.name
        elif args.fix and had_bad:
            write_target = p

        if write_target is not None:
            write_target.write_text(cleaned, encoding="utf-8")
            print(f"Wrote cleaned file → {write_target}")

        # validate the (cleaned) content so your run doesn't fail just because of terminators
        errs = validate_string(cleaned, str(p if write_target is None else write_target))
        any_fail |= (errs > 0)

    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()
