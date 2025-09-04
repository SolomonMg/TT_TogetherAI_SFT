# json_utils.py
"""
Utility functions for data processing in the TikTok SFT pipeline.

This module provides:
- File I/O functions for JSONL, CSV, and Parquet files
- Data processing utilities for cleaning and normalizing data
- JSON extraction and validation functions
- Schema validation for model outputs
- Functions for merging model predictions back with original data

Example usage for merging predictions with CSV:
    python -c "
    from json_utils import merge_predictions_with_csv
    merge_predictions_with_csv(
        csv_path='data/df_for_label_removal_2.csv',
        parsed_jsonl_path='out/preds_label_removal_2.parsed.jsonl', 
        output_path='data/df_for_label_removal_2_with_preds.csv'
    )
    "
"""
import json, re, os
import pandas as pd
from typing import Optional, List

ALLOWED_YES_NO_CD = {"yes", "no", "cannot_determine"}
ALLOWED_LANGS = ["english", "mandarin", "spanish", "other", "no_language"]
REQ_KEYS = {"china_stance_score", "china_sensitive", "collective_action", "languages"}

# ---------- file I/O ----------
def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file with consistent error handling and line processing."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = re.sub(r'[\u2028\u2029]', '', line).rstrip("\r\n")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}: line {ln} invalid JSON: {e}")
    return out

def write_jsonl(path: str, rows: list) -> None:
    """Write JSONL file with consistent formatting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {path}  n={len(rows)}")

def load_table(path: str) -> pd.DataFrame:
    """Load CSV or Parquet file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def gold_of(example: dict) -> dict:
    # Ground truth is the last assistant message's content (JSON string)
    return json.loads(example["messages"][-1]["content"])

def prompt_messages(example: dict) -> list[dict]:
    # Use dataset messages as-is, EXCEPT drop the gold assistant at the end
    msgs = example["messages"][:-1]
    # Keep only role+content (SDK doesn't need other keys)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

def user_text(example: dict) -> str:
    parts = [m.get("content", "") for m in example["messages"] if m.get("role") == "user"]
    return "\n\n".join(parts).strip()

# ---------- data processing ----------
def clamp01(x) -> Optional[float]:
    """Clamp value to [0, 1] range with percentage handling."""
    try:
        v = float(x)
        if v > 1.0 and v <= 100.0:  # allow percentages
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return None

def clamp11(x) -> Optional[float]:
    """Clamp value to [-1, 1] range."""
    try:
        v = float(x)
        return max(-1.0, min(1.0, v))
    except Exception:
        return None

def safe_text(x) -> str:
    """Coerce any value (including NaN/float) to a clean text string."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def is_valid_text(text: str, min_len: int = 10) -> bool:
    """Check if text is valid (not empty, NA, NULL, or too short)."""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip().lower()
    
    # Filter out common missing value indicators
    invalid_values = {"", "na", "null", "none", "n/a", "nan", "missing"}
    if text in invalid_values:
        return False
    
    # Check minimum length
    if len(text) < min_len:
        return False
        
    return True

def to_str_meta(x) -> str:
    """Convert meta_id to string, handling floats like 123.0 -> 123."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s

def yesno_to_label(x) -> str:
    """Convert various inputs to yes/no/cannot_determine labels."""
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ALLOWED_YES_NO_CD:
            return s
    return ""

# ---------- schema ----------
def ok_langs(x):
    return (
        isinstance(x, list) and len(x) > 0 and
        all(isinstance(s, str) and s in ALLOWED_LANGS for s in x) and
        (("no_language" not in x) or (len(x) == 1))
    )

def valid_yn_field(x) -> bool:
    """Accept either string labels or numeric [0,1] values"""
    if x in ALLOWED_YES_NO_CD:
        return True
    try:
        f = float(x)
        return 0.0 <= f <= 1.0
    except:
        return False

def valid_schema(y: dict) -> bool:
    if not isinstance(y, dict):
        return False
    
    # Required keys: china_stance_score, china_sensitive
    required_keys = {"china_stance_score", "china_sensitive"}
    if not required_keys.issubset(y.keys()):
        return False
    
    try:
        s = float(y.get("china_stance_score"))
    except Exception:
        return False
    if not (-1.0 <= s <= 1.0):
        return False
    if not valid_yn_field(y.get("china_sensitive")):
        return False
    
    # Optional: collective_action (if present, must be valid)
    if "collective_action" in y:
        if not valid_yn_field(y.get("collective_action")):
            return False
    
    # Optional: languages (if present, must be valid)
    if "languages" in y:
        if not ok_langs(y.get("languages")):
            return False
    
    return True

# ---------- robust JSON extraction ----------
_CODEFENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.M)
_TAIL_JSON_RE = re.compile(r"(\{[\s\S]*\})\s*$", re.S)

def _strip_codefence(s: str) -> str:
    m = _CODEFENCE_RE.search(s)
    return m.group(1) if m else s

def _try_whole(s: str):
    try:
        obj = json.loads(s)
        return obj if valid_schema(obj) else None
    except Exception:
        return None

def _try_tail(s: str):
    m = _TAIL_JSON_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj if valid_schema(obj) else None
    except Exception:
        return None

def _last_balanced_anywhere(s: str):
    last = None
    n = len(s)
    i = 0
    while i < n:
        i = s.find('{', i)
        if i == -1:
            break
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        cand = s[i:j+1]
                        try:
                            obj = json.loads(cand)
                            if valid_schema(obj):
                                last = obj
                        except Exception:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break
    return last

def extract_json_best(text: str) -> tuple[dict, bool]:
    """whole → codefence+tail → last-balanced"""
    if not isinstance(text, str) or not text.strip():
        return {"invalid": True}, False

    obj = _try_whole(text)
    if obj is not None:
        return obj, True

    s = _strip_codefence(text)
    obj = _try_tail(s)
    if obj is not None:
        return obj, True

    obj = _last_balanced_anywhere(s)
    if obj is not None:
        return obj, True

    return {"invalid": True}, False

def merge_predictions_with_csv(csv_path: str, parsed_jsonl_path: str, output_path: str) -> None:
    """
    Merge model predictions back with original CSV data.
    
    Args:
        csv_path: Path to original CSV file with meta_id column
        parsed_jsonl_path: Path to parsed predictions JSONL (from parse.py)
        output_path: Path to write merged CSV with prediction columns
    """
    import os
    
    # Load original CSV
    print(f"[info] Loading original CSV: {csv_path}")
    df = load_table(csv_path)
    df = norm_cols(df)  # Normalize column names
    
    if "meta_id" not in df.columns:
        raise RuntimeError("[error] Original CSV missing meta_id column")
    
    # Load parsed predictions
    print(f"[info] Loading parsed predictions: {parsed_jsonl_path}")
    parsed_data = load_jsonl(parsed_jsonl_path)
    
    # Create mapping of meta_id -> predictions
    pred_map = {}
    for row in parsed_data:
        if row.get("parsed") and row.get("json"):
            meta_id = str(row.get("meta_id", ""))
            if meta_id:
                pred_map[meta_id] = row["json"]
    
    print(f"[info] Found predictions for {len(pred_map)} examples")
    
    # Add prediction columns
    df["pred_china_stance_score"] = None
    df["pred_china_sensitive"] = None
    df["pred_collective_action"] = None
    df["pred_parsed"] = False
    
    # Merge predictions by meta_id
    matched = 0
    for idx, row in df.iterrows():
        meta_id = str(row["meta_id"])
        if meta_id in pred_map:
            pred = pred_map[meta_id]
            df.at[idx, "pred_china_stance_score"] = pred.get("china_stance_score")
            df.at[idx, "pred_china_sensitive"] = pred.get("china_sensitive") 
            df.at[idx, "pred_collective_action"] = pred.get("collective_action")
            df.at[idx, "pred_parsed"] = True
            matched += 1
    
    print(f"[info] Matched predictions for {matched}/{len(df)} rows ({matched/len(df)*100:.1f}%)")
    
    # Convert meta_id to string to preserve precision for large integers
    df["meta_id"] = df["meta_id"].astype(str)
    
    # Write merged CSV with proper quoting to handle commas in text fields
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, quoting=1, escapechar='\\')  # quoting=1 = QUOTE_ALL
    print(f"[write] {output_path}  n={len(df)} (with {matched} predictions)")
