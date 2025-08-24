# json_utils.py
import json, re

ALLOWED_YES_NO_CD = {"yes", "no", "cannot_determine"}
ALLOWED_LANGS = ["english", "mandarin", "spanish", "other", "no_language"]
REQ_KEYS = {"china_stance_score", "china_sensitive", "collective_action", "languages"}

# ---------- file I/O ----------
def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            out.append(json.loads(line))
    return out

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

# ---------- schema ----------
def ok_langs(x):
    return (
        isinstance(x, list) and len(x) > 0 and
        all(isinstance(s, str) and s in ALLOWED_LANGS for s in x) and
        (("no_language" not in x) or (len(x) == 1))
    )

def valid_schema(y: dict) -> bool:
    if not isinstance(y, dict) or set(y.keys()) != REQ_KEYS:
        return False
    try:
        s = float(y.get("china_stance_score"))
    except Exception:
        return False
    if not (-1.0 <= s <= 1.0):
        return False
    if y.get("china_sensitive") not in ALLOWED_YES_NO_CD:
        return False
    if y.get("collective_action") not in ALLOWED_YES_NO_CD:
        return False
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
