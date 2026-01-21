#!/usr/bin/env python3
"""
parse.py — tolerant JSON extractor for model outputs

Purpose
-------
Takes raw model outputs produced by infer.py (JSONL with fields like
{"idx": ..., "raw": "...", ...}), extracts the assistant's final JSON,
lightly sanitizes common glitches, validates against the codebook schema,
and writes a parsed JSONL with either the parsed object or a failure.

Supports group-aware validation for per-group prompts (--group-mode).

Usage
-----
python parse.py \
  --raw out/preds_base.raw.jsonl \
  --out out/preds_base.parsed.jsonl \
  --print-bad 5

# Group-aware parsing (validates only expected keys per group)
python parse.py \
  --raw out/preds_pergroup.raw.jsonl \
  --out out/preds_pergroup.parsed.jsonl \
  --group-mode by-category \
  --print-bad 5

python parse.py \
  --raw out/preds_minimal.raw.jsonl \
  --out out/preds_minimal.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_ft.raw.jsonl \
  --out out/preds_ft.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_llama3.1-70B.raw.jsonl \
  --out out/preds_llama3.1-70B.parsed.jsonl \
  --print-bad 5

python parse.py \
  --raw out/preds_llama3.3-70B.raw.jsonl \
  --out out/preds_llama3.3-70B.parsed.jsonl \
  --print-bad 5
  """

import os, sys, json, re, argparse
from json_utils import ALLOWED_YES_NO_CD, ALLOWED_LANGS, REQ_KEYS, load_jsonl
from prompt_groups import (
    get_group_configs, parse_compound_id, validate_group_schema,
    GroupConfig, DIMENSIONS
)

# --- helpers -------------------------------------------------------------

def ok_langs(x):
    return (isinstance(x, list) and len(x) > 0
            and all(isinstance(s, str) and s in ALLOWED_LANGS for s in x)
            and (("no_language" not in x) or (len(x) == 1)))

def valid_yn_field(x) -> bool:
    """Accept either string labels or numeric [0,1] values"""
    if x in ALLOWED_YES_NO_CD:
        return True
    try:
        f = float(x)
        return 0.0 <= f <= 1.0
    except:
        return False

def valid_yes_no_only(x) -> bool:
    """Accept yes/no labels or numeric [0,1] values (no cannot_determine)"""
    if x in {"yes", "no"}:
        return True
    try:
        f = float(x)
        return 0.0 <= f <= 1.0
    except:
        return False

def valid_schema(y: dict, group: GroupConfig = None, numeric_labels: bool = False) -> bool:
    """Validate schema, optionally for a specific group.

    Args:
        y: The parsed JSON object
        group: Optional GroupConfig for partial validation (group mode)
        numeric_labels: Whether numeric labels are expected (for group validation)

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(y, dict): return False

    # If group is specified, use group-specific validation
    if group is not None:
        return validate_group_schema(y, group, numeric_labels)

    # Check for comprehensive categorical format (new)
    comprehensive_categorical_keys = {
        "china_ccp_government", "china_people_culture", "china_technology_development",
        "china_sensitive", "collective_action", "hate_speech", "harmful_content",
        "news_segments", "inauthentic_content", "derivative_content", "china_related"
    }

    if comprehensive_categorical_keys.issubset(y.keys()):
        # Validate comprehensive categorical format
        china_related = y.get("china_related")
        
        # china_related only accepts yes/no (not cannot_determine)
        if not valid_yes_no_only(china_related):
            return False
        
        # China stance dimensions must always be valid stance values
        # (The post-processing will convert to empty strings when china_related='no')
        allowed_stance_values = {"pro", "anti", "neutral/unclear"}
        for key in ["china_ccp_government", "china_people_culture", "china_technology_development"]:
            val = y.get(key)
            if val not in allowed_stance_values:
                return False
        
        # china_sensitive validation - must always be valid yes/no/cannot_determine
        # (The post-processing will convert to empty string when china_related='no')
        china_sensitive = y.get("china_sensitive")
        if not valid_yn_field(china_sensitive):
            return False

        # Other dimensions are always required (not conditional on china_related)
        for key in ["collective_action", "hate_speech", "harmful_content",
                   "news_segments", "inauthentic_content", "derivative_content"]:
            if not valid_yn_field(y.get(key)):
                return False
        
        return True

    # Check for partial group match (any subset of comprehensive keys)
    # This handles outputs from group-mode where only some dimensions are present
    all_comprehensive_keys = set(DIMENSIONS.keys())
    present_keys = set(y.keys()) & all_comprehensive_keys
    if present_keys and not comprehensive_categorical_keys.issubset(y.keys()):
        # Partial comprehensive output - validate only present keys
        allowed_stance_values = {"pro", "anti", "neutral", "cannot_determine"}
        stance_keys = {"china_ccp_government", "china_people_culture", "china_technology_development"}
        for key in present_keys:
            if key in stance_keys:
                if y.get(key) not in allowed_stance_values:
                    return False
            else:
                if not valid_yn_field(y.get(key)):
                    return False
        return True

    # Check for traditional format (legacy)
    required_keys = {"china_stance_score", "china_sensitive"}
    if not required_keys.issubset(y.keys()): return False

    try:
        s = float(y.get("china_stance_score"))
    except Exception:
        return False
    if not (-1.0 <= s <= 1.0): return False
    if not valid_yn_field(y.get("china_sensitive")): return False

    # Optional fields that should be numeric [0,1] if present
    optional_numeric_fields = [
        "collective_action", "inauthentic_content", "hate_speech",
        "harmful_content", "news_segments"
    ]

    for field in optional_numeric_fields:
        if field in y:
            if not valid_yn_field(y.get(field)):
                return False

    # Optional: languages (if present, must be valid)
    if "languages" in y:
        if not ok_langs(y.get("languages")): return False

    return True

# strip code fences if present
_CODEFENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.M)

def _strip_code_fence(text: str) -> str:
    m = _CODEFENCE_RE.search(text)
    return m.group(1) if m else text

# scan for balanced {...} blocks and return the LAST that parses
def _last_balanced_json_anywhere(s: str):
    last = None
    n = len(s)
    i = 0
    while i < n:
        i = s.find('{', i)
        if i == -1: break
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
                        last = cand
                        i = j + 1
                        break
            j += 1
        else:
            break
    return last

# --- enhanced sanitizers for common parsing issues ----------------------

# Fix double decimals in any numeric field (e.g., 0.0.0 -> 0.0)
_FIX_DOUBLE_DECIMAL_RE = re.compile(r'(\d+\.\d+)\.0+(?=[\s,}\]])')

# Fix malformed keys like "china_stance_score:0" -> "china_stance_score":0
_FIX_MALFORMED_KEY_RE = re.compile(r'"([^"]+):([^"]*)"(\s*:\s*)"([^"]*)"')

# Fix missing quotes on string values like "china_stance_score:0,"
_FIX_UNQUOTED_KEY_VALUE_RE = re.compile(r'"([^"]+):([^"]*)"')

# Fix truncated JSON like "collective_action": or ending with incomplete values
_FIX_TRUNCATED_VALUE_RE = re.compile(r':\s*$|:\s*,')

# Fix space-separated values (just numbers) like "0.0 0.5 0.2 0.0 0.0 0.0 0.0"
_SPACE_SEPARATED_VALUES_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?){6})\s*$')

# Fix missing quotes in key names like harmful0 -> "harmful_content"
_FIX_MALFORMED_KEYS_RE = re.compile(r'([,{]\s*)(harmful0|hate0|news0|inauthentic0)(\s*:)')

# Fix incomplete JSON objects ending with trailing characters like {"...,
_INCOMPLETE_JSON_RE = re.compile(r'^\{[^}]*,\s*$')

def _fix_double_decimal_bug(s: str) -> str:
    """Fix double decimals like 0.0.0 -> 0.0 and scientific notation like 0.4.0e-1 -> 0.04"""
    # Fix scientific notation like 0.4.0e-1 -> 0.04
    s = re.sub(r'(\d+\.\d+)\.0e-1', lambda m: str(float(m.group(1)) * 0.1), s)
    
    # Fix regular double decimals like 0.0.0 -> 0.0
    s = _FIX_DOUBLE_DECIMAL_RE.sub(r'\1', s)
    
    return s

def _fix_malformed_keys(s: str) -> str:
    """
    Fix malformed keys like:
    - "china_stance_score:0":"0" -> "china_stance_score":0
    - "china_stance_score:0," -> "china_stance_score":0,
    - "inauthentic:0," -> "inauthentic_content":0,
    """
    # Fix specific pattern: "china_stance_score:0":"0" -> "china_stance_score":0
    s = re.sub(r'"china_stance_score:0":"0"', '"china_stance_score":0', s)
    
    # Fix pattern like: "key:value" where the whole thing is in quotes
    # This handles cases like "china_stance_score:0,"
    s = re.sub(r'"([a-z_]+):([^"]*)"(\s*:\s*)"([^"]*)"', r'"\1":\2', s)
    s = re.sub(r'"([a-z_]+):([^"]*)"(?=\s*[,}])', r'"\1":\2', s)
    
    # Fix missing key name quotes and incomplete key names
    s = re.sub(r'"inauthentic:([^"]*)"', r'"inauthentic_content":\1', s)
    s = re.sub(r'"hate:([^"]*)"', r'"hate_speech":\1', s)
    s = re.sub(r'"harmful:([^"]*)"', r'"harmful_content":\1', s)
    s = re.sub(r'"news:([^"]*)"', r'"news_segments":\1', s)
    
    # Fix bare key names without quotes followed by colon value
    s = re.sub(r'([,{]\s*)inauthentic:(\d)', r'\1"inauthentic_content":\2', s)
    s = re.sub(r'([,{]\s*)hate:(\d)', r'\1"hate_speech":\2', s)
    s = re.sub(r'([,{]\s*)harmful:(\d)', r'\1"harmful_content":\2', s)
    s = re.sub(r'([,{]\s*)news:(\d)', r'\1"news_segments":\2', s)
    
    # Fix the specific malformed pattern: "inauthentic:0,
    s = re.sub(r'"inauthentic:0,', r'"inauthentic_content":0,', s)
    
    # Fix common typos in key names
    s = re.sub(r'([,{]\s*)(harmful0)(\s*:)', r'\1"harmful_content"\3', s)
    s = re.sub(r'([,{]\s*)(hate0)(\s*:)', r'\1"hate_speech"\3', s)
    s = re.sub(r'([,{]\s*)(news0)(\s*:)', r'\1"news_segments"\3', s)
    s = re.sub(r'([,{]\s*)(inauthentic0)(\s*:)', r'\1"inauthentic_content"\3', s)
    
    # Fix missing key names like just "china_stance" instead of "china_stance_score"
    s = re.sub(r'"china_stance"(\s*:)', r'"china_stance_score"\1', s)
    
    return s

def _fix_truncated_json(s: str) -> str:
    """Fix truncated JSON by removing incomplete trailing parts and attempting to close"""
    # Remove trailing colons with no values
    s = _FIX_TRUNCATED_VALUE_RE.sub('', s)
    
    # Handle various truncation patterns
    # 1. Ends with just a number (missing closing brace)
    if re.match(r'^\{[^}]*\d+\s*$', s):
        s = s.strip() + '}'
    
    # 2. Ends with comma (remove comma and close)
    elif re.match(r'^\{[^}]*,\s*$', s):
        s = s.rstrip(',').strip() + '}'
    
    # 3. Ends with incomplete key-value pair like "key":value (missing closing)
    elif re.match(r'^\{[^}]*"[^"]*":\d+\.?\d*\s*$', s):
        s = s.strip() + '}'
    
    return s

def _convert_space_separated_to_json(s: str) -> str:
    """Convert space-separated values like '0.0 0.5 0.2 0.0 0.0 0.0 0.0' to proper JSON"""
    match = _SPACE_SEPARATED_VALUES_RE.match(s)
    if match:
        values = match.group(1).split()
        if len(values) == 7:
            keys = ["china_stance_score", "china_sensitive", "collective_action", 
                   "inauthentic_content", "hate_speech", "harmful_content", "news_segments"]
            json_pairs = [f'"{key}":{val}' for key, val in zip(keys, values)]
            return '{' + ','.join(json_pairs) + '}'
    return s

def _fix_missing_quotes_in_strings(s: str) -> str:
    """Fix missing quotes around '...' values"""
    return re.sub(r'"\.\.\."|"…"', '"truncated"', s)

# Optional: remove trailing commas before } or ]
_TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')
def _strip_trailing_commas(s: str) -> str:
    return _TRAILING_COMMA_RE.sub(r'\1', s)

def try_parse_json_with_fixes(s: str):
    """Attempt json.loads; if it fails, apply comprehensive fixes then retry."""
    # First, try parsing as-is
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Apply fixes in order of frequency/likelihood
    original_s = s
    
    # 1. Check if it's space-separated values and convert to JSON
    s = _convert_space_separated_to_json(s)
    if s != original_s:
        try:
            return json.loads(s)
        except Exception:
            pass
    
    # 2. Fix double decimals (most common issue)
    s = _fix_double_decimal_bug(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 3. Fix malformed keys and values (do this early as it catches many patterns)
    s = _fix_malformed_keys(s)
    
    # 3a. Also fix this specific edge case that wasn't caught above
    s = re.sub(r'"inauthentic:0,"', r'"inauthentic_content":0,"', s)
    
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 4. Fix truncated JSON
    s = _fix_truncated_json(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 5. Fix missing quotes in string values
    s = _fix_missing_quotes_in_strings(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # 6. Remove trailing commas (final cleanup)
    s = _strip_trailing_commas(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Give up - couldn't fix it
    raise

def standardize_json_keys(obj: dict) -> dict:
    """Standardize JSON keys to canonical comprehensive format"""
    if not isinstance(obj, dict):
        return obj

    # Define canonical key mappings for legacy format
    legacy_key_mappings = {
        'china_stance_score': ['china_stance_score', 'china_stance', 'collective_stance_score', 'harmful_stance_score'],
        'china_sensitive': ['china_sensitive', 'collective_sensitive', 'hate_sensitivity', 'hate_sens', 'hate_senstive'],
        'collective_action': [
            'collective_action', 'collective', 'collective action', 'collective content',
            'collective or collective_action', 'collective or portrays collective action',
            'collective,', 'collective?', 'collective_action0', 'collective_action:',
            'collective_action?', 'collective_action_action', 'collective_action_score',
            'collective_action_stance_score', 'collective_content', 'collective_score'
        ],
        'inauthentic_content': [
            'inauthentic_content', 'ina', 'ina0uthentic_content', 'inaction_content',
            'inauth', 'inauth...', 'inauth0', 'inauth?', 'inauth_content', 'inauth_speech',
            'inauthan_content', 'inauthent_content', 'inauthentic', 'inauthentic content',
            'inauthentic or content', 'inauthentic,', 'inauthentic?', 'inauthentic_content?',
            'inauthentic_stance_score', 'inauthful_content', 'inauthorentic_content', 'inhuman_content'
        ],
        'hate_speech': [
            'hate_speech', 'hate speech', 'hate_s or speech', 'hate_s0peech', 'hate_s?',
            'hate_scor', 'hate_sense', 'hate_senspeech', 'hate_situation', 'hate_speech,',
            'hate_speech?', 'hate_speech_content', 'hate_speech_score', 'hateful_content',
            'and_hate_speech'
        ],
        'harmful_content': [
            'harmful_content', 'harm', 'harm?', 'harmful', 'harmful (??)', 'harmful action',
            'harmful content', 'harmful stance_score', 'harmful0', 'harmful?', 'harmful_ content',
            'harmful_chinese_content', 'harmful_content?', 'harmful_sensitivty', 'harmful_stance_score',
            'harmfulfill_harmful_content', 'harmfulfilling_content', 'harmfulld_content',
            'harmonious_content', 'china_harmful_content'
        ],
        'news_segments': [
            'news_segments', 'news', 'news segments', 'news0segments', 'news_segments0'
        ]
    }

    # Define comprehensive categorical keys (new format)
    comprehensive_categorical_keys = {
        "china_ccp_government", "china_people_culture", "china_technology_development",
        "china_sensitive", "collective_action", "hate_speech", "harmful_content",
        "news_segments", "inauthentic_content", "derivative_content", "china_related"
    }

    # Create reverse mapping for legacy keys
    reverse_mapping = {}
    for canonical, variants in legacy_key_mappings.items():
        for variant in variants:
            reverse_mapping[variant] = canonical

    # Standardize the object
    standardized = {}
    for key, value in obj.items():
        # Skip invalid/junk keys
        if key in ['0', '0.0', '0.1', '0.4', '0.6', '1', '4', 'D'] or len(str(key)) > 100:
            continue

        # Check if it's a comprehensive categorical key (keep as-is)
        if key in comprehensive_categorical_keys:
            # Normalize stance values for the three China attitude dimensions
            if key in ["china_ccp_government", "china_people_culture", "china_technology_development"]:
                if isinstance(value, str):
                    val_lower = value.lower().strip()
                    # Map various neutral forms to canonical "neutral/unclear"
                    if val_lower in ["neutral", "unclear", "cannot_determine", "neutral/unclear", "unclear/neutral"]:
                        standardized[key] = "neutral/unclear"
                    elif val_lower in ["pro", "anti"]:
                        standardized[key] = val_lower
                    else:
                        standardized[key] = value  # Keep original if not recognized
                else:
                    standardized[key] = value
            else:
                standardized[key] = value
        else:
            # Try to map legacy/variant keys
            canonical_key = reverse_mapping.get(key, key)
            if canonical_key in legacy_key_mappings:
                standardized[canonical_key] = value

    return standardized

def apply_china_related_logic(obj: dict) -> dict:
    """Post-process parsed JSON to apply conditional logic for china_related.
    
    When china_related='no', convert China-specific fields to empty strings:
    - china_ccp_government
    - china_people_culture
    - china_technology_development
    - china_sensitive
    
    This moves the conditional logic from the model prompt to the parsing stage,
    simplifying the prompt and improving model adherence to the output schema.
    """
    if not isinstance(obj, dict):
        return obj
    
    # Check if this is comprehensive categorical format
    comprehensive_categorical_keys = {
        "china_ccp_government", "china_people_culture", "china_technology_development",
        "china_sensitive", "collective_action", "hate_speech", "harmful_content",
        "news_segments", "inauthentic_content", "derivative_content", "china_related"
    }
    
    if not comprehensive_categorical_keys.issubset(obj.keys()):
        # Not comprehensive format, return as-is
        return obj
    
    # Apply conditional logic
    china_related = obj.get("china_related")
    if china_related == "no":
        # Convert China-specific fields to empty strings
        obj["china_ccp_government"] = ""
        obj["china_people_culture"] = ""
        obj["china_technology_development"] = ""
        obj["china_sensitive"] = ""
    
    return obj

def extract_first_valid_json(text: str, group: GroupConfig = None, numeric_labels: bool = False):
    """
    Strategy:
    1) If whole string parses as JSON -> validate -> return
    2) Strip code fences; find LAST balanced {...} block -> try parse (+fixes) -> validate
    3) Look for incomplete JSON at the end and try to complete it
    Returns (obj, ok:bool, mode:str)

    Args:
        text: Raw model output text
        group: Optional GroupConfig for group-aware validation
        numeric_labels: Whether numeric labels are expected
    """
    if not isinstance(text, str) or not text:
        return {"invalid": True}, False, "none"

    # 1) Whole-string JSON (try with fixes first)
    try:
        obj = try_parse_json_with_fixes(text.strip())
        obj = standardize_json_keys(obj)  # Standardize keys
        if valid_schema(obj, group, numeric_labels):
            obj = apply_china_related_logic(obj)  # Apply conditional logic
            return obj, True, "strict"
    except Exception:
        pass

    # 2) Strip code fences and try again
    s = _strip_code_fence(text)
    try:
        obj = try_parse_json_with_fixes(s.strip())
        obj = standardize_json_keys(obj)  # Standardize keys
        if valid_schema(obj, group, numeric_labels):
            obj = apply_china_related_logic(obj)  # Apply conditional logic
            return obj, True, "codefence_stripped"
    except Exception:
        pass

    # 3) Find last balanced {...} block
    cand = _last_balanced_json_anywhere(s)
    if cand:
        try:
            obj = try_parse_json_with_fixes(cand)
            obj = standardize_json_keys(obj)  # Standardize keys
            if valid_schema(obj, group, numeric_labels):
                obj = apply_china_related_logic(obj)  # Apply conditional logic
                return obj, True, "balanced_block"
        except Exception:
            pass

    # 4) Look for JSON-like patterns at the end of verbose output
    # Find patterns like: ..."china_stance_score":0.6,"china_sensitive":1.0
    # or any dimension key pattern for group mode
    search_keys = ["china_stance_score", "china_ccp_government", "china_sensitive"]
    if group:
        search_keys = group.dimensions[:1]  # Use first dimension of group

    for search_key in search_keys:
        json_pattern = re.search(rf'\{{[^{{}}]*"{search_key}"[^{{}}]*\}}(?=[^{{}}]*$)', text, re.DOTALL)
        if json_pattern:
            try:
                obj = try_parse_json_with_fixes(json_pattern.group(0))
                obj = standardize_json_keys(obj)  # Standardize keys
                if valid_schema(obj, group, numeric_labels):
                    obj = apply_china_related_logic(obj)  # Apply conditional logic
                    return obj, True, "pattern_match"
            except Exception:
                pass

    return {"invalid": True}, False, "none"

# --- CLI ---------------------------------------------------------------

def run(raw: str, out: str, print_bad: int = 0, group_mode: str = None, numeric_labels: bool = False):
    """
    raw: path to raw outputs JSONL (from infer.py)
    out: where to write parsed JSONL
    group_mode: optional group mode for validation ('by-category', 'binary', 'per-item')
    numeric_labels: whether numeric labels are expected
    """
    raw_recs = load_jsonl(raw)
    ok = 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    bad_log = "data/bad_outputs.log"
    # truncate log
    open(bad_log, "w", encoding="utf-8").close()

    # Get group configs if group_mode is specified
    groups = None
    if group_mode and group_mode != "single":
        groups = get_group_configs(group_mode)
        print(f"[info] Group-aware parsing mode: {group_mode} ({len(groups)} groups)")

    with open(out, "w", encoding="utf-8") as fw:
        for rec in raw_recs:
            idx = rec.get("idx")
            meta_id = rec.get("meta_id", str(idx))  # Extract meta_id, fallback to idx
            text = rec.get("raw", "")

            # Determine group for this record (from compound ID or record field)
            group = None
            original_meta_id = meta_id
            group_id = rec.get("group_id")

            if group_id is None:
                # Try to parse from compound ID
                original_meta_id, group_id = parse_compound_id(meta_id)

            if groups and group_id and group_id in groups:
                group = groups[group_id]

            obj, parsed, mode = extract_first_valid_json(text, group, numeric_labels)

            if parsed:
                ok += 1
                out_rec = {
                    "idx": idx,
                    "meta_id": meta_id,
                    "original_meta_id": original_meta_id,
                    "parsed": True,
                    "raw": text,
                    "json": obj
                }
                if group_id:
                    out_rec["group_id"] = group_id
            else:
                out_rec = {
                    "idx": idx,
                    "meta_id": meta_id,
                    "original_meta_id": original_meta_id,
                    "parsed": False,
                    "raw": text,
                    "json": None
                }
                if group_id:
                    out_rec["group_id"] = group_id
                with open(bad_log, "a", encoding="utf-8") as bl:
                    bl.write(f"IDX={idx} META_ID={meta_id} GROUP={group_id}\n{text}\n---\n")
            fw.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    total = len(raw_recs)
    rate = ok / total if total else 0.0
    print(f"\nDone. Parsed {ok}/{total} = {rate:.3f}")
    print(f"Wrote: {out}")

    if print_bad > 0:
        if not os.path.exists(bad_log):
            print("\n(no bad_outputs.log found)")
        else:
            print(f"\n--- First {print_bad} bad outputs (bad_outputs.log) ---")
            shown = 0
            block = []
            with open(bad_log, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() == "---":
                        print("".join(block).rstrip())
                        print("---")
                        shown += 1
                        block = []
                        if shown >= print_bad:
                            break
                    else:
                        block.append(line)
            if shown == 0 and block:
                print("".join(block).rstrip())
                print("---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw model outputs (from infer.py)")
    ap.add_argument("--out", required=True, help="Where to write parsed JSONL")
    ap.add_argument("--print-bad", type=int, default=0, help="Show first K bad outputs from bad_outputs.log")
    ap.add_argument("--group-mode", default=None,
                   choices=["single", "by-category", "binary", "per-item"],
                   help="Group mode for validation (enables group-aware partial schema validation)")
    ap.add_argument("--numeric-labels", action="store_true",
                   help="Expect numeric labels instead of categorical")
    args = ap.parse_args()
    run(args.raw, args.out, args.print_bad, args.group_mode, args.numeric_labels)
