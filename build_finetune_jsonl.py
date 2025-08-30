#!/usr/bin/env python3
"""
build_finetune_jsonl.py

Simplified JSONL builder for evaluation/training. Single responsibility:
- Input: Single CSV/Parquet with labels + text content (already merged)
- Output: JSONL with OpenAI chat format

Usage:
    python build_finetune_jsonl.py \
        --input data/china_labeling_sample_all_Jul30_merged.csv \
        --output data/val_merged.jsonl

Input file must contain:
- meta_id: unique identifier
- china_stance_score: float in [-1, 1] 
- sensitive: numeric (0-1 range)
- collective_action: numeric (0-1 range) [optional]
- Text content in one of these column combinations:
  * subtitle + meta_desc, OR
  * transcript + description

Output JSONL format:
{
  "meta_id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "TRANSCRIPT:...\\nDESCRIPTION:..."},
    {"role": "assistant", "content": "{\\"china_stance_score\\":...}"}
  ]
}
"""

import json
import argparse

import pandas as pd
from json_utils import (
    load_table, norm_cols, to_str_meta, clamp11, safe_text, 
    write_jsonl
)

SYSTEM_PROMPT = (
    "You are a meticulous labeling assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
    "CODEBOOK — DEFINITIONS & TASKS\n"
    "china_sensitive - does the video include mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
    "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
    "of Xi Jinping or the CCP; corruption in China.\n\n"
    "china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look."
    "Use the full range; 0 = neutral/unclear.\n\n"
    "Do not default to the most frequent class; base labels on explicit evidence. Use cannot_determine only when evidence is insufficient."
    "LABEL THE VIDEO ON TWO DIMENSIONS:\n"
    "1) china_stance_score — a float in [-1, 1]\n"
    "2) china_sensitive — 'yes' | 'no' | 'cannot_determine'\n\n"
    "FORMAT RULES\n"
    "• Output ONLY a minified JSON object with keys: china_stance_score, china_sensitive.\n"
    "• china_stance_score must be a number in [-1, 1].\n"
    "• Use 'cannot_determine' when unsure. Do not add extra keys or prose."
)

def build_user_text(transcript: str, description: str) -> str:
    """Combine transcript and description into user message format."""
    parts = []
    if transcript.strip():
        parts.append(f"TRANSCRIPT:\n{transcript.strip()}")
    if description.strip():
        parts.append(f"DESCRIPTION:\n{description.strip()}")
    return "\n\n".join(parts)

def labelize_sensitive(value: float, thresh: float = 0.5) -> str:
    """Convert numeric sensitive value to label."""
    if pd.isna(value):
        return "cannot_determine"
    return "yes" if float(value) >= thresh else "no"

def process_file(input_path: str, output_path: str, yn_thresh: float = 0.5):
    """Process single labeled file into JSONL format."""
    print(f"[info] Loading {input_path}")
    df = norm_cols(load_table(input_path))
    
    # Check for meta_id column (flexible naming)
    meta_col = None
    for col in ["meta_id", "id"]:
        if col in df.columns:
            meta_col = col
            break
    if not meta_col:
        raise SystemExit("[error] Input file missing meta_id or id column")
    
    # Check for required label columns
    required_cols = ["china_stance_score", "sensitive"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] Missing required columns: {missing}")
    
    # Check for text content columns (flexible naming)
    transcript_col = None
    desc_col = None
    
    # Try different column name variations
    for col in df.columns:
        col_lower = col.lower()
        if "subtitle" in col_lower or "transcript" in col_lower:
            transcript_col = col
        elif "desc" in col_lower or "description" in col_lower:
            desc_col = col
    
    if not transcript_col and not desc_col:
        raise SystemExit("[error] No text content columns found (need subtitle/transcript and/or meta_desc/description)")
    
    print(f"[info] Using text columns: transcript='{transcript_col}', description='{desc_col}'")
    print(f"[info] Processing {len(df)} rows")
    
    rows = []
    for _, r in df.iterrows():
        meta_id = to_str_meta(r[meta_col])
        if not meta_id:
            continue
            
        # Extract text content
        transcript = safe_text(r.get(transcript_col, "")) if transcript_col else ""
        description = safe_text(r.get(desc_col, "")) if desc_col else ""
        user_text = build_user_text(transcript, description)
        
        if not user_text.strip():
            print(f"[warning] Empty text content for meta_id {meta_id}, skipping")
            continue
        
        # Build gold JSON
        stance = clamp11(r["china_stance_score"])
        if pd.isna(stance):
            print(f"[warning] Invalid stance score for meta_id {meta_id}, skipping")
            continue
            
        sensitive_label = labelize_sensitive(r["sensitive"], yn_thresh)
        
        gold = {
            "china_stance_score": float(stance),
            "china_sensitive": sensitive_label
        }
        
        # Add collective_action if present
        if "collective_action" in df.columns and not pd.isna(r["collective_action"]):
            collective_label = labelize_sensitive(r["collective_action"], yn_thresh)
            gold["collective_action"] = collective_label
        
        # Create messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))}
        ]
        
        rows.append({
            "meta_id": meta_id,
            "messages": messages
        })
    
    print(f"[info] Successfully processed {len(rows)} examples")
    write_jsonl(output_path, rows)

def main():
    parser = argparse.ArgumentParser(description="Convert labeled CSV/Parquet to JSONL format")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file with labels and text")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--yn-thresh", type=float, default=0.5, 
                       help="Threshold for converting numeric labels to yes/no (default: 0.5)")
    
    args = parser.parse_args()
    
    process_file(args.input, args.output, args.yn_thresh)

if __name__ == "__main__":
    main()