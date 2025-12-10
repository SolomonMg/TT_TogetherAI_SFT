#!/usr/bin/env python3
"""
build_finetune_jsonl.py

Simplified JSONL builder for evaluation/training with support for multiple labeling modes.

Key Features:
- Input: Single CSV/Parquet with labels + text content (already merged)
- Output: JSONL with OpenAI chat format
- Supports 3 labeling modes: standard (3-dim), comprehensive numeric (8-dim), comprehensive categorical (10-dim)

Usage Examples:
    # Standard 3-dimension mode
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl

    # Comprehensive numeric mode (9 dimensions, 0-1 scale)
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl --comprehensive --numeric-labels

    # Comprehensive categorical mode (11 dimensions, categorical labels)
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl --comprehensive

    # Inference mode (no gold labels)
    python build_finetune_jsonl.py --input data/unlabeled.csv --output data/inference.jsonl --no-labels

Input file requirements:
- meta_id: unique identifier (or id, tt_video_id, yt_video_id, video_id)
- china_stance_score: float in [-1, 1] (for labeled mode)
- sensitive: numeric (0-1 range) (for labeled mode)
- collective_action: numeric (0-1 range) [optional] (for labeled mode)
- Text content in one of these column combinations:
  * subtitle/transcript + meta_desc/description

Output JSONL format:
{
  "meta_id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "TRANSCRIPT:...\\nDESCRIPTION:..."},
    {"role": "assistant", "content": "{\\"china_stance_score\\":...}"}  // Only in labeled mode
  ]
}

Labeling Modes:
1. Standard (default): 3 dimensions - china_stance_score (float), china_sensitive/collective_action (categorical)
2. Numeric comprehensive (--comprehensive --numeric-labels): 9 dimensions, all numeric 0-1 scale
3. Categorical comprehensive (--comprehensive): 11 dimensions with structured categorical labels
"""

import json
import argparse

import pandas as pd
from json_utils import (
    load_table, norm_cols, to_str_meta, clamp11, safe_text, 
    write_jsonl, is_valid_text
)

def get_system_prompt(numeric_labels: bool = False, comprehensive: bool = False) -> str:
    """Generate system prompt for different labeling modes.

    Args:
        numeric_labels: If True, use numeric (0-1) labels instead of categorical (yes/no/cannot_determine)
        comprehensive: If True, use comprehensive analysis with additional content dimensions

    Returns:
        System prompt string with appropriate labeling instructions

    Modes:
        - Default (False, False): Standard 3-dimension categorical labeling
        - (True, False): Standard 3-dimension numeric labeling
        - (False, True): Comprehensive 11-dimension categorical labeling
        - (True, True): Comprehensive 9-dimension numeric labeling
    """
    
    if comprehensive:
        if numeric_labels:
            return (
                "You are a meticulous content analysis assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
                "CODEBOOK — DEFINITIONS & TASKS\n"
                "1) china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look. Use the full range; 0 = neutral/unclear.\n\n"
                "2) china_sensitive - on a scale from 0 (no sensitive content) to 1 (definitely includes sensitive content), score whether the video includes mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule of Xi Jinping or the CCP; corruption in China.\n\n"
                "3) collective_action - on a scale from 0 (no collective action) to 1 (clear collective action), score whether this video encourages or portrays collective action.\n\n"
                "4) inauthentic_content - on a scale from 0 (authentic content) to 1 (clearly inauthentic), score whether the video contains: misinformation or false claims; conspiracy theories or unfounded speculation; undisclosed use of AI-generated content; deepfakes or manipulated media.\n\n"
                "5) hate_speech - on a scale from 0 (no hate speech) to 1 (clear hate speech), score whether the video contains: attacks on individuals based on race, ethnicity, religion, gender, sexual orientation, or other protected characteristics; promotion of hateful ideologies or groups; dehumanizing language toward specific communities.\n\n"
                "6) harmful_content - on a scale from 0 (not harmful) to 1 (clearly harmful), score whether the video contains: nudity, sexual content, or sexually suggestive material; graphic violence, gore, or disturbing imagery; promotion of dangerous activities or self-harm; illegal substance use or promotion.\n\n"
                "7) news_segments - on a scale from 0 (not news) to 1 (clearly news content), score whether the video contains: clips from professional news broadcasts; journalist reporting or news commentary; coverage of current events in news format; content from established news organizations.\n\n"
                "8) derivative_content - on a scale from 0 (original content) to 1 (clearly derivative), score whether less than 50% of the video adds original or meaningful commentary.\n\n"
                "9) china_related - on a scale from 0 (not related to China) to 1 (clearly related to China), score whether the video is related to the politics, culture, society, or economy of China.\n\n"
                "Do not default to the most frequent score; base labels on explicit evidence. Use the midpoint of the scale when uncertain.\n\n"
                "SCORE THE VIDEO ON NINE DIMENSIONS:\n"
                "1) china_stance_score — a float in [-1, 1]\n"
                "2) china_sensitive — a float in [0, 1]\n"
                "3) collective_action — a float in [0, 1]\n"
                "4) inauthentic_content — a float in [0, 1]\n"
                "5) hate_speech — a float in [0, 1]\n"
                "6) harmful_content — a float in [0, 1]\n"
                "7) news_segments — a float in [0, 1]\n"
                "8) derivative_content — a float in [0, 1]\n"
                "9) china_related — a float in [0, 1]\n"
                "FORMAT RULES\n"
                "• Output ONLY a minified JSON object with all nine keys.\n"
                "• All values must be numbers in their specified ranges.\n"
                "• Use the midpoint of the scale for uncertain cases. Do not add extra keys or prose."
            )
        else:
            return (
                "You are a meticulous content analysis assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
                "CODEBOOK — DEFINITIONS & TASKS\n\n"
                "1) china_related - Label 'yes' if the video is related to the politics, culture, society, or economy of China. Otherwise label 'no'.\n\n"
                "IMPORTANT: If china_related is 'no', code all China-specific variables (china_ccp_government, china_people_culture, china_technology_development, china_sensitive) as \"\" (empty string).\n\n"
                "ATTITUDES TOWARD CHINA (4 DIMENSIONS — only code if china_related = 'yes')\n"
                "Use these four separate sub-categories to distinguish the target of sentiment:\n\n"
                "2) china_ccp_government - Label the video's stance toward the CCP/PRC Government:\n"
                "• 'pro' - Supports or praises the CCP, PRC government, or leadership (e.g., 'China's system is more efficient than the West'). Supports PRC/CCP domestic or international policies or actions.\n"
                "• 'anti' - Criticizes or mocks the CCP, PRC leadership, or China's political system (e.g., censorship, authoritarianism). Against PRC/CCP domestic or international policies or actions.\n"
                "• 'neutral/unclear' - No clear evaluative stance, or mixed views.\n"
                "• \"\" (empty string) - When china_related is 'no'\n\n"
                "3) china_people_culture - Label the video's stance toward Chinese People/Culture:\n"
                "• 'pro' - Positive portrayal of Chinese citizens, traditions, or cultural achievements (e.g., cuisine, festivals, kindness of people).\n"
                "• 'anti' - Negative generalizations or hostility toward Chinese individuals or culture (e.g., 'Chinese people are ...').\n"
                "• 'neutral/unclear' - Cultural reference without judgment or sentiment not directed at people/culture.\n"
                "• \"\" (empty string) - When china_related is 'no'\n\n"
                "4) china_technology_development - Label the video's stance toward Chinese Technology/Development:\n"
                "• 'pro' - Praises Chinese technological progress, innovation, or infrastructure (e.g., 'China's AI development is impressive').\n"
                "• 'anti' - Criticizes Chinese technological products, companies, or policies (e.g., 'Huawei steals IP').\n"
                "• 'neutral/unclear' - Mentions technology or development without evaluative tone.\n"
                "• \"\" (empty string) - When china_related is 'no'\n\n"
                "5) china_sensitive - Label 'yes' if the video relates to any of the following topics:\n"
                "• Taiwan independence or pro-DPP party content\n"
                "• Hong Kong national security law or political repression\n"
                "• Independence movements or repression in Tibet or Xinjiang (Uyghurs)\n"
                "• The June 4 Tiananmen Square movement\n"
                "• Criticism, ridicule, or satire of Xi Jinping or the Chinese Communist Party (CCP)\n"
                "• Corruption or political scandals in China\n"
                "Otherwise label 'no'. Use 'cannot_determine' only when evidence is insufficient AND china_related is 'yes'.\n"
                "Use \"\" (empty string) when china_related is 'no'.\n\n"
                "GENERAL CONTENT DIMENSIONS (6 DIMENSIONS — code for all videos)\n\n"
                "6) collective_action - Label 'yes' if the video encourages or portrays the coordinated effort of a group of people to achieve a common goal (e.g., protests, strikes, boycotts).\n\n"
                "7) hate_speech - Label 'yes' if the content includes:\n"
                "• Attacks on protected groups (race, gender, religion, sexual orientation, disability, etc.)\n"
                "• Promotion of hateful ideologies or organizations (beliefs of supremacy, hateful conspiracies, or denying well-documented atrocities)\n"
                "• Dehumanization or claims of inferiority (comparing a group to animals or objects, portraying them as inherently dangerous or criminal, or claiming they are physically or mentally inferior)\n"
                "• Use of hateful slurs (language specifically associated with a protected attribute)\n\n"
                "8) harmful_content - Label 'yes' if the content includes:\n"
                "• Sensitive or mature themes (e.g., sexual activity, nudity, shocking/graphic content, animal abuse)\n"
                "• Regulated Goods and Commercial Activities (gambling, alcohol, firearms, illegal drugs, paid political marketing, frauds and scams)\n"
                "• Safety and Civility (promotion of violence, hate speech, violent organisations, sexual abuse, human trafficking, harrassment/bullying)\n"
                "• Youth endangerment or exploitation (e.g. sexual content involving minors)\n"
                "• Showing or promoting mental/behavioural health issues (e.g., suicide, self-harm, eating disorders) or dangerous activities/challenges\n"
                "• Privacy and Security (personal information, e.g., private phone numbers and addresses)\n"
                "Note: If labeled as hate speech, automatically also label as harmful content.\n\n"
                "9) news_segments - Label 'yes' if more than 50% of the video consists of:\n"
                "• Clips from TV or online news programs\n"
                "• Anchors or reporters narrating events\n"
                "• Overlays of news footage with captions\n\n"
                "10) inauthentic_content - Label 'yes' if the content appears to:\n"
                "• Contain misinformation or conspiracy theories (especially political or public concerns)\n"
                "• Use AI-generated media without disclosure in a deceptive way\n"
                "• Be coordinated, spammy, or manipulative\n\n"
                "11) derivative_content - Label 'yes' if less than 50% of the video adds original or meaningful commentary. Examples:\n"
                "• Reaction videos without commentary\n"
                "• Reposts of other TikTok content, or content from other platforms (look for watermarks, mirrored footage, or stitched reposts)\n\n"
                "Do not default to the most frequent label; base labels on explicit evidence. Use 'cannot_determine' when evidence is insufficient.\n\n"
                "LABEL THE VIDEO ON ELEVEN DIMENSIONS:\n"
                "1) china_related — 'yes' | 'no'\n"
                "2) china_ccp_government — 'pro' | 'anti' | 'neutral/unclear' | \"\" (empty string if china_related='no')\n"
                "3) china_people_culture — 'pro' | 'anti' | 'neutral/unclear' | \"\" (empty string if china_related='no')\n"
                "4) china_technology_development — 'pro' | 'anti' | 'neutral/unclear' | \"\" (empty string if china_related='no')\n"
                "5) china_sensitive — 'yes' | 'no' | 'cannot_determine' | \"\" (empty string if china_related='no')\n"
                "6) collective_action — 'yes' | 'no' | 'cannot_determine'\n"
                "7) hate_speech — 'yes' | 'no' | 'cannot_determine'\n"
                "8) harmful_content — 'yes' | 'no' | 'cannot_determine'\n"
                "9) news_segments — 'yes' | 'no' | 'cannot_determine'\n"
                "10) inauthentic_content — 'yes' | 'no' | 'cannot_determine'\n"
                "11) derivative_content — 'yes' | 'no' | 'cannot_determine'\n"
                "FORMAT RULES\n"
                "• Output ONLY a minified JSON object with all eleven keys.\n"
                "• Use exact values as specified above.\n"
                "• When china_related is 'no', use \"\" (empty string) for china_ccp_government, china_people_culture, china_technology_development, and china_sensitive.\n"
                "• Do not add extra keys or prose."
            )
    elif numeric_labels:
        return (
            "You are a meticulous labeling assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
            "CODEBOOK — DEFINITIONS & TASKS\n"
            "1) china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look.\n\n"
            "Use the full range; 0 = neutral/unclear.\n\n"
            "2) china_sensitive - on a scale from 0 (no sensitive content) to 1 (definitely includes sensitive content), score whether the video includes mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
            "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
            "of Xi Jinping or the CCP; corruption in China. Use 0.5 when uncertain.\n\n"
            "3) collective_action - on a scale from 0 (no collective action) to 1 (clear collective action), score whether this video encourages or portrays collective action.\n\n"
            "Do not default to the most frequent score; base labels on explicit evidence. Use 0.5 when uncertain.\n\n"
            "SCORE THE VIDEO ON THREE DIMENSIONS:\n"
            "1) china_stance_score — a float in [-1, 1]\n"
            "2) china_sensitive — a float in [0, 1]\n"
            "3) collective_action — a float in [0, 1]\n"
            "FORMAT RULES\n"
            "• Output ONLY a minified JSON object with keys: china_stance_score, china_sensitive, collective_action.\n"
            "• All values must be numbers in their specified ranges.\n"
            "• Use the center value for uncertain cases. Do not add extra keys or prose."
        )
    else:
        return (
            "You are a meticulous labeling assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
            "CODEBOOK — DEFINITIONS & TASKS\n"
            "1) china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look.\n\n"
            "Use the full range; 0 = neutral/unclear.\n\n"
            "2) china_sensitive - does the video include mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
            "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
            "of Xi Jinping or the CCP; corruption in China.\n\n"
            "3) collective_action - Does this video encourage or portray collective action?\n\n"
            "Do not default to the most frequent score/class; base labels on explicit evidence. Use cannot_determine only when evidence is insufficient.\n\n"
            "SCORE THE VIDEO ON THREE DIMENSIONS:\n"
            "1) china_stance_score — a float in [-1, 1]\n"
            "2) china_sensitive — 'yes' | 'no' | 'cannot_determine'\n"
            "3) collective_action — 'yes' | 'no' | 'cannot_determine'\n"
            "FORMAT RULES\n"
            "• Output ONLY a minified JSON object with keys: china_stance_score, china_sensitive, collective_action.\n"
            "• china_stance_score must be a number in [-1, 1].\n"
            "• Use 'cannot_determine' when unsure. Do not add extra keys or prose."
        )

def build_user_text(transcript: str, description: str) -> str:
    """Combine transcript and description into user message format."""
    parts = []
    
    # Include any non-empty text content
    if transcript and transcript.strip():
        parts.append(f"TRANSCRIPT:\n{transcript.strip()}")
    if description and description.strip():
        parts.append(f"DESCRIPTION:\n{description.strip()}")
        
    return "\n\n".join(parts)

def labelize_sensitive(value: float, thresh: float = 0.5) -> str:
    """Convert numeric sensitive value to label."""
    if pd.isna(value):
        return "cannot_determine"
    return "yes" if float(value) >= thresh else "no"

def process_file(
    input_path: str,
    output_path: str,
    yn_thresh: float = 0.5,
    min_text_len: int = 10,
    label_mode: bool = True,
    numeric_labels: bool = False,
    comprehensive: bool = False,
    strip_meta_id: bool = False,
):
    """Process single labeled file into JSONL format.

    strip_meta_id: omit meta_id column in output (Together FT rejects extra top-level keys)
    """
    print(f"[info] Loading {input_path}")
    df = norm_cols(load_table(input_path))
    
    # Check for meta_id column (flexible naming)
    meta_col = None
    for col in ["meta_id", "id", "tt_video_id", "yt_video_id", "video_id"]:
        if col in df.columns:
            meta_col = col
            break
    if not meta_col:
        raise SystemExit("[error] Input file missing meta_id, id, tt_video_id, yt_video_id, or video_id column")
    
    # Check for required label columns (only in label mode)
    if label_mode:
        if comprehensive:
            # Comprehensive mode requires different columns based on the CSV structure
            # Required: china_related and the yes/no dimensions
            required_cols = ["china_related", "collective_action", "hate_speech", 
                           "harmful", "news", "inauthentic", 
                           "derivative"]
            # China-specific columns (required if china_related exists)
            china_cols = ["stance_gov", "stance_culture", "stance_tech", "sensitive"]
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise SystemExit(f"[error] Missing required columns for comprehensive mode: {missing}")
            
            # Check for china stance columns (warn if missing but don't fail)
            missing_china = [c for c in china_cols if c not in df.columns]
            if missing_china:
                print(f"[warning] Missing China stance columns: {missing_china}")
        else:
            # Standard mode requires china_stance_score
            required_cols = ["china_stance_score", "sensitive"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise SystemExit(f"[error] Missing required columns: {missing}")
    
    # Check for text content columns (flexible naming)
    transcript_col = None
    desc_col = None
    
    # Try different column name variations (prioritize exact matches)
    for col in df.columns:
        col_lower = col.lower()
        if "subtitle" in col_lower or "transcript" in col_lower:
            transcript_col = col
        elif col_lower in ["meta_desc", "description", "processed_desc"]:
            desc_col = col

    # Fallback to any column with "desc" if no exact match found
    if not desc_col:
        for col in df.columns:
            col_lower = col.lower()
            if "desc" in col_lower:
                desc_col = col
                break
    
    if not transcript_col and not desc_col:
        raise SystemExit("[error] No text content columns found (need subtitle/transcript and/or meta_desc/description)")
    
    print(f"[info] Using text columns: transcript='{transcript_col}', description='{desc_col}'")
    print(f"[info] Text filtering: minimum {min_text_len} characters combined text")
    print(f"[info] Processing {len(df)} rows")
    
    rows = []
    filtered_count = 0
    
    for _, r in df.iterrows():
        meta_id = to_str_meta(r[meta_col])
        if not meta_id:
            filtered_count += 1
            continue
            
        # Extract text content
        transcript = safe_text(r.get(transcript_col, "")) if transcript_col else ""
        description = safe_text(r.get(desc_col, "")) if desc_col else ""
        user_text = build_user_text(transcript, description)
        
        # Check combined text length
        if not user_text.strip() or not is_valid_text(user_text, min_text_len):
            print(f"[warning] Insufficient text content for meta_id {meta_id} (< {min_text_len} chars), skipping")
            filtered_count += 1
            continue
        
        # Create messages
        messages = [
            {"role": "system", "content": get_system_prompt(numeric_labels, comprehensive)},
            {"role": "user", "content": user_text}
        ]
        
        # Add assistant message with gold labels only in label mode
        if label_mode:
            if comprehensive:
                # Build comprehensive gold JSON
                gold = {}
                
                # China-related flag (required)
                gold["china_related"] = str(r.get("china_related", "")).strip()
                
                # China stance dimensions (required)
                gold["china_ccp_government"] = str(r.get("stance_gov", "")).strip()
                gold["china_people_culture"] = str(r.get("stance_culture", "")).strip()
                gold["china_technology_development"] = str(r.get("stance_tech", "")).strip()
                gold["china_sensitive"] = str(r.get("sensitive", "")).strip()
                
                # Other dimensions (always present)
                gold["collective_action"] = str(r.get("collective_action", "")).strip()
                gold["hate_speech"] = str(r.get("hate_speech", "")).strip()
                gold["harmful_content"] = str(r.get("harmful", "")).strip()
                gold["news_segments"] = str(r.get("news", "")).strip()
                gold["inauthentic_content"] = str(r.get("inauthentic", "")).strip()
                gold["derivative_content"] = str(r.get("derivative", "")).strip()
                
                messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))})
            else:
                # Build standard gold JSON
                stance = clamp11(r["china_stance_score"])
                if pd.isna(stance):
                    print(f"[warning] Invalid stance score for meta_id {meta_id}, skipping")
                    filtered_count += 1
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
                
                messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))})
        
        
        if strip_meta_id:
            rows.append({"messages": messages})
        else:
            rows.append({
                "meta_id": meta_id,
                "messages": messages
            })
    
    print(f"[info] Successfully processed {len(rows)} examples")
    print(f"[info] Filtered out {filtered_count} rows due to missing/insufficient data")
    write_jsonl(output_path, rows)

def main():
    parser = argparse.ArgumentParser(description="Convert CSV/Parquet to JSONL format for training or inference")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file with text content")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--yn-thresh", type=float, default=0.5, 
                       help="Threshold for converting numeric labels to yes/no (default: 0.5)")
    parser.add_argument("--min-text-len", type=int, default=10,
                       help="Minimum combined text length to include row (default: 10)")
    parser.add_argument("--label-mode", action="store_true", default=True,
                       help="Include gold standard labels in output (default: true)")
    parser.add_argument("--no-labels", action="store_false", dest="label_mode",
                       help="Skip gold standard labels for inference-only data")
    parser.add_argument("--numeric-labels", action="store_true",
                       help="Use numeric labels (0-1) instead of categorical (yes/no/cannot_determine)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Use comprehensive content analysis prompt with 11 dimensions (categorical) or 9 dimensions (numeric) including china_related, inauthentic_content, hate_speech, harmful_content, derivative_content, and news_segments")
    parser.add_argument("--strip-meta-id", action="store_true",
                       help="Omit meta_id from each JSON line (Together FT requires only messages)")
    
    args = parser.parse_args()
    
    process_file(
        args.input,
        args.output,
        args.yn_thresh,
        args.min_text_len,
        args.label_mode,
        args.numeric_labels,
        args.comprehensive,
        args.strip_meta_id,
    )

if __name__ == "__main__":
    main()