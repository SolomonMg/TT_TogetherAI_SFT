#!/usr/bin/env python3
"""
build_finetune_jsonl.py

Simplified JSONL builder for evaluation/training with support for multiple labeling modes.

Key Features:
- Input: Single CSV/Parquet with labels + text content (already merged)
- Output: JSONL with OpenAI chat format
- Supports 3 labeling modes: standard (3-dim), comprehensive numeric (8-dim), comprehensive categorical (10-dim)
- Supports per-group prompts for comprehensive mode (--group-mode)

Usage Examples:
    # Standard 3-dimension mode
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl

    # Comprehensive numeric mode (9 dimensions, 0-1 scale)
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl --comprehensive --numeric-labels

    # Comprehensive categorical mode (11 dimensions, categorical labels)
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl --comprehensive

    # Per-group comprehensive mode (splits into 4 groups by category)
    python build_finetune_jsonl.py --input data/merged.csv --output data/val.jsonl --comprehensive --group-mode by-category

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
  "meta_id": "...",  # or "{meta_id}__group_{group_id}" for group mode
  "group_id": "...",  # optional, present in group mode
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

Group Modes (for comprehensive only):
- single (default): All dimensions in one prompt
- by-category: 4 groups (A: China Attitudes, B: Political Sensitivity, C: Content Safety, D: Content Type)
- binary: 2 groups (china-related, content analysis)
- per-item: Each dimension as its own prompt

"""

import json
import argparse
from pathlib import Path

import pandas as pd
from json_utils import (
    load_table, norm_cols, to_str_meta, clamp11, safe_text,
    write_jsonl, is_valid_text
)
from prompt_groups import (
    get_group_configs, generate_group_prompt, make_compound_id,
    GroupConfig, DIMENSIONS
)
from image_helpers import get_frame_paths, encode_image_base64

def get_system_prompt(numeric_labels: bool = False, comprehensive: bool = False,
                      group: GroupConfig = None, include_images: bool = False) -> str:
    """Generate system prompt for different labeling modes.

    Args:
        numeric_labels: If True, use numeric (0-1) labels instead of categorical (yes/no/cannot_determine)
        comprehensive: If True, use comprehensive analysis with additional content dimensions
        group: Optional GroupConfig for per-group prompts (comprehensive mode only)
        include_images: If True, add vision context to the prompt

    Returns:
        System prompt string with appropriate labeling instructions

    Modes:
        - Default (False, False): Standard 3-dimension categorical labeling
        - (True, False): Standard 3-dimension numeric labeling
        - (False, True): Comprehensive 11-dimension categorical labeling
        - (True, True): Comprehensive 9-dimension numeric labeling
        - With group: Focused prompt for specific dimension group
    """

    # Vision prefix when images are included
    vision_prefix = ""
    if include_images:
        vision_prefix = (
            "You are analyzing TikTok videos using both visual and textual content. "
            "You will be given one or more representative frames from the video (in chronological order), "
            "along with the transcript and description. Analyze both the visual content across all frames "
            "and the text to determine the appropriate labels.\n\n"
        )

    # If group is specified, use the group-specific prompt generator
    if group is not None:
        return vision_prefix + generate_group_prompt(group, numeric_labels)
    
    if comprehensive:
        if numeric_labels:
            return vision_prefix + (
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
            return vision_prefix + (
                "You are a meticulous content analysis assistant for TikTok videos. Follow the CODEBOOK and output format exactly.\n\n"
                "CODEBOOK — DEFINITIONS & TASKS\n\n"
                "1) china_related - Label 'yes' if the video is related to the politics, culture, society, or economy of China. Otherwise label 'no'.\n\n"
                "ATTITUDES TOWARD CHINA (4 DIMENSIONS)\n"
                "Use these four separate sub-categories to distinguish the target of sentiment:\n\n"
                "2) china_ccp_government - Label the video's stance toward the CCP/PRC Government:\n"
                "• 'pro' - Supports or praises the CCP, PRC government, or leadership (e.g., 'China's system is more efficient than the West'). Supports PRC/CCP domestic or international policies or actions.\n"
                "• 'anti' - Criticizes or mocks the CCP, PRC leadership, or China's political system (e.g., censorship, authoritarianism). Against PRC/CCP domestic or international policies or actions.\n"
                "• 'neutral/unclear' - No clear evaluative stance, or mixed views.\n\n"
                "3) china_people_culture - Label the video's stance toward Chinese People/Culture:\n"
                "• 'pro' - Positive portrayal of Chinese citizens, traditions, or cultural achievements (e.g., cuisine, festivals, kindness of people).\n"
                "• 'anti' - Negative generalizations or hostility toward Chinese individuals or culture (e.g., 'Chinese people are ...').\n"
                "• 'neutral/unclear' - Cultural reference without judgment or sentiment not directed at people/culture.\n\n"
                "4) china_technology_development - Label the video's stance toward Chinese Technology/Development:\n"
                "• 'pro' - Praises Chinese technological progress, innovation, or infrastructure (e.g., 'China's AI development is impressive').\n"
                "• 'anti' - Criticizes Chinese technological products, companies, or policies (e.g., 'Huawei steals IP').\n"
                "• 'neutral/unclear' - Mentions technology or development without evaluative tone.\n\n"
                "5) china_sensitive - Label 'yes' if the video relates to any of the following topics:\n"
                "• Taiwan independence or pro-DPP party content\n"
                "• Hong Kong national security law or political repression\n"
                "• Independence movements or repression in Tibet or Xinjiang (Uyghurs)\n"
                "• The June 4 Tiananmen Square movement\n"
                "• Criticism, ridicule, or satire of Xi Jinping or the Chinese Communist Party (CCP)\n"
                "• Corruption or political scandals in China\n"
                "Otherwise label 'no'. Use 'cannot_determine' only when evidence is insufficient.\n\n"
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
                "2) china_ccp_government — 'pro' | 'anti' | 'neutral/unclear'\n"
                "3) china_people_culture — 'pro' | 'anti' | 'neutral/unclear'\n"
                "4) china_technology_development — 'pro' | 'anti' | 'neutral/unclear'\n"
                "5) china_sensitive — 'yes' | 'no' | 'cannot_determine'\n"
                "6) collective_action — 'yes' | 'no' | 'cannot_determine'\n"
                "7) hate_speech — 'yes' | 'no' | 'cannot_determine'\n"
                "8) harmful_content — 'yes' | 'no' | 'cannot_determine'\n"
                "9) news_segments — 'yes' | 'no' | 'cannot_determine'\n"
                "10) inauthentic_content — 'yes' | 'no' | 'cannot_determine'\n"
                "11) derivative_content — 'yes' | 'no' | 'cannot_determine'\n"
                "FORMAT RULES\n"
                "• Output ONLY a minified JSON object with all eleven keys.\n"
                "• Use exact values as specified above.\n"
                "• Do not add extra keys or prose."
            )
    elif numeric_labels:
        return vision_prefix + (
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
        return vision_prefix + (
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


def build_user_content(
    transcript: str, 
    description: str, 
    frames_dir: Path | None = None, 
    video_id: str | None = None,
    include_images: bool = False
) -> str | list:
    """Build user message content, optionally including images.
    
    Args:
        transcript: Video transcript text
        description: Video description text
        frames_dir: Directory containing frame folders (required if include_images=True)
        video_id: Video/aweme ID (required if include_images=True)
        include_images: Whether to include images in the content
        
    Returns:
        Either a plain string (text-only) or a list of content parts (multimodal)
    """
    # Build the text content
    text_content = build_user_text(transcript, description)
    
    if not include_images or frames_dir is None or video_id is None:
        return text_content
    
    # Get all frames for this video
    frame_paths = get_frame_paths(frames_dir, video_id)
    
    if not frame_paths:
        print(f"[warning] No frames found for video {video_id}, using text-only")
        return text_content
    
    # Build multimodal content: images first, then text
    content_parts = []
    
    for frame_path in frame_paths:
        encoded = encode_image_base64(frame_path)
        if encoded:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": encoded}
            })
        else:
            print(f"[warning] Failed to encode frame {frame_path}")
    
    # Add text content at the end
    if text_content:
        content_parts.append({
            "type": "text",
            "text": text_content
        })
    
    # If no images were successfully encoded, fall back to text-only
    if not any(p.get("type") == "image_url" for p in content_parts):
        return text_content
    
    return content_parts


def labelize_sensitive(value: float, thresh: float = 0.5) -> str:
    """Convert numeric sensitive value to label."""
    if pd.isna(value):
        return "cannot_determine"
    return "yes" if float(value) >= thresh else "no"

def process_file(input_path: str, output_path: str, yn_thresh: float = 0.5,
                 min_text_len: int = 10, label_mode: bool = True,
                 numeric_labels: bool = False, comprehensive: bool = False,
                 group_mode: str = "single",strip_meta_id: bool = False,
                 include_images: bool = False, frames_dir: str | None = None):
    """Process single labeled file into JSONL format.

    Args:
        input_path: Input CSV or Parquet file
        output_path: Output JSONL file
        yn_thresh: Threshold for converting numeric labels to yes/no
        min_text_len: Minimum combined text length to include row
        label_mode: Include gold standard labels in output
        numeric_labels: Use numeric labels instead of categorical
        comprehensive: Use comprehensive content analysis prompt
        group_mode: Grouping mode for comprehensive prompts ('single', 'by-category', 'binary', 'per-item')
        strip_meta_id: omit meta_id column in output (Together FT rejects extra top-level keys)
    """
    # Validate group_mode is only used with comprehensive mode
    if group_mode != "single" and not comprehensive:
        print(f"[warning] --group-mode '{group_mode}' only applies to comprehensive mode, ignoring")
        group_mode = "single"
        
    # Validate image arguments
    if include_images and not frames_dir:
        raise SystemExit("[error] --frames-dir is required when --include-images is set")
    
    frames_path = Path(frames_dir) if frames_dir else None
    if include_images and not frames_path.exists():
        raise SystemExit(f"[error] Frames directory does not exist: {frames_dir}")

    print(f"[info] Loading {input_path}")
    if include_images:
        print(f"[info] Including images from: {frames_dir}")
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

    # Get group configurations
    groups = None
    if comprehensive and group_mode != "single":
        groups = get_group_configs(group_mode)
        print(f"[info] Group mode: {group_mode} ({len(groups)} groups: {list(groups.keys())})")
    elif comprehensive:
        # Single group with all dimensions
        groups = get_group_configs("single")

    rows = []
    filtered_count = 0

    def build_gold_labels(row: pd.Series) -> dict | None:
        if comprehensive:
            gold = {
                "china_related": str(row.get("china_related", "")).strip(),
                "china_ccp_government": str(row.get("stance_gov", "")).strip(),
                "china_people_culture": str(row.get("stance_culture", "")).strip(),
                "china_technology_development": str(row.get("stance_tech", "")).strip(),
                "china_sensitive": str(row.get("sensitive", "")).strip(),
                "collective_action": str(row.get("collective_action", "")).strip(),
                "hate_speech": str(row.get("hate_speech", "")).strip(),
                "harmful_content": str(row.get("harmful", "")).strip(),
                "news_segments": str(row.get("news", "")).strip(),
                "inauthentic_content": str(row.get("inauthentic", "")).strip(),
                "derivative_content": str(row.get("derivative", "")).strip(),
            }
            return gold

        stance = clamp11(row["china_stance_score"])
        if pd.isna(stance):
            return None

        sensitive_label = labelize_sensitive(row["sensitive"], yn_thresh)
        gold = {
            "china_stance_score": float(stance),
            "china_sensitive": sensitive_label,
        }

        if "collective_action" in df.columns and not pd.isna(row["collective_action"]):
            collective_label = labelize_sensitive(row["collective_action"], yn_thresh)
            gold["collective_action"] = collective_label

        return gold

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
        # Build user content (may include images if enabled)
        user_content = build_user_content(
            transcript, description,
            frames_dir=frames_path,
            video_id=meta_id,
            include_images=include_images
        )

        # For comprehensive mode with groups, generate one entry per group
        if comprehensive and groups:
            gold = None
            if label_mode:
                gold = build_gold_labels(r)
                if gold is None:
                    print(f"[warning] Invalid stance score for meta_id {meta_id}, skipping")
                    filtered_count += 1
                    continue

            for group_id, group in groups.items():
                messages = [
                    {"role": "system", "content": get_system_prompt(numeric_labels, comprehensive, group, include_images)},
                    {"role": "user", "content": user_content}
                ]

                if label_mode:
                    messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))})

                if group_mode != "single":
                    record_id = make_compound_id(meta_id, group_id)
                else:
                    record_id = meta_id

                if strip_meta_id:
                    rows.append({"messages": messages})
                else:
                    row_data = {
                        "meta_id": record_id,
                        "messages": messages
                    }

                    if group_mode != "single":
                        row_data["group_id"] = group_id
                        row_data["original_meta_id"] = meta_id

                    rows.append(row_data)
        else:
            messages = [
                {"role": "system", "content": get_system_prompt(numeric_labels, comprehensive, include_images=include_images)},
                {"role": "user", "content": user_content}
            ]

            if label_mode:
                gold = build_gold_labels(r)
                if gold is None:
                    print(f"[warning] Invalid stance score for meta_id {meta_id}, skipping")
                    filtered_count += 1
                    continue

                messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))})

            if strip_meta_id:
                rows.append({"messages": messages})
            else:
                rows.append({
                    "meta_id": meta_id,
                    "messages": messages
                })

    # Calculate summary stats
    source_records = len(df) - filtered_count
    if comprehensive and groups and group_mode != "single":
        print(f"[info] Successfully processed {source_records} source records into {len(rows)} entries ({len(groups)} groups per record)")
    else:
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
    parser.add_argument("--group-mode", default="single",
                       choices=["single", "by-category", "binary", "per-item"],
                       help="Grouping mode for comprehensive prompts (default: single). "
                            "by-category: 4 groups (China Attitudes, Political Sensitivity, Content Safety, Content Type). "
                            "binary: 2 groups (China-related, Content Analysis). "
                            "per-item: Each dimension as its own prompt.")
    parser.add_argument("--strip-meta-id", action="store_true",
                       help="Omit meta_id from each JSON line (Together FT requires only messages)")
    parser.add_argument("--include-images", action="store_true",
                       help="Include video frames in user messages for vision models")
    parser.add_argument("--frames-dir", type=str, default=None,
                       help="Directory containing frame folders (required if --include-images is set)")
    
    args = parser.parse_args()
    
    process_file(
        args.input,
        args.output,
        args.yn_thresh,
        args.min_text_len,
        args.label_mode,
        args.numeric_labels,
        args.comprehensive,
        args.group_mode,
        args.strip_meta_id,
        args.include_images,
        args.frames_dir,
    )

if __name__ == "__main__":
    main()