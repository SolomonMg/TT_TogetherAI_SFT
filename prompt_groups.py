#!/usr/bin/env python3
"""
prompt_groups.py — Dimension specifications and group configurations for per-group prompts.

This module provides:
- Dimension specifications (categorical vs numeric, value ranges)
- Group configurations (single, by-category, binary, per-item)
- Functions to generate focused system prompts per group

Groupings (by-category mode):
    A) China Attitudes: china_ccp_government, china_people_culture, china_technology_development
    B) Political Sensitivity: china_sensitive, collective_action
    C) Content Safety: hate_speech, harmful_content
    D) Content Type: news_segments, inauthentic_content, derivative_content
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class LabelType(Enum):
    """Label type for a dimension."""
    STANCE = "stance"  # pro/anti/neutral
    BINARY = "binary"  # yes/no/cannot_determine
    BINARY_STRICT = "binary_strict"  # yes/no only (no cannot_determine)
    NUMERIC_STANCE = "numeric_stance"  # float [-1, 1]
    NUMERIC_01 = "numeric_01"  # float [0, 1]


@dataclass
class DimensionSpec:
    """Specification for a single labeling dimension."""
    key: str
    name: str
    description: str
    categorical_type: LabelType
    numeric_type: LabelType
    format_categorical: str  # e.g., "'pro' | 'anti' | 'neutral'"
    format_numeric: str  # e.g., "a float in [-1, 1]"


# All dimension specifications
DIMENSIONS: Dict[str, DimensionSpec] = {
    "china_ccp_government": DimensionSpec(
        key="china_ccp_government",
        name="CCP/Government Stance",
        description=(
            "Label the video's stance toward the CCP/PRC Government:\n"
            "• 'pro' - Supports or praises the CCP, PRC government, or leadership (e.g., 'China's system is more efficient than the West'). Supports PRC/CCP domestic or international policies or actions.\n"
            "• 'anti' - Criticizes or mocks the CCP, PRC leadership, or China's political system (e.g., censorship, authoritarianism). Against PRC/CCP domestic or international policies or actions.\n"
            "• 'neutral/unclear' - No clear evaluative stance, or mixed views."
        ),
        categorical_type=LabelType.STANCE,
        numeric_type=LabelType.NUMERIC_STANCE,
        format_categorical="'pro' | 'anti' | 'neutral/unclear'",
        format_numeric="a float in [-1, 1] (-1=anti, 0=neutral/unclear, +1=pro)"
    ),
    "china_people_culture": DimensionSpec(
        key="china_people_culture",
        name="People/Culture Stance",
        description=(
            "Label the video's stance toward Chinese People/Culture:\n"
            "• 'pro' - Positive portrayal of Chinese citizens, traditions, or cultural achievements (e.g., cuisine, festivals, kindness of people).\n"
            "• 'anti' - Negative generalizations or hostility toward Chinese individuals or culture (e.g., 'Chinese people are ...').\n"
            "• 'neutral/unclear' - Cultural reference without judgment or sentiment not directed at people/culture."
        ),
        categorical_type=LabelType.STANCE,
        numeric_type=LabelType.NUMERIC_STANCE,
        format_categorical="'pro' | 'anti' | 'neutral/unclear'",
        format_numeric="a float in [-1, 1] (-1=anti, 0=neutral/unclear, +1=pro)"
    ),
    "china_technology_development": DimensionSpec(
        key="china_technology_development",
        name="Technology/Development Stance",
        description=(
            "Label the video's stance toward Chinese Technology/Development:\n"
            "• 'pro' - Praises Chinese technological progress, innovation, or infrastructure (e.g., 'China's AI development is impressive').\n"
            "• 'anti' - Criticizes Chinese technological products, companies, or policies (e.g., 'Huawei steals IP').\n"
            "• 'neutral/unclear' - Mentions technology or development without evaluative tone."
        ),
        categorical_type=LabelType.STANCE,
        numeric_type=LabelType.NUMERIC_STANCE,
        format_categorical="'pro' | 'anti' | 'neutral/unclear'",
        format_numeric="a float in [-1, 1] (-1=anti, 0=neutral/unclear, +1=pro)"
    ),
    "china_sensitive": DimensionSpec(
        key="china_sensitive",
        name="Political Sensitivity",
        description=(
            "Label 'yes' if the video relates to any of the following topics:\n"
            "• Taiwan independence or pro-DPP party content\n"
            "• Hong Kong national security law or political repression\n"
            "• Independence movements or repression in Tibet or Xinjiang (Uyghurs)\n"
            "• The June 4 Tiananmen Square movement\n"
            "• Criticism, ridicule, or satire of Xi Jinping or the Chinese Communist Party (CCP)\n"
            "• Corruption or political scandals in China"
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "collective_action": DimensionSpec(
        key="collective_action",
        name="Collective Action",
        description=(
            "Label 'yes' if the video encourages or portrays the coordinated effort of a group of people "
            "to achieve a common goal (e.g., protests, strikes, boycotts)."
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "hate_speech": DimensionSpec(
        key="hate_speech",
        name="Hate Speech",
        description=(
            "Label 'yes' if the content includes:\n"
            "• Attacks on protected groups (race, gender, religion, sexual orientation, disability, etc.)\n"
            "• Promotion of hateful ideologies or organizations (beliefs of supremacy, hateful conspiracies, or denying well-documented atrocities)\n"
            "• Dehumanization or claims of inferiority (comparing a group to animals or objects, portraying them as inherently dangerous or criminal, or claiming they are physically or mentally inferior)\n"
            "• Use of hateful slurs (language specifically associated with a protected attribute)"
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "harmful_content": DimensionSpec(
        key="harmful_content",
        name="Harmful Content",
        description=(
            "Label 'yes' if the content includes:\n"
            "• Sensitive or mature themes (e.g., sexual activity, nudity, shocking/graphic content, animal abuse)\n"
            "• Regulated Goods and Commercial Activities (gambling, alcohol, firearms, illegal drugs, paid political marketing, frauds and scams)\n"
            "• Safety and Civility (promotion of violence, hate speech, violent organisations, sexual abuse, human trafficking, harrassment/bullying)\n"
            "• Youth endangerment or exploitation (e.g. sexual content involving minors)\n"
            "• Showing or promoting mental/behavioural health issues (e.g., suicide, self-harm, eating disorders) or dangerous activities/challenges\n"
            "• Privacy and Security (personal information, e.g., private phone numbers and addresses)\n"
            "Note: If labeled as hate speech, automatically also label as harmful content."
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "news_segments": DimensionSpec(
        key="news_segments",
        name="News Segments",
        description=(
            "Label 'yes' if more than 50% of the video consists of:\n"
            "• Clips from TV or online news programs\n"
            "• Anchors or reporters narrating events\n"
            "• Overlays of news footage with captions"
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "inauthentic_content": DimensionSpec(
        key="inauthentic_content",
        name="Inauthentic Content",
        description=(
            "Label 'yes' if the content appears to:\n"
            "• Contain misinformation or conspiracy theories (especially political or public concerns)\n"
            "• Use AI-generated media without disclosure in a deceptive way\n"
            "• Be coordinated, spammy, or manipulative"
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "derivative_content": DimensionSpec(
        key="derivative_content",
        name="Derivative Content",
        description=(
            "Label 'yes' if less than 50% of the video adds original or meaningful commentary. Examples:\n"
            "• Reaction videos without commentary\n"
            "• Reposts of other TikTok content, or content from other platforms (look for watermarks, mirrored footage, or stitched reposts)"
        ),
        categorical_type=LabelType.BINARY,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no' | 'cannot_determine'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
    "china_related": DimensionSpec(
        key="china_related",
        name="China Related",
        description=(
            "Label 'yes' if the video is related to the politics, culture, society, or economy of China. "
            "Otherwise label 'no'."
        ),
        categorical_type=LabelType.BINARY_STRICT,
        numeric_type=LabelType.NUMERIC_01,
        format_categorical="'yes' | 'no'",
        format_numeric="a float in [0, 1] (0=no, 1=yes)"
    ),
}


@dataclass
class GroupConfig:
    """Configuration for a group of dimensions."""
    id: str
    name: str
    dimensions: List[str]
    description: str


# Group configurations for different modes
GROUPS_BY_CATEGORY: Dict[str, GroupConfig] = {
    "A": GroupConfig(
        id="A",
        name="China Attitudes",
        dimensions=["china_related", "china_ccp_government", "china_people_culture", "china_technology_development"],
        description="Determine if the video is China-related and evaluate its stance toward different aspects of China."
    ),
    "B": GroupConfig(
        id="B",
        name="Political Sensitivity",
        dimensions=["china_sensitive", "collective_action"],
        description="Identify politically sensitive content and collective action."
    ),
    "C": GroupConfig(
        id="C",
        name="Content Safety",
        dimensions=["hate_speech", "harmful_content"],
        description="Detect hate speech and harmful content."
    ),
    "D": GroupConfig(
        id="D",
        name="Content Type",
        dimensions=["news_segments", "inauthentic_content", "derivative_content"],
        description="Classify the type and authenticity of the content."
    ),
}

GROUPS_BINARY: Dict[str, GroupConfig] = {
    "china": GroupConfig(
        id="china",
        name="China-Related",
        dimensions=["china_related", "china_ccp_government", "china_people_culture", "china_technology_development", "china_sensitive", "collective_action"],
        description="Determine if the video is China-related, evaluate China-related attitudes and political sensitivity."
    ),
    "content": GroupConfig(
        id="content",
        name="Content Analysis",
        dimensions=["hate_speech", "harmful_content", "news_segments", "inauthentic_content", "derivative_content"],
        description="Analyze content safety and type."
    ),
}


def get_group_configs(mode: str) -> Dict[str, GroupConfig]:
    """Get group configurations for a given mode.

    Args:
        mode: One of 'single', 'by-category', 'binary', 'per-item'

    Returns:
        Dictionary mapping group IDs to GroupConfig objects
    """
    if mode == "single":
        # Single group containing all dimensions
        all_dims = list(DIMENSIONS.keys())
        return {
            "all": GroupConfig(
                id="all",
                name="All Dimensions",
                dimensions=all_dims,
                description="Comprehensive content analysis with all dimensions."
            )
        }
    elif mode == "by-category":
        return GROUPS_BY_CATEGORY
    elif mode == "binary":
        return GROUPS_BINARY
    elif mode == "per-item":
        # Each dimension as its own group
        return {
            dim_key: GroupConfig(
                id=dim_key,
                name=DIMENSIONS[dim_key].name,
                dimensions=[dim_key],
                description=f"Evaluate {DIMENSIONS[dim_key].name}."
            )
            for dim_key in DIMENSIONS
        }
    else:
        raise ValueError(f"Unknown group mode: {mode}. Expected 'single', 'by-category', 'binary', or 'per-item'")


def generate_group_prompt(group: GroupConfig, numeric_labels: bool = False) -> str:
    """Generate a focused system prompt for a specific group.

    Args:
        group: GroupConfig specifying which dimensions to include
        numeric_labels: If True, use numeric labels; if False, use categorical

    Returns:
        System prompt string tailored to the group's dimensions
    """
    dims = [DIMENSIONS[d] for d in group.dimensions]
    n_dims = len(dims)

    # Header
    header = (
        f"You are a meticulous content analysis assistant for TikTok videos. "
        f"Follow the CODEBOOK and output format exactly.\n\n"
        f"TASK: {group.description}\n\n"
        f"CODEBOOK — DEFINITIONS & TASKS\n"
    )

    # Build definitions
    definitions = []
    for i, dim in enumerate(dims, 1):
        if numeric_labels:
            definitions.append(f"{i}) {dim.key} - {dim.description}")
        else:
            definitions.append(f"{i}) {dim.key} - {dim.description}")

    definitions_text = "\n\n".join(definitions)

    # Build format section
    if numeric_labels:
        format_items = [f"{i}) {dim.key} — {dim.format_numeric}" for i, dim in enumerate(dims, 1)]
        format_instruction = f"SCORE THE VIDEO ON {n_dims} DIMENSION{'S' if n_dims > 1 else ''}:\n"
        value_rule = "• All values must be numbers in their specified ranges.\n• Use the midpoint of the scale for uncertain cases."
    else:
        format_items = [f"{i}) {dim.key} — {dim.format_categorical}" for i, dim in enumerate(dims, 1)]
        format_instruction = f"LABEL THE VIDEO ON {n_dims} DIMENSION{'S' if n_dims > 1 else ''}:\n"
        value_rule = "• Use exact values as specified above.\n• Use 'cannot_determine' when evidence is insufficient."

    format_text = format_instruction + "\n".join(format_items)

    # Build JSON keys list for format rules
    json_keys = ", ".join(group.dimensions)

    # Combine all sections
    prompt = (
        f"{header}"
        f"{definitions_text}\n\n"
        f"Do not default to the most frequent {'score' if numeric_labels else 'label'}; base labels on explicit evidence.\n\n"
        f"{format_text}\n\n"
        f"FORMAT RULES\n"
        f"• Output ONLY a minified JSON object with {'key' if n_dims == 1 else 'keys'}: {json_keys}.\n"
        f"{value_rule}\n"
        f"• Do not add extra keys or prose."
    )

    return prompt


def get_expected_keys(group: GroupConfig) -> List[str]:
    """Get the expected JSON keys for a group."""
    return group.dimensions.copy()


def get_all_expected_keys(mode: str) -> List[str]:
    """Get all expected JSON keys across all groups for a mode."""
    groups = get_group_configs(mode)
    all_keys = set()
    for group in groups.values():
        all_keys.update(group.dimensions)
    return sorted(list(all_keys))


def make_compound_id(meta_id: str, group_id: str) -> str:
    """Create a compound ID from meta_id and group_id.

    Format: {meta_id}__group_{group_id}
    """
    return f"{meta_id}__group_{group_id}"


def parse_compound_id(compound_id: str) -> Tuple[str, Optional[str]]:
    """Parse a compound ID back into meta_id and group_id.

    Returns:
        Tuple of (meta_id, group_id). group_id is None if not a compound ID.
    """
    if "__group_" in compound_id:
        parts = compound_id.rsplit("__group_", 1)
        return parts[0], parts[1]
    return compound_id, None


def validate_group_schema(obj: dict, group: GroupConfig, numeric_labels: bool = False) -> bool:
    """Validate that a JSON object matches the expected schema for a group.

    Args:
        obj: The parsed JSON object
        group: The GroupConfig to validate against
        numeric_labels: Whether numeric labels are expected

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(obj, dict):
        return False

    expected_keys = set(group.dimensions)

    # Check that all expected keys are present
    if not expected_keys.issubset(obj.keys()):
        return False

    # Validate each dimension
    for dim_key in group.dimensions:
        dim = DIMENSIONS[dim_key]
        value = obj.get(dim_key)

        if numeric_labels:
            # Numeric validation
            try:
                f = float(value)
                if dim.numeric_type == LabelType.NUMERIC_STANCE:
                    if not (-1.0 <= f <= 1.0):
                        return False
                elif dim.numeric_type == LabelType.NUMERIC_01:
                    if not (0.0 <= f <= 1.0):
                        return False
            except (TypeError, ValueError):
                return False
        else:
            # Categorical validation
            if dim.categorical_type == LabelType.STANCE:
                if value not in {"pro", "anti", "neutral", "neutral/unclear", "cannot_determine"}:
                    return False
            elif dim.categorical_type == LabelType.BINARY_STRICT:
                if value not in {"yes", "no"}:
                    # Also accept numeric values for backward compatibility
                    try:
                        f = float(value)
                        if not (0.0 <= f <= 1.0):
                            return False
                    except (TypeError, ValueError):
                        return False
            elif dim.categorical_type == LabelType.BINARY:
                if value not in {"yes", "no", "cannot_determine"}:
                    # Also accept numeric values for backward compatibility
                    try:
                        f = float(value)
                        if not (0.0 <= f <= 1.0):
                            return False
                    except (TypeError, ValueError):
                        return False

    return True


# Convenience function to get a single group's prompt
def get_group_system_prompt(group_id: str, mode: str = "by-category", numeric_labels: bool = False) -> str:
    """Get the system prompt for a specific group.

    Args:
        group_id: The group identifier
        mode: The grouping mode
        numeric_labels: Whether to use numeric labels

    Returns:
        System prompt string
    """
    groups = get_group_configs(mode)
    if group_id not in groups:
        raise ValueError(f"Unknown group '{group_id}' for mode '{mode}'. Available: {list(groups.keys())}")
    return generate_group_prompt(groups[group_id], numeric_labels)


if __name__ == "__main__":
    # Test/demo the module
    import argparse

    parser = argparse.ArgumentParser(description="Generate and display group prompts")
    parser.add_argument("--mode", default="by-category", choices=["single", "by-category", "binary", "per-item"])
    parser.add_argument("--numeric", action="store_true", help="Use numeric labels")
    parser.add_argument("--group", help="Show only specific group")
    args = parser.parse_args()

    groups = get_group_configs(args.mode)

    if args.group:
        if args.group not in groups:
            print(f"Unknown group: {args.group}. Available: {list(groups.keys())}")
        else:
            group = groups[args.group]
            print(f"=== Group {group.id}: {group.name} ===")
            print(f"Dimensions: {group.dimensions}")
            print()
            print(generate_group_prompt(group, args.numeric))
    else:
        for group_id, group in groups.items():
            print(f"\n{'='*60}")
            print(f"=== Group {group.id}: {group.name} ===")
            print(f"Dimensions: {group.dimensions}")
            print(f"{'='*60}\n")
            print(generate_group_prompt(group, args.numeric))
            print()
