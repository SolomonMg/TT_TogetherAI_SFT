#!/usr/bin/env python3
"""
merge_groups.py — Merge per-group predictions back into unified records.

Purpose
-------
After running inference with group-mode prompts, predictions are split across
multiple records (one per group per video). This script merges them back into
unified records with all dimensions.

Usage
-----
# Merge parsed predictions from per-group inference
python merge_groups.py \
  --input out/preds_pergroup.parsed.jsonl \
  --output out/preds_merged.jsonl

# Merge and also join with original CSV/Parquet data
python merge_groups.py \
  --input out/preds_pergroup.parsed.jsonl \
  --output out/preds_merged.csv \
  --original-data data/nyu_rand_china.parquet

# Specify group mode for validation
python merge_groups.py \
  --input out/preds_pergroup.parsed.jsonl \
  --output out/preds_merged.jsonl \
  --group-mode by-category
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Any

import pandas as pd
from json_utils import load_jsonl, write_jsonl, load_table, norm_cols
from prompt_groups import parse_compound_id, get_group_configs, get_all_expected_keys


def merge_group_predictions(
    input_path: str,
    output_path: str,
    group_mode: str = "by-category",
    require_complete: bool = False
) -> Dict[str, Any]:
    """Merge per-group predictions into unified records.

    Args:
        input_path: Path to parsed JSONL with per-group predictions
        output_path: Path to write merged JSONL
        group_mode: Group mode used during inference
        require_complete: If True, only output records with all groups present

    Returns:
        Dictionary with merge statistics
    """
    print(f"[info] Loading parsed predictions: {input_path}")
    records = load_jsonl(input_path)

    # Get expected groups and keys
    groups = get_group_configs(group_mode)
    expected_group_ids = set(groups.keys())
    all_expected_keys = set(get_all_expected_keys(group_mode))

    print(f"[info] Group mode: {group_mode}")
    print(f"[info] Expected groups: {sorted(expected_group_ids)}")
    print(f"[info] Expected keys: {sorted(all_expected_keys)}")

    # Group records by original_meta_id
    by_meta_id: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for rec in records:
        if not rec.get("parsed") or not rec.get("json"):
            continue

        # Get original meta_id (strip group suffix)
        meta_id = rec.get("meta_id", "")
        original_meta_id = rec.get("original_meta_id")
        group_id = rec.get("group_id")

        if original_meta_id is None:
            original_meta_id, group_id = parse_compound_id(meta_id)

        if not original_meta_id:
            continue

        # Store this group's predictions
        by_meta_id[original_meta_id][group_id] = rec

    print(f"[info] Found {len(by_meta_id)} unique meta_ids")

    # Merge predictions for each meta_id
    merged_records = []
    stats = {
        "total_meta_ids": len(by_meta_id),
        "complete": 0,
        "partial": 0,
        "groups_found": defaultdict(int)
    }

    for meta_id, group_recs in by_meta_id.items():
        # Merge all JSON predictions
        merged_json = {}
        groups_present = set()

        for group_id, rec in group_recs.items():
            if rec.get("json"):
                merged_json.update(rec["json"])
                if group_id:
                    groups_present.add(group_id)
                    stats["groups_found"][group_id] += 1

        # Check completeness
        is_complete = expected_group_ids.issubset(groups_present)
        if is_complete:
            stats["complete"] += 1
        else:
            stats["partial"] += 1
            missing = expected_group_ids - groups_present
            if require_complete:
                print(f"[warning] Skipping incomplete record {meta_id} (missing groups: {missing})")
                continue

        # Build merged record
        merged_rec = {
            "meta_id": meta_id,
            "parsed": True,
            "complete": is_complete,
            "groups_present": sorted(list(groups_present)),
            "json": merged_json
        }

        # Include raw outputs from all groups (optional, for debugging)
        # merged_rec["raw_by_group"] = {g: r.get("raw", "") for g, r in group_recs.items()}

        merged_records.append(merged_rec)

    # Write output
    write_jsonl(output_path, merged_records)

    # Print statistics
    print(f"\n[stats] Merge complete:")
    print(f"  Total unique meta_ids: {stats['total_meta_ids']}")
    print(f"  Complete (all groups): {stats['complete']}")
    print(f"  Partial (some groups): {stats['partial']}")
    print(f"  Output records: {len(merged_records)}")
    print(f"  Groups found:")
    for g in sorted(stats["groups_found"].keys()):
        print(f"    {g}: {stats['groups_found'][g]}")

    return stats


def merge_with_original_data(
    merged_jsonl_path: str,
    original_data_path: str,
    output_path: str
) -> None:
    """Merge predictions with original CSV/Parquet data.

    Args:
        merged_jsonl_path: Path to merged JSONL predictions
        original_data_path: Path to original CSV/Parquet file
        output_path: Path to write merged output (CSV or Parquet based on extension)
    """
    print(f"[info] Loading merged predictions: {merged_jsonl_path}")
    preds = load_jsonl(merged_jsonl_path)

    print(f"[info] Loading original data: {original_data_path}")
    df = norm_cols(load_table(original_data_path))

    # Find meta_id column
    meta_col = None
    for col in ["meta_id", "id", "tt_video_id", "yt_video_id", "video_id"]:
        if col in df.columns:
            meta_col = col
            break

    if not meta_col:
        raise RuntimeError("[error] Original data missing meta_id column")

    # Convert meta_id to string for matching
    df[meta_col] = df[meta_col].astype(str)

    # Build prediction mapping
    pred_map = {}
    for rec in preds:
        meta_id = str(rec.get("meta_id", ""))
        if meta_id and rec.get("parsed") and rec.get("json"):
            pred_map[meta_id] = rec

    print(f"[info] Found predictions for {len(pred_map)} meta_ids")

    # Get all prediction keys from the data
    all_pred_keys = set()
    for rec in pred_map.values():
        if rec.get("json"):
            all_pred_keys.update(rec["json"].keys())

    # Add prediction columns
    for key in sorted(all_pred_keys):
        df[f"pred_{key}"] = None
    df["pred_parsed"] = False
    df["pred_complete"] = False

    # Merge predictions
    matched = 0
    for idx, row in df.iterrows():
        meta_id = str(row[meta_col])
        if meta_id in pred_map:
            rec = pred_map[meta_id]
            pred_json = rec.get("json", {})
            for key in all_pred_keys:
                df.at[idx, f"pred_{key}"] = pred_json.get(key)
            df.at[idx, "pred_parsed"] = True
            df.at[idx, "pred_complete"] = rec.get("complete", False)
            matched += 1

    print(f"[info] Matched {matched}/{len(df)} rows ({matched/len(df)*100:.1f}%)")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, quoting=1, escapechar='\\')

    print(f"[write] {output_path}  n={len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-group predictions into unified records"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input parsed JSONL with per-group predictions"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file (JSONL, or CSV/Parquet if --original-data provided)"
    )
    parser.add_argument(
        "--group-mode", default="by-category",
        choices=["by-category", "binary", "per-item"],
        help="Group mode used during inference (default: by-category)"
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Only output records with all groups present"
    )
    parser.add_argument(
        "--original-data",
        help="Optional: Original CSV/Parquet to merge predictions with"
    )

    args = parser.parse_args()

    # Determine output format
    ext = os.path.splitext(args.output)[1].lower()

    if args.original_data:
        # Two-step: merge groups, then merge with original data
        if ext in [".csv", ".parquet"]:
            # Write intermediate merged JSONL, then merge with original
            intermediate_path = args.output.rsplit(".", 1)[0] + "_merged.jsonl"
            merge_group_predictions(
                args.input,
                intermediate_path,
                args.group_mode,
                args.require_complete
            )
            merge_with_original_data(
                intermediate_path,
                args.original_data,
                args.output
            )
        else:
            raise ValueError(
                "When --original-data is provided, output must be .csv or .parquet"
            )
    else:
        # Just merge groups
        if ext != ".jsonl":
            print(f"[warning] Output extension '{ext}' unusual for JSONL, continuing anyway")
        merge_group_predictions(
            args.input,
            args.output,
            args.group_mode,
            args.require_complete
        )


if __name__ == "__main__":
    main()
