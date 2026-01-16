# Vision Support for TikTok Video Analysis

This document describes the vision (image) support added to the Together.AI SFT pipeline, enabling multimodal analysis of TikTok videos using both visual frames and text content.

## Overview

The pipeline now supports including video frames in the analysis, allowing vision-capable models (like Llama 4 Scout/Maverick, Qwen2.5-VL) to analyze both visual and textual content from TikTok videos.

## Requirements

- Video frames extracted and organized in folders by video ID
- Frame naming convention: `frame_0.jpg`, `frame_1.jpg`, etc.
- Vision-capable model (see [Together.AI Vision Models](https://docs.together.ai/docs/vision-overview))

### Supported Vision Models

| Model | Context Length | Notes |
|-------|----------------|-------|
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 512K | Recommended for cost efficiency |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | 1M | Higher capacity |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 32K | Alternative vision model |

## Directory Structure

Frames should be organized as follows:

```
frames/
├── 7531309745475554591/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   └── frame_2.jpg
├── 7532078460303330573/
│   └── frame_0.jpg
└── 7550688756894846238/
    ├── frame_0.jpg
    └── frame_1.jpg
```

## Usage

### Building Training/Inference Data with Images

```bash
# Basic usage with images
python build_finetune_jsonl.py \
    --input data/full_training_data.csv \
    --output data/train_with_images.jsonl \
    --include-images \
    --frames-dir ./frames

# Comprehensive categorical mode with images
python build_finetune_jsonl.py \
    --input data/full_training_data.csv \
    --output data/train_comprehensive.jsonl \
    --include-images \
    --frames-dir ./frames \
    --comprehensive

# Inference-only (no gold labels) with images
python build_finetune_jsonl.py \
    --input data/unlabeled_videos.csv \
    --output data/inference_with_images.jsonl \
    --include-images \
    --frames-dir ./frames \
    --no-labels
```

### Running Inference with Images

```bash
# Basic inference with images
python infer.py \
    --val-file data/inference_with_images.jsonl \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --out out/vision_preds.raw.jsonl \
    --include-images \
    --frames-dir ./frames \
    --concurrency 4 \
    --temperature 0 \
    --max-tokens 512

# With truncation retry (recommended for comprehensive mode)
python infer.py \
    --val-file data/inference_with_images.jsonl \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --out out/vision_preds.raw.jsonl \
    --include-images \
    --frames-dir ./frames \
    --concurrency 4 \
    --temperature 0 \
    --max-tokens 512 \
    --retry-on-trunc \
    --max-tokens-cap 1024
```

## New CLI Arguments

### `build_finetune_jsonl.py`

| Argument | Type | Description |
|----------|------|-------------|
| `--include-images` | flag | Enable image inclusion in training data |
| `--frames-dir` | path | Directory containing frame folders (required if `--include-images` is set) |

### `infer.py`

| Argument | Type | Description |
|----------|------|-------------|
| `--include-images` | flag | Enable image inclusion in inference |
| `--frames-dir` | path | Directory containing frame folders (required if `--include-images` is set) |

## Message Format

When images are enabled, the user message content changes from a plain string to a multimodal content array:

### Text-Only (Default)
```json
{
  "role": "user",
  "content": "TRANSCRIPT:\n...\n\nDESCRIPTION:\n..."
}
```

### With Images (Multimodal)
```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "TRANSCRIPT:\n...\n\nDESCRIPTION:\n..."}
  ]
}
```

Images are included in chronological order (frame_0, frame_1, ...), followed by the text content.

## System Prompt

When images are enabled, the system prompt is prefixed with vision context:

> "You are analyzing TikTok videos using both visual and textual content. You will be given one or more representative frames from the video (in chronological order), along with the transcript and description. Analyze both the visual content across all frames and the text to determine the appropriate labels."

## Token Cost Estimation

Image tokens are calculated based on the Together.AI formula:

```
tokens = min(2, max(H // 560, 1)) * min(2, max(W // 560, 1)) * 1601
```

| Resolution | Tokens per Image |
|------------|------------------|
| SD (480×720) | 1,601 |
| HD (720×1280) | 3,202 |

**Cost Example:**
- 1M videos × 4 frames/video × 3,202 tokens/frame × $0.18/M tokens = ~$2,305

## Error Handling

- **Missing frames:** Warning logged, falls back to text-only for that example
- **Invalid image encoding:** Skipped, continues with other frames
- **No frames found:** Falls back to text-only content
- **Missing `--frames-dir`:** Error raised if `--include-images` is set

## Backwards Compatibility

All changes are **opt-in** via the `--include-images` flag. Without this flag:
- `build_finetune_jsonl.py` behavior is identical to previous version
- `infer.py` behavior is identical to previous version
- Existing scripts and workflows continue to work unchanged

## Complete Workflow Example

```bash
# 1. Build training data with images
python build_finetune_jsonl.py \
    --input data/labeled_videos.csv \
    --output data/train.jsonl \
    --include-images \
    --frames-dir ./frames \
    --comprehensive

# 2. Build inference data with images (no labels)
python build_finetune_jsonl.py \
    --input data/unlabeled_videos.csv \
    --output data/inference.jsonl \
    --include-images \
    --frames-dir ./frames \
    --comprehensive \
    --no-labels

# 3. Run inference
python infer.py \
    --val-file data/inference.jsonl \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --out out/preds.raw.jsonl \
    --include-images \
    --frames-dir ./frames \
    --concurrency 4 \
    --temperature 0 \
    --max-tokens 800 \
    --retry-on-trunc

# 4. Parse predictions
python parse.py \
    --raw out/preds.raw.jsonl \
    --out out/preds.parsed.jsonl \
    --print-bad 5

# 5. Score (if validation data has labels)
python score.py \
    --val-file data/train.jsonl \
    --preds out/preds.parsed.jsonl \
    --dump-csv out/results.csv
```

## Troubleshooting

### "No frames found for video X"
- Verify the frames directory structure matches expected format
- Check that video ID in CSV matches folder name exactly
- Ensure frames are named `frame_*.jpg`

### "Model does not support images"
- Use a vision-capable model (see supported models above)
- Check Together.AI documentation for current vision model availability

### Large JSONL file size
- Base64 encoding increases file size significantly
- Consider using remote image URLs if supported by your workflow
- For very large datasets, process in batches

---

**Last Updated:** January 2026
