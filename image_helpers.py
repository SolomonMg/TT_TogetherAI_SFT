#!/usr/bin/env python3
"""
image_helpers.py

Helper functions for working with video frame images in multimodal content.
"""

import base64
from pathlib import Path


def get_frame_paths(frames_dir: Path, video_id: str) -> list:
    """Return sorted list of all frame files for a video.

    Args:
        frames_dir: Directory containing frame folders
        video_id: The video/aweme ID

    Returns:
        List of Path objects for each frame, sorted by frame number
    """
    video_folder = frames_dir / str(video_id)
    if not video_folder.exists():
        return []
    # Match frame_0.jpg, frame_1.jpg, etc.
    frames = list(video_folder.glob("frame_*.jpg"))
    # Sort by frame number extracted from filename
    frames.sort(key=lambda p: int(p.stem.split("_")[1]))
    return frames


def encode_image_base64(image_path: Path) -> str | None:
    """Read image and return base64-encoded data URL.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 data URL string or None if file doesn't exist
    """
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
