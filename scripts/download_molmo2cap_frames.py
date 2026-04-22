#!/usr/bin/env python3
"""Download videos from Molmo2-Cap and extract middle frames.

Downloads YouTube videos by ID and saves the middle frame as a JPEG.

Usage:
    python scripts/download_molmo2cap_frames.py \
        --output-dir data/molmo2cap_frames_100/ \
        --num-videos 100
"""

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_and_extract_middle_frame(video_id: str, output_path: Path) -> bool:
    """Download a YouTube video and save its middle frame."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "video.mp4"

        # Download video (low quality, just need frames)
        cmd = [
            "yt-dlp", "-f", "worst[ext=mp4]",
            "--no-playlist", "--quiet",
            "-o", str(video_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not video_path.exists():
            return False

        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        mid_time = duration / 2

        # Extract middle frame
        frame_path = Path(tmpdir) / "frame.jpg"
        extract_cmd = [
            "ffmpeg", "-ss", str(mid_time),
            "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            "-y", str(frame_path),
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or not frame_path.exists():
            return False

        # Verify it's a valid image and copy
        img = Image.open(frame_path)
        img.save(output_path, "JPEG", quality=90)
        return True


def main():
    parser = argparse.ArgumentParser()
    scratch = os.environ.get("SCRATCH", "data")
    parser.add_argument("--output-dir", type=Path, default=Path(f"{scratch}/latentlens/molmo2cap_frames_100"))
    parser.add_argument("--num-videos", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading Molmo2-Cap val split (streaming)...")
    ds = load_dataset("allenai/Molmo2-Cap", split="val", streaming=True)

    count = 0
    skipped = 0
    for sample in ds:
        if count >= args.num_videos:
            break

        video_id = sample["video_id"]
        output_path = args.output_dir / f"frame_{count:04d}.jpg"

        if output_path.exists():
            log.info(f"  [{count+1}] Already exists: {output_path.name}")
            count += 1
            continue

        log.info(f"  [{count+1}/{args.num_videos}] Downloading {video_id}...")
        try:
            ok = download_and_extract_middle_frame(video_id, output_path)
        except Exception as e:
            log.warning(f"    Failed: {e}")
            ok = False

        if ok:
            count += 1
            log.info(f"    Saved {output_path.name}")
        else:
            skipped += 1
            log.warning(f"    Skipped {video_id} (download/extract failed)")

    log.info(f"Done: {count} frames saved, {skipped} skipped")


if __name__ == "__main__":
    main()
