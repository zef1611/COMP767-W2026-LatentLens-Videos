#!/usr/bin/env python3
"""Download videos from Molmo2-Cap dataset.

Downloads YouTube videos by ID and saves them as MP4 files.

Usage:
    python scripts/download_molmo2cap_videos.py \
        --output-dir $SCRATCH/latentlens/molmo2cap_videos_500 \
        --num-videos 500
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_video(video_id: str, output_path: Path) -> bool:
    """Download a YouTube video as MP4."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp", "-f", "worst[ext=mp4]",
        "--no-playlist", "--quiet",
        "-o", str(output_path),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    scratch = os.environ.get("SCRATCH", "data")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"{scratch}/latentlens/molmo2cap_videos_500"),
    )
    parser.add_argument("--num-videos", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(args.output_dir.glob("*.mp4")))
    log.info(f"Output: {args.output_dir}")
    log.info(f"Already have {existing} videos, targeting {args.num_videos}")

    if existing >= args.num_videos:
        log.info("Already have enough videos, nothing to do.")
        return

    log.info("Loading Molmo2-Cap val split (streaming)...")
    ds = load_dataset("allenai/Molmo2-Cap", split="val", streaming=True)

    count = existing
    skipped = 0
    for sample in ds:
        if count >= args.num_videos:
            break

        video_id = sample["video_id"]
        output_path = args.output_dir / f"video_{count:04d}.mp4"

        if output_path.exists():
            log.info(f"  [{count+1}] Already exists: {output_path.name}")
            count += 1
            continue

        log.info(f"  [{count+1}/{args.num_videos}] Downloading {video_id}...")
        ok = download_video(video_id, output_path)

        if ok:
            count += 1
            log.info(f"    Saved {output_path.name}")
        else:
            skipped += 1
            log.warning(f"    Skipped {video_id} (download failed)")

    log.info(f"Done: {count} videos saved, {skipped} skipped")


if __name__ == "__main__":
    main()
