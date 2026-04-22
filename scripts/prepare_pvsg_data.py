#!/usr/bin/env python3
"""Prepare PVSG video subset for RQ3 re-evaluation.

Selects 50 videos from Ego4D and 50 from VidOR, symlinks them into a unified
videos directory, and extracts the middle frame from each as a JPEG.

Output directories:
    $SCRATCH/latentlens/pvsg_videos_100/  — symlinks to selected mp4s
    $SCRATCH/latentlens/pvsg_frames_100/  — middle-frame JPEGs

Usage:
    python scripts/prepare_pvsg_data.py [--seed 42] [--dry-run]
"""

import argparse
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"
EGO4D_DIR = PVSG_ROOT / "ego4d" / "videos"
VIDOR_DIR = PVSG_ROOT / "vidor" / "videos"


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def extract_middle_frame(video_path: Path, output_path: Path) -> bool:
    try:
        duration = get_video_duration(video_path)
        mid_time = duration / 2

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "frame.jpg"
            cmd = [
                "ffmpeg", "-ss", str(mid_time),
                "-i", str(video_path),
                "-vframes", "1", "-q:v", "2",
                "-y", str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0 or not frame_path.exists():
                return False

            img = Image.open(frame_path).convert("RGB")
            img.save(output_path, "JPEG", quality=90)
            return True
    except Exception as e:
        log.warning(f"  Frame extraction failed for {video_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-dataset", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scratch = SCRATCH / "latentlens"
    videos_out = scratch / "pvsg_videos_100"
    frames_out = scratch / "pvsg_frames_100"

    if not args.dry_run:
        videos_out.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)

    # Collect all videos from each dataset
    ego4d_vids = sorted(EGO4D_DIR.glob("*.mp4"))
    vidor_vids = sorted(VIDOR_DIR.glob("*.mp4"))
    log.info(f"Found {len(ego4d_vids)} Ego4D videos, {len(vidor_vids)} VidOR videos")

    # Select N from each with fixed seed
    rng = random.Random(args.seed)
    ego4d_sel = sorted(rng.sample(ego4d_vids, min(args.n_per_dataset, len(ego4d_vids))))
    rng2 = random.Random(args.seed)
    vidor_sel = sorted(rng2.sample(vidor_vids, min(args.n_per_dataset, len(vidor_vids))))

    selected = []
    for src in ego4d_sel:
        selected.append(("ego4d", src))
    for src in vidor_sel:
        selected.append(("vidor", src))

    log.info(f"Selected {len(ego4d_sel)} Ego4D + {len(vidor_sel)} VidOR = {len(selected)} total")

    n_ok = 0
    for idx, (dataset, src_path) in enumerate(selected):
        name = f"pvsg_{idx:04d}"
        link_path = videos_out / f"{name}.mp4"
        frame_path = frames_out / f"{name}.jpg"

        log.info(f"[{idx+1:03d}/{len(selected)}] {dataset}/{src_path.name}")

        if args.dry_run:
            log.info(f"  DRY RUN: would symlink → {link_path}, extract → {frame_path}")
            continue

        # Symlink video
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(src_path)

        # Extract middle frame
        if frame_path.exists():
            log.info(f"  Frame already exists, skipping extraction")
            n_ok += 1
            continue

        ok = extract_middle_frame(src_path, frame_path)
        if ok:
            log.info(f"  Saved {frame_path.name}")
            n_ok += 1
        else:
            log.warning(f"  FAILED frame extraction, removing symlink")
            link_path.unlink()

    if not args.dry_run:
        log.info(f"\nDone: {n_ok}/{len(selected)} videos prepared")
        log.info(f"Videos: {videos_out}")
        log.info(f"Frames: {frames_out}")
        actual_vids = len(list(videos_out.glob("*.mp4")))
        actual_frames = len(list(frames_out.glob("*.jpg")))
        log.info(f"  {actual_vids} videos, {actual_frames} frames")


if __name__ == "__main__":
    main()
