#!/usr/bin/env python3
"""Generate side-by-side video with PVSG mask overlays.

Renders a video with the original frame on the left and the mask overlay
(colored regions + contours + object labels) on the right.

Usage:
    # Single video
    python scripts/generate_pvsg_mask_video.py --video-id P01_10 --dataset epickitchen

    # Multiple videos
    python scripts/generate_pvsg_mask_video.py --video-id P01_10 P26_02 --dataset epickitchen

    # All videos in a dataset
    python scripts/generate_pvsg_mask_video.py --dataset ego4d --all

    # Custom output dir and max width
    python scripts/generate_pvsg_mask_video.py --video-id 0001_4164158586 --dataset vidor \
        --output-dir pvsg_visualization --max-width 1280
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb


PVSG_ROOT = "/home/mila/l/leh/scratch/latentlens/pvsg"
PVSG_JSON = os.path.join(PVSG_ROOT, "PVSG_Dataset.json")

DATASET_DIRS = {
    "ego4d": {"videos": "ego4d/videos", "masks": "ego4d/masks"},
    "vidor": {"videos": "vidor/videos", "masks": "vidor/masks"},
    "epickitchen": {"videos": "epickitchen/videos", "masks": "epickitchen/masks"},
}


def get_color(obj_id):
    """Generate a distinct color for an object ID using golden ratio spacing."""
    hue = (obj_id * 0.618033988749895) % 1.0
    rgb = hsv_to_rgb([hue, 0.8, 0.9])
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def find_video_file(video_dir, video_id):
    """Find video file with case-insensitive extension matching."""
    for ext in [".mp4", ".MP4", ".avi", ".mkv"]:
        path = os.path.join(video_dir, video_id + ext)
        if os.path.exists(path):
            return path
    return None


def get_available_videos(dataset):
    """Get video IDs that have both video files and mask directories."""
    dirs = DATASET_DIRS[dataset]
    video_dir = os.path.join(PVSG_ROOT, dirs["videos"])
    mask_dir = os.path.join(PVSG_ROOT, dirs["masks"])

    if not os.path.isdir(video_dir) or not os.path.isdir(mask_dir):
        return []

    video_ids = {Path(f).stem for f in os.listdir(video_dir)
                 if f.lower().endswith((".mp4", ".avi", ".mkv"))}
    mask_ids = set(os.listdir(mask_dir))
    return sorted(video_ids & mask_ids)


def render_video(video_id, dataset, entry, output_dir, max_width=1280,
                 max_duration=None):
    """Render a side-by-side mask overlay video."""
    dirs = DATASET_DIRS[dataset]
    video_dir = os.path.join(PVSG_ROOT, dirs["videos"])
    mask_dir = os.path.join(PVSG_ROOT, dirs["masks"], video_id)

    video_path = find_video_file(video_dir, video_id)
    if video_path is None:
        print(f"  ERROR: video file not found for {video_id}")
        return None

    mask_files = sorted(os.listdir(mask_dir))
    if not mask_files:
        print(f"  ERROR: no mask files for {video_id}")
        return None

    meta = entry["meta"]

    # Limit to max_duration seconds
    if max_duration is not None:
        max_frames = int(max_duration * meta["fps"])
        mask_files = mask_files[:max_frames]
    obj_map = {o["object_id"]: o["category"] for o in entry["objects"]}
    ann_fps = meta["fps"]

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Get frame dimensions
    ret, sample = cap.read()
    if not ret:
        print(f"  ERROR: cannot read video {video_id}")
        cap.release()
        return None
    h, w = sample.shape[:2]

    # Scale to fit max_width (for the side-by-side output)
    scale = min(1.0, max_width / w)
    out_w = int(w * scale)
    out_h = int(h * scale)

    # Write intermediate mp4v, then re-encode to h264
    tmp_path = os.path.join(output_dir, f".{video_id}_tmp.mp4")
    out_path = os.path.join(output_dir, f"{video_id}_masks.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, ann_fps, (out_w * 2, out_h))

    for i, mf in enumerate(mask_files):
        frame_idx = int(mf.replace(".png", ""))
        actual_frame = int(frame_idx * (video_fps / ann_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        mask = cv2.imread(os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE)
        if frame.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Build overlay
        overlay = frame.copy().astype(np.float32)
        unique_ids = [oid for oid in np.unique(mask) if oid != 0]
        for oid in unique_ids:
            color = get_color(oid)
            m = mask == oid
            for c in range(3):
                overlay[:, :, c] = np.where(
                    m, frame[:, :, c] * 0.4 + color[c] * 0.6, overlay[:, :, c]
                )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Draw contours
        for oid in unique_ids:
            color = get_color(oid)
            m = (mask == oid).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        # Add object labels at centroids
        for oid in unique_ids:
            m = mask == oid
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label = obj_map.get(oid, str(oid))
            color = get_color(oid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 4),
                          (0, 0, 0), -1)
            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        # Timestamp
        t = frame_idx / ann_fps
        for img in [frame, overlay]:
            cv2.putText(img, f"t={t:.1f}s  f{frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # Resize and concatenate
        frame_s = cv2.resize(frame, (out_w, out_h))
        overlay_s = cv2.resize(overlay, (out_w, out_h))
        writer.write(np.hstack([frame_s, overlay_s]))

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(mask_files)} frames")

    writer.release()
    cap.release()

    # Re-encode to h264 for broad compatibility
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", out_path],
        capture_output=True, text=True,
    )
    os.remove(tmp_path)

    if result.returncode != 0:
        print(f"  WARNING: ffmpeg re-encode failed, keeping mp4v version")
        os.rename(tmp_path, out_path) if os.path.exists(tmp_path) else None

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side PVSG mask overlay videos"
    )
    parser.add_argument("--video-id", nargs="+", help="Video ID(s) to render")
    parser.add_argument("--dataset", required=True,
                        choices=["ego4d", "vidor", "epickitchen"],
                        help="Dataset name")
    parser.add_argument("--all", action="store_true",
                        help="Render all videos with masks in the dataset")
    parser.add_argument("--output-dir", default="pvsg_visualization",
                        help="Output directory")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="Max width per side (total output = 2x this)")
    parser.add_argument("--max-duration", type=float, default=10,
                        help="Max video duration in seconds (default: 10, 0 for full)")
    args = parser.parse_args()

    if args.max_duration == 0:
        args.max_duration = None

    if not args.video_id and not args.all:
        parser.error("Provide --video-id or --all")

    # Load metadata
    with open(PVSG_JSON) as f:
        data = json.load(f)
    vid2entry = {e["video_id"]: e for e in data["data"]}

    # Determine which videos to render
    if args.all:
        video_ids = get_available_videos(args.dataset)
        print(f"Rendering all {len(video_ids)} videos from {args.dataset}")
    else:
        video_ids = args.video_id

    os.makedirs(args.output_dir, exist_ok=True)

    for vid_id in video_ids:
        if vid_id not in vid2entry:
            print(f"SKIP {vid_id}: not in PVSG_Dataset.json")
            continue

        entry = vid2entry[vid_id]
        meta = entry["meta"]
        n_obj = len(entry["objects"])
        print(f"Rendering {vid_id} ({args.dataset}) — "
              f"{meta['width']}x{meta['height']} @ {meta['fps']}fps, "
              f"{meta['num_frames']} frames, {n_obj} objects")

        out = render_video(vid_id, args.dataset, entry, args.output_dir,
                           args.max_width, args.max_duration)
        if out:
            print(f"  -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
