#!/usr/bin/env python3
"""Visualize PVSG masks overlaid on video frames to verify alignment.

Picks random videos from each dataset (ego4d, vidor, epickitchen),
extracts frames, and overlays the corresponding masks with colored
transparency. Saves a grid of examples to verify masks match videos.

Usage:
    python scripts/visualize_pvsg_masks.py [--n-videos 3] [--n-frames 4] [--output viz_pvsg_masks.png]
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


PVSG_ROOT = "/home/mila/l/leh/scratch/latentlens/pvsg"
PVSG_JSON = os.path.join(PVSG_ROOT, "masks", "PVSG_Dataset.json")

DATASETS = {
    "ego4d": {
        "videos": os.path.join(PVSG_ROOT, "ego4d", "videos"),
        "masks": os.path.join(PVSG_ROOT, "ego4d", "masks"),
    },
    "vidor": {
        "videos": os.path.join(PVSG_ROOT, "vidor", "videos"),
        "masks": os.path.join(PVSG_ROOT, "vidor", "masks"),
    },
    # epickitchen videos not extracted yet, skip if missing
    "epickitchen": {
        "videos": os.path.join(PVSG_ROOT, "epickitchen", "videos"),
        "masks": os.path.join(PVSG_ROOT, "epickitchen", "masks"),
    },
}


def load_pvsg_metadata():
    """Load PVSG_Dataset.json and build video_id -> entry mapping."""
    with open(PVSG_JSON) as f:
        data = json.load(f)

    vid2entry = {entry["video_id"]: entry for entry in data["data"]}

    # Build video_id -> dataset source mapping from splits
    vid2src = {}
    src_name_map = {"vidor": "vidor", "epic_kitchen": "epickitchen", "ego4d": "ego4d"}
    for src, splits in data["split"].items():
        for split_name, vids in splits.items():
            for vid in vids:
                vid2src[vid] = src_name_map[src]

    return vid2entry, vid2src


def generate_colormap(n_objects):
    """Generate distinct colors for object IDs."""
    colors = {}
    for i in range(n_objects + 1):
        if i == 0:
            colors[i] = (0, 0, 0, 0)  # background = transparent
        else:
            hue = (i * 0.618033988749895) % 1.0  # golden ratio for spread
            rgb = hsv_to_rgb([hue, 0.8, 0.9])
            colors[i] = (*rgb, 0.5)
    return colors


def extract_frame(video_path, frame_idx, fps=5):
    """Extract a specific frame from a video (PVSG uses 5fps annotations)."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # PVSG masks are at 5fps; compute actual video frame index
    actual_frame = int(frame_idx * (video_fps / fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def overlay_mask(frame, mask, alpha=0.5):
    """Overlay a colored mask on a video frame."""
    unique_ids = np.unique(mask)
    colors = generate_colormap(int(unique_ids.max()) if len(unique_ids) > 0 else 0)

    overlay = frame.copy().astype(np.float32)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        color = colors.get(obj_id, colors.get(1))[:3]
        obj_mask = mask == obj_id
        for c in range(3):
            overlay[:, :, c] = np.where(
                obj_mask,
                frame[:, :, c] * (1 - alpha) + color[c] * 255 * alpha,
                overlay[:, :, c],
            )

    return overlay.astype(np.uint8)


def get_available_videos(dataset_name):
    """Get video IDs that have both video files and mask directories."""
    ds = DATASETS[dataset_name]
    if not os.path.isdir(ds["videos"]) or not os.path.isdir(ds["masks"]):
        return []

    video_ids = {Path(f).stem for f in os.listdir(ds["videos"]) if f.endswith(".mp4")}
    mask_ids = set(os.listdir(ds["masks"]))
    return sorted(video_ids & mask_ids)


def visualize_video(video_id, dataset_name, entry, n_frames=4):
    """Create overlay visualizations for a single video."""
    ds = DATASETS[dataset_name]
    video_path = os.path.join(ds["videos"], f"{video_id}.mp4")
    mask_dir = os.path.join(ds["masks"], video_id)

    meta = entry["meta"]
    total_frames = meta["num_frames"]
    fps = meta["fps"]

    # Pick evenly-spaced frames
    mask_files = sorted(os.listdir(mask_dir))
    n_available = len(mask_files)
    if n_available == 0:
        return None, None

    indices = np.linspace(0, n_available - 1, n_frames, dtype=int)

    frames = []
    overlays = []
    for idx in indices:
        frame_idx = int(mask_files[idx].replace(".png", ""))

        # Load frame from video
        frame = extract_frame(video_path, frame_idx, fps)
        if frame is None:
            continue

        # Load mask
        mask_path = os.path.join(mask_dir, mask_files[idx])
        mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))

        # Resize frame to mask dimensions if needed (or vice versa)
        if frame.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        overlay = overlay_mask(frame, mask)
        frames.append(frame)
        overlays.append(overlay)

    # Build object legend
    objects = entry.get("objects", [])
    obj_legend = {obj["object_id"]: obj["category"] for obj in objects}

    return frames, overlays, obj_legend, indices


def main():
    parser = argparse.ArgumentParser(description="Visualize PVSG masks on video frames")
    parser.add_argument("--n-videos", type=int, default=3,
                        help="Number of videos per dataset")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of frames per video")
    parser.add_argument("--output", type=str, default="viz_pvsg_masks.png",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    vid2entry, vid2src = load_pvsg_metadata()

    # Collect videos to visualize
    rows = []
    for ds_name in ["ego4d", "vidor", "epickitchen"]:
        available = get_available_videos(ds_name)
        if not available:
            print(f"  {ds_name}: no matched video+mask pairs (skipping)")
            continue

        selected = random.sample(available, min(args.n_videos, len(available)))
        print(f"  {ds_name}: {len(available)} matched, selected {len(selected)}")

        for vid_id in selected:
            if vid_id not in vid2entry:
                print(f"    {vid_id}: not in PVSG_Dataset.json (skipping)")
                continue
            entry = vid2entry[vid_id]
            result = visualize_video(vid_id, ds_name, entry, args.n_frames)
            if result is None or result[0] is None:
                continue
            frames, overlays, obj_legend, frame_indices = result

            # Verification stats
            mask_dir = os.path.join(DATASETS[ds_name]["masks"], vid_id)
            n_mask_frames = len(os.listdir(mask_dir))
            meta = entry["meta"]

            print(f"    {vid_id}: {meta['num_frames']} json frames, "
                  f"{n_mask_frames} mask files, "
                  f"{meta['width']}x{meta['height']} @ {meta['fps']}fps, "
                  f"{len(obj_legend)} objects")

            rows.append({
                "video_id": vid_id,
                "dataset": ds_name,
                "frames": frames,
                "overlays": overlays,
                "obj_legend": obj_legend,
                "frame_indices": frame_indices,
                "meta": meta,
            })

    if not rows:
        print("No videos to visualize!")
        return

    # Plot: 2 rows per video (raw frame, overlay), n_frames columns
    n_cols = args.n_frames
    n_rows = len(rows) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 2:
        axes = axes.reshape(2, -1)

    for i, row in enumerate(rows):
        for j in range(len(row["frames"])):
            ax_frame = axes[2 * i, j]
            ax_overlay = axes[2 * i + 1, j]

            ax_frame.imshow(row["frames"][j])
            ax_frame.set_xticks([])
            ax_frame.set_yticks([])
            if j == 0:
                ax_frame.set_ylabel(
                    f"{row['dataset']}\n{row['video_id']}\n(frame)",
                    fontsize=8, rotation=0, labelpad=80, va="center",
                )

            ax_overlay.imshow(row["overlays"][j])
            ax_overlay.set_xticks([])
            ax_overlay.set_yticks([])
            if j == 0:
                legend_str = ", ".join(
                    f"{k}:{v}" for k, v in sorted(row["obj_legend"].items())[:6]
                )
                ax_overlay.set_ylabel(
                    f"mask overlay\n{legend_str}",
                    fontsize=6, rotation=0, labelpad=80, va="center",
                )

        # Hide extra columns if fewer frames returned
        for j in range(len(row["frames"]), n_cols):
            axes[2 * i, j].axis("off")
            axes[2 * i + 1, j].axis("off")

    plt.suptitle("PVSG Mask-Video Alignment Verification", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")

    # Print summary statistics
    print("\n=== Alignment Summary ===")
    for ds_name in ["ego4d", "vidor", "epickitchen"]:
        available = get_available_videos(ds_name)
        if not available:
            continue
        mismatches = 0
        for vid_id in available:
            if vid_id not in vid2entry:
                continue
            entry = vid2entry[vid_id]
            mask_dir = os.path.join(DATASETS[ds_name]["masks"], vid_id)
            n_masks = len(os.listdir(mask_dir))
            n_json = entry["meta"]["num_frames"]
            if n_masks != n_json:
                mismatches += 1
                print(f"  MISMATCH {ds_name}/{vid_id}: "
                      f"{n_masks} masks vs {n_json} json frames")
        if mismatches == 0:
            print(f"  {ds_name}: all {len(available)} videos OK "
                  f"(mask count == json num_frames)")


if __name__ == "__main__":
    main()
