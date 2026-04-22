#!/usr/bin/env python3
"""Prepare PVSG video subset pre-filtered for trackable objects.

Scans all PVSG videos, checks which objects are large enough to dominate
at least one patch on a 14x14 grid (>50% coverage), and selects 100 videos
where at least one object is trackable across sampled frames.

Creates:
    $SCRATCH/latentlens/pvsg_videos_100/  -- symlinks to qualified mp4s
    $SCRATCH/latentlens/pvsg_frames_100/  -- middle-frame JPEGs
    results/rq5_qualified_videos.json     -- manifest of selected videos

Usage:
    python scripts/rq5/prepare_pvsg_qualified.py [--seed 42] [--dry-run]
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"
PVSG_JSON = PVSG_ROOT / "PVSG_Dataset.json"

DATASETS = {
    "ego4d": {
        "videos": PVSG_ROOT / "ego4d" / "videos",
        "masks": PVSG_ROOT / "ego4d" / "masks",
    },
    "vidor": {
        "videos": PVSG_ROOT / "vidor" / "videos",
        "masks": PVSG_ROOT / "vidor" / "masks",
    },
    "epickitchen": {
        "videos": PVSG_ROOT / "epickitchen" / "videos",
        "masks": PVSG_ROOT / "epickitchen" / "masks",
    },
}

# Conservative grid: Molmo2 uses 14x14 global crop (coarsest grid)
GRID_H, GRID_W = 14, 14
MIN_COVERAGE = 0.5
N_SAMPLE_FRAMES = 5  # sample 5 uniformly-spaced mask frames to estimate object sizes


def load_pvsg_metadata():
    """Load PVSG_Dataset.json and build mappings."""
    with open(PVSG_JSON) as f:
        data = json.load(f)

    vid2entry = {entry["video_id"]: entry for entry in data["data"]}

    vid2src = {}
    src_name_map = {"vidor": "vidor", "epic_kitchen": "epickitchen", "ego4d": "ego4d"}
    for src, splits in data["split"].items():
        for split_name, vids in splits.items():
            for vid in vids:
                vid2src[vid] = src_name_map[src]

    return vid2entry, vid2src


def get_available_videos(dataset_name):
    """Get video IDs that have both video files and mask directories."""
    ds = DATASETS[dataset_name]
    if not ds["videos"].is_dir() or not ds["masks"].is_dir():
        return []
    video_ids = {p.stem for p in ds["videos"].glob("*.mp4")}
    mask_ids = set(os.listdir(ds["masks"]))
    return sorted(video_ids & mask_ids)


def check_video_objects(video_id, dataset_name, entry, min_coverage=MIN_COVERAGE):
    """Check which objects in a video are large enough to track.

    Returns list of qualified objects with their areas, or empty list.
    """
    ds = DATASETS[dataset_name]
    mask_dir = ds["masks"] / video_id

    mask_files = sorted(os.listdir(mask_dir))
    if len(mask_files) == 0:
        return []

    # Sample N_SAMPLE_FRAMES uniformly spaced mask frames
    n_available = len(mask_files)
    if n_available <= N_SAMPLE_FRAMES:
        sample_indices = list(range(n_available))
    else:
        sample_indices = np.linspace(0, n_available - 1, N_SAMPLE_FRAMES, dtype=int).tolist()

    # Track which objects appear large enough in each sampled frame
    objects_in_entry = {obj["object_id"]: obj for obj in entry.get("objects", [])}
    # For each object: track how many sampled frames it's large enough in
    object_ok_frames = defaultdict(int)
    object_areas = defaultdict(list)

    for idx in sample_indices:
        mask_path = mask_dir / mask_files[idx]
        # CRITICAL: masks are palette (P mode) PNGs — must use PIL, not cv2
        # cv2.IMREAD_GRAYSCALE reads raw palette indices, giving wrong IDs
        try:
            mask = np.array(Image.open(str(mask_path)))
        except Exception:
            continue

        mask_h, mask_w = mask.shape
        # Patch size in mask pixels
        patch_h = mask_h / GRID_H
        patch_w = mask_w / GRID_W
        patch_area = patch_h * patch_w
        min_area = min_coverage * patch_area

        # For each patch, find the dominant object
        # But we actually need to check: does this object cover >50% of ANY patch?
        # More efficient: for each object, compute its area within each patch
        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            if obj_id == 0:  # background
                continue
            obj_mask = (mask == obj_id)

            # Check each patch for this object
            found_in_patch = False
            for r in range(GRID_H):
                if found_in_patch:
                    break
                for c in range(GRID_W):
                    y0 = int(r * patch_h)
                    y1 = int((r + 1) * patch_h)
                    x0 = int(c * patch_w)
                    x1 = int((c + 1) * patch_w)
                    patch_region = obj_mask[y0:y1, x0:x1]
                    coverage = patch_region.sum() / max(patch_region.size, 1)
                    if coverage >= min_coverage:
                        found_in_patch = True
                        break

            if found_in_patch:
                object_ok_frames[obj_id] += 1
                # Compute total object area for reporting
                total_pixels = obj_mask.sum()
                object_areas[obj_id].append(int(total_pixels))

    # An object is qualified if it's large enough in ALL sampled frames
    n_sampled = len(sample_indices)
    qualified = []
    for obj_id, n_ok in object_ok_frames.items():
        if n_ok == n_sampled:
            obj_info = objects_in_entry.get(obj_id, {})
            qualified.append({
                "object_id": int(obj_id),
                "category": obj_info.get("category", "unknown"),
                "is_thing": obj_info.get("is_thing", True),
                "mean_area_px": int(np.mean(object_areas[obj_id])),
                "n_frames_checked": n_sampled,
            })

    return qualified


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
    parser = argparse.ArgumentParser(
        description="Prepare PVSG videos pre-filtered for trackable objects"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-videos", type=int, default=100,
                        help="Target number of qualified videos")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                        help="Min fraction of patch area an object must cover")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats without creating files")
    args = parser.parse_args()

    vid2entry, vid2src = load_pvsg_metadata()

    # Scan all videos across datasets for qualified objects
    log.info("Scanning all PVSG videos for trackable objects...")
    candidates = []  # (dataset, video_id, qualified_objects)

    for ds_name in ["ego4d", "vidor"]:
        available = get_available_videos(ds_name)
        log.info(f"  {ds_name}: {len(available)} videos with masks")

        for vid_id in available:
            if vid_id not in vid2entry:
                continue
            entry = vid2entry[vid_id]
            qualified_objs = check_video_objects(
                vid_id, ds_name, entry, min_coverage=args.min_coverage
            )
            if qualified_objs:
                candidates.append((ds_name, vid_id, qualified_objs))

    log.info(f"\nTotal qualified videos: {len(candidates)}")
    by_ds = defaultdict(list)
    for ds, vid, objs in candidates:
        by_ds[ds].append((vid, objs))
    for ds, vids in sorted(by_ds.items()):
        total_objs = sum(len(o) for _, o in vids)
        log.info(f"  {ds}: {len(vids)} videos, {total_objs} trackable objects")

    # Select target number, trying to balance datasets
    rng = random.Random(args.seed)
    n_target = args.n_videos

    # Try 50/50 split first, fill remainder from larger pool
    ego4d_pool = by_ds.get("ego4d", [])
    vidor_pool = by_ds.get("vidor", [])
    rng.shuffle(ego4d_pool)
    rng.shuffle(vidor_pool)

    n_ego4d = min(n_target // 2, len(ego4d_pool))
    n_vidor = min(n_target - n_ego4d, len(vidor_pool))
    # If one pool is short, take more from the other
    if n_ego4d + n_vidor < n_target:
        remaining = n_target - n_ego4d - n_vidor
        if len(ego4d_pool) > n_ego4d:
            extra = min(remaining, len(ego4d_pool) - n_ego4d)
            n_ego4d += extra
            remaining -= extra
        if remaining > 0 and len(vidor_pool) > n_vidor:
            extra = min(remaining, len(vidor_pool) - n_vidor)
            n_vidor += extra

    selected_ego4d = ego4d_pool[:n_ego4d]
    selected_vidor = vidor_pool[:n_vidor]

    selected = []
    for vid_id, objs in selected_ego4d:
        selected.append(("ego4d", vid_id, objs))
    for vid_id, objs in selected_vidor:
        selected.append(("vidor", vid_id, objs))

    log.info(f"\nSelected {len(selected)} videos ({n_ego4d} ego4d + {n_vidor} vidor)")
    total_objs = sum(len(o) for _, _, o in selected)
    log.info(f"Total trackable objects: {total_objs}")

    if args.dry_run:
        # Print sample
        for ds, vid, objs in selected[:10]:
            obj_str = ", ".join(f"{o['category']}(id={o['object_id']})" for o in objs[:5])
            log.info(f"  {ds}/{vid}: {len(objs)} objects — {obj_str}")
        if len(selected) > 10:
            log.info(f"  ... and {len(selected) - 10} more")
        return

    # Create output directories
    scratch = SCRATCH / "latentlens"
    videos_out = scratch / "pvsg_videos_100"
    frames_out = scratch / "pvsg_frames_100"

    # Clean existing
    if videos_out.exists():
        log.info(f"Removing existing {videos_out}")
        shutil.rmtree(videos_out)
    if frames_out.exists():
        log.info(f"Removing existing {frames_out}")
        shutil.rmtree(frames_out)

    videos_out.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    # Create symlinks and extract frames
    manifest = []
    n_ok = 0
    for idx, (dataset, vid_id, qualified_objs) in enumerate(selected):
        name = f"pvsg_{idx:04d}"
        src_path = DATASETS[dataset]["videos"] / f"{vid_id}.mp4"
        link_path = videos_out / f"{name}.mp4"
        frame_path = frames_out / f"{name}.jpg"

        log.info(f"[{idx+1:03d}/{len(selected)}] {dataset}/{vid_id} "
                 f"({len(qualified_objs)} objects)")

        # Symlink video
        link_path.symlink_to(src_path)

        # Extract middle frame
        ok = extract_middle_frame(src_path, frame_path)
        if ok:
            n_ok += 1
            manifest.append({
                "pvsg_name": name,
                "video_id": vid_id,
                "dataset": dataset,
                "qualified_objects": qualified_objs,
            })
        else:
            log.warning(f"  FAILED frame extraction, removing symlink")
            link_path.unlink()

    # Save manifest
    manifest_path = Path("results/rq5_qualified_videos.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "n_videos": len(manifest),
            "n_ego4d": sum(1 for m in manifest if m["dataset"] == "ego4d"),
            "n_vidor": sum(1 for m in manifest if m["dataset"] == "vidor"),
            "total_trackable_objects": sum(len(m["qualified_objects"]) for m in manifest),
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "min_coverage": args.min_coverage,
            "seed": args.seed,
            "videos": manifest,
        }, f, indent=2)

    log.info(f"\nDone: {n_ok}/{len(selected)} videos prepared")
    log.info(f"Videos: {videos_out}")
    log.info(f"Frames: {frames_out}")
    log.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
