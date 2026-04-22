#!/usr/bin/env python3
"""Map PVSG object masks to spatial patches from all-frames LatentLens results.

For each video and each frame, determines which spatial patches overlap with
which PVSG objects (>50% coverage). Filters to objects present in ALL frames.

Also supports n_frames=1 using existing _spatial/ results (flat patch format).

Usage:
    python scripts/rq5/map_masks_to_patches.py \\
        --results-dir results/pvsg_100_molmo2_4f_allframes/ \\
        --n-frames 4 --layer 24 \\
        --output results/rq5_patch_map_molmo2_4f.json
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"
PVSG_JSON = PVSG_ROOT / "PVSG_Dataset.json"
VIDEOS_DIR = SCRATCH / "latentlens" / "pvsg_videos_100"

MIN_COVERAGE = 0.5


def load_pvsg_metadata():
    """Load PVSG metadata and build mappings."""
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


def resolve_symlink(pvsg_name: str) -> tuple:
    """Resolve pvsg_NNNN symlink to (video_id, dataset)."""
    link_path = VIDEOS_DIR / f"{pvsg_name}.mp4"
    if not link_path.is_symlink():
        return None, None
    target = link_path.resolve()
    # Path like .../pvsg/{dataset}/videos/{video_id}.mp4
    parts = target.parts
    # Find 'pvsg' in path, then dataset is next
    for i, part in enumerate(parts):
        if part == "pvsg" and i + 2 < len(parts):
            dataset = parts[i + 1]
            video_id = target.stem
            return video_id, dataset
    return target.stem, "unknown"


def compute_timestamps(n_frames: int, duration: float) -> list:
    """Same formula as extract_frames in run_latentlens_video.py."""
    middle_idx = n_frames // 2
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    timestamps[middle_idx] = duration / 2
    return timestamps


def load_mask(mask_dir: Path, mask_frame_idx: int) -> np.ndarray:
    """Load a PVSG mask PNG using PIL (palette mode)."""
    mask_path = mask_dir / f"{mask_frame_idx:04d}.png"
    if not mask_path.exists():
        return None
    return np.array(Image.open(mask_path))


def assign_patches_to_objects(
    mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    min_coverage: float = MIN_COVERAGE,
) -> list:
    """Assign each patch to the object with >min_coverage of its area.

    Returns list of {patch_row, patch_col, object_id, coverage} for assigned patches.
    """
    mask_h, mask_w = mask.shape
    patch_h = mask_h / grid_h
    patch_w = mask_w / grid_w

    assignments = []
    for r in range(grid_h):
        for c in range(grid_w):
            y0 = int(r * patch_h)
            y1 = int((r + 1) * patch_h)
            x0 = int(c * patch_w)
            x1 = int((c + 1) * patch_w)

            patch_region = mask[y0:y1, x0:x1]
            total_pixels = patch_region.size
            if total_pixels == 0:
                continue

            # Find dominant object
            unique_ids, counts = np.unique(patch_region, return_counts=True)
            for obj_id, count in zip(unique_ids, counts):
                if obj_id == 0:  # background
                    continue
                coverage = count / total_pixels
                if coverage >= min_coverage:
                    assignments.append({
                        "patch_row": r,
                        "patch_col": c,
                        "object_id": int(obj_id),
                        "coverage": round(float(coverage), 3),
                    })
                    break  # Only assign to the first object exceeding threshold

    return assignments


def process_video_allframes(
    video_data: dict,
    vid2entry: dict,
    vid2src: dict,
    n_frames: int,
    min_coverage: float,
) -> dict:
    """Process one video from allframes LatentLens results."""
    video_name = video_data.get("video_name", video_data["image_path"].replace(".jpg", ""))
    video_id, dataset = resolve_symlink(video_name)

    if video_id is None:
        log.warning(f"  Cannot resolve symlink for {video_name}")
        return None

    entry = vid2entry.get(video_id)
    if entry is None:
        log.warning(f"  {video_id} not in PVSG_Dataset.json")
        return None

    meta = entry["meta"]
    duration = meta["duration"]
    mask_dir = PVSG_ROOT / dataset / "masks" / video_id

    if not mask_dir.exists():
        log.warning(f"  No mask dir for {video_id}")
        return None

    # Count available mask files
    mask_files = sorted(mask_dir.glob("*.png"))
    n_mask_files = len(mask_files)
    if n_mask_files == 0:
        return None

    # Build object ID → category mapping
    obj_info = {obj["object_id"]: obj for obj in entry.get("objects", [])}

    timestamps = compute_timestamps(n_frames, duration)
    frames_data = video_data.get("frames", [])

    all_assignments = []
    for frame_entry in frames_data:
        frame_idx = frame_entry["frame_idx"]
        input_frames = frame_entry.get("input_frames", [frame_idx])
        grid_h = frame_entry["grid_h"]
        grid_w = frame_entry["grid_w"]

        # Use middle input frame's timestamp for mask lookup
        mid_input = input_frames[len(input_frames) // 2]
        ts = timestamps[mid_input]
        mask_frame_idx = min(round(ts * 5), n_mask_files - 1)
        mask_frame_idx = max(0, mask_frame_idx)

        mask = load_mask(mask_dir, mask_frame_idx)
        if mask is None:
            log.warning(f"  Missing mask frame {mask_frame_idx} for {video_id}")
            continue

        # Resize mask if needed (NEAREST to preserve object IDs)
        # We don't have the actual frame size, but mask should match video resolution
        # The patch grid maps to the mask directly
        assignments = assign_patches_to_objects(mask, grid_h, grid_w, min_coverage)

        all_assignments.append({
            "frame_idx": frame_idx,
            "input_frames": input_frames,
            "mask_frame_idx": mask_frame_idx,
            "timestamp": round(ts, 2),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "patch_assignments": assignments,
        })

    # Filter to objects present in ALL frames
    n_output_frames = len(all_assignments)
    obj_frame_count = defaultdict(int)
    for frame_assign in all_assignments:
        frame_obj_ids = set(a["object_id"] for a in frame_assign["patch_assignments"])
        for oid in frame_obj_ids:
            obj_frame_count[oid] += 1

    objects_in_all = [oid for oid, count in obj_frame_count.items()
                      if count == n_output_frames]

    if not objects_in_all:
        return None

    # Build qualified objects list
    qualified_objects = []
    for oid in sorted(objects_in_all):
        info = obj_info.get(oid, {})
        qualified_objects.append({
            "object_id": oid,
            "category": info.get("category", "unknown"),
            "is_thing": info.get("is_thing", True),
        })

    # Filter assignments to only include qualified objects
    for frame_assign in all_assignments:
        frame_assign["patch_assignments"] = [
            a for a in frame_assign["patch_assignments"]
            if a["object_id"] in objects_in_all
        ]

    return {
        "video_name": video_name,
        "video_id": video_id,
        "dataset": dataset,
        "objects_in_all_frames": qualified_objects,
        "n_output_frames": n_output_frames,
        "frames": all_assignments,
    }


def process_video_1f(
    video_data: dict,
    vid2entry: dict,
    vid2src: dict,
    min_coverage: float,
) -> dict:
    """Process one video from existing 1f spatial results (flat patch format)."""
    video_name = video_data["image_path"].replace(".jpg", "")
    video_id, dataset = resolve_symlink(video_name)

    if video_id is None:
        return None

    entry = vid2entry.get(video_id)
    if entry is None:
        return None

    meta = entry["meta"]
    duration = meta["duration"]
    mask_dir = PVSG_ROOT / dataset / "masks" / video_id

    if not mask_dir.exists():
        return None

    mask_files = sorted(mask_dir.glob("*.png"))
    n_mask_files = len(mask_files)
    if n_mask_files == 0:
        return None

    obj_info = {obj["object_id"]: obj for obj in entry.get("objects", [])}

    # 1 frame → middle frame at duration/2
    ts = duration / 2
    mask_frame_idx = min(round(ts * 5), n_mask_files - 1)
    mask_frame_idx = max(0, mask_frame_idx)

    mask = load_mask(mask_dir, mask_frame_idx)
    if mask is None:
        return None

    grid_h = video_data.get("grid_h", 14)
    grid_w = video_data.get("grid_w", 14)

    assignments = assign_patches_to_objects(mask, grid_h, grid_w, min_coverage)
    if not assignments:
        return None

    # All objects present (single frame = trivially "all frames")
    obj_ids = set(a["object_id"] for a in assignments)
    qualified_objects = []
    for oid in sorted(obj_ids):
        info = obj_info.get(oid, {})
        qualified_objects.append({
            "object_id": oid,
            "category": info.get("category", "unknown"),
            "is_thing": info.get("is_thing", True),
        })

    return {
        "video_name": video_name,
        "video_id": video_id,
        "dataset": dataset,
        "objects_in_all_frames": qualified_objects,
        "n_output_frames": 1,
        "frames": [{
            "frame_idx": 0,
            "input_frames": [0],
            "mask_frame_idx": mask_frame_idx,
            "timestamp": round(ts, 2),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "patch_assignments": assignments,
        }],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Map PVSG masks to spatial patches"
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="LatentLens results directory (allframes or _spatial for 1f)")
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vid2entry, vid2src = load_pvsg_metadata()

    # Load LatentLens results
    results_path = args.results_dir / f"latentlens_layer{args.layer}.json"
    with open(results_path) as f:
        data = json.load(f)
    log.info(f"Loaded {len(data['results'])} videos from {results_path}")

    is_1f = (args.n_frames == 1)

    videos_output = []
    n_skipped = 0
    total_objects = 0

    for video_data in data["results"]:
        if is_1f:
            result = process_video_1f(video_data, vid2entry, vid2src, args.min_coverage)
        else:
            result = process_video_allframes(
                video_data, vid2entry, vid2src, args.n_frames, args.min_coverage
            )

        if result is None:
            n_skipped += 1
            continue

        videos_output.append(result)
        total_objects += len(result["objects_in_all_frames"])

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_frames": args.n_frames,
        "layer": args.layer,
        "min_coverage": args.min_coverage,
        "n_videos": len(videos_output),
        "n_skipped": n_skipped,
        "total_trackable_objects": total_objects,
        "videos": videos_output,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {args.output}")
    log.info(f"  {len(videos_output)} videos kept, {n_skipped} skipped")
    log.info(f"  {total_objects} total trackable objects")


if __name__ == "__main__":
    main()
