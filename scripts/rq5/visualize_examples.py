#!/usr/bin/env python3
"""RQ5: Generate paper-quality example figures showing object tracking.

Shows N frames side-by-side with the tracked object highlighted via
colored mask overlay + bbox, with top-3 NN tokens shown below each frame.

Usage:
    python scripts/rq5/visualize_examples.py \\
        --patch-map results/rq5_patch_map_molmo2_4f.json \\
        --pos-results results/rq5_object_pos_molmo2_4f.json \\
        --results-dir results/pvsg_100_molmo2_4f_allframes/ \\
        --n-frames 4 --model-key molmo2
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"
VIDEOS_DIR = SCRATCH / "latentlens" / "pvsg_videos_100"

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def compute_timestamps(n_frames: int, duration: float) -> list:
    """Compute uniform timestamps (same formula as extract_frames)."""
    middle_idx = n_frames // 2
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    timestamps[middle_idx] = duration / 2
    return timestamps


MAX_CAPTION_LEN = 70
N_NEIGHBORS = 3

# Import from parent scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve_symlink(pvsg_name: str):
    link = VIDEOS_DIR / f"{pvsg_name}.mp4"
    if not link.is_symlink():
        return None, None
    target = link.resolve()
    parts = target.parts
    for i, part in enumerate(parts):
        if part == "pvsg" and i + 2 < len(parts):
            return target.stem, parts[i + 1]
    return target.stem, "unknown"


def extract_frame_at_timestamp(video_path: Path, timestamp: float) -> Image.Image:
    """Extract a single frame from video at given timestamp.

    Uses -ss after -i for frame-accurate seeking (slower but avoids
    keyframe-based misalignment that causes mask/frame mismatch).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "frame.jpg"
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ss", str(timestamp),
            "-vframes", "1", "-q:v", "2",
            "-y", str(frame_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if frame_path.exists():
            return Image.open(frame_path).convert("RGB").copy()
    return None


def load_mask_at_timestamp(video_id: str, dataset: str, timestamp: float,
                           n_mask_files: int) -> np.ndarray:
    """Load PVSG mask for nearest frame to timestamp."""
    mask_idx = min(round(timestamp * 5), n_mask_files - 1)
    mask_idx = max(0, mask_idx)
    mask_dir = PVSG_ROOT / dataset / "masks" / video_id
    mask_path = mask_dir / f"{mask_idx:04d}.png"
    if mask_path.exists():
        return np.array(Image.open(mask_path))
    return None


def overlay_object_mask(image: Image.Image, mask: np.ndarray,
                        object_id: int, color=(255, 0, 0),
                        alpha: float = 0.35) -> Image.Image:
    """Overlay a colored mask with contour outline for a specific object."""
    from scipy import ndimage

    img_array = np.array(image).astype(np.float32)

    # Resize mask to image dimensions if needed
    if mask.shape[:2] != img_array.shape[:2]:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]),
                                   Image.NEAREST)
        mask = np.array(mask_pil)

    obj_mask = (mask == object_id)

    # Semi-transparent color fill
    for c in range(3):
        img_array[:, :, c] = np.where(
            obj_mask,
            img_array[:, :, c] * (1 - alpha) + color[c] * alpha,
            img_array[:, :, c],
        )

    # Draw contour outline (erode mask and take boundary)
    eroded = ndimage.binary_erosion(obj_mask, iterations=2)
    contour = obj_mask & ~eroded
    lw = max(1, min(img_array.shape[:2]) // 200)
    if lw > 1:
        contour = ndimage.binary_dilation(contour, iterations=lw - 1)
    for c in range(3):
        img_array[:, :, c] = np.where(contour, color[c], img_array[:, :, c])

    return Image.fromarray(img_array.astype(np.uint8))


def draw_object_bbox(image: Image.Image, mask: np.ndarray,
                     object_id: int, grid_h: int, grid_w: int) -> Image.Image:
    """Draw a red bounding box around the object's patch region."""
    img_w, img_h = image.size

    # Resize mask to image dims
    if mask.shape[:2] != (img_h, img_w):
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
        mask = np.array(mask_pil)

    obj_mask = (mask == object_id)
    if not obj_mask.any():
        return image

    rows, cols = np.where(obj_mask)
    y0, y1 = rows.min(), rows.max()
    x0, x1 = cols.min(), cols.max()

    # Add small padding
    pad = max(2, min(img_w, img_h) // 80)
    y0 = max(0, y0 - pad)
    y1 = min(img_h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(img_w - 1, x1 + pad)

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    lw = max(2, min(img_w, img_h) // 150)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=lw)

    return img_copy


def truncate_caption(caption: str, token_str: str, max_len: int = MAX_CAPTION_LEN) -> str:
    """Truncate caption around the matched token."""
    token_clean = token_str.strip()
    idx = caption.lower().find(token_clean.lower())
    if idx == -1:
        return caption[:max_len] + ("..." if len(caption) > max_len else "")

    token_end = idx + len(token_clean)
    half = (max_len - len(token_clean)) // 2
    start = max(0, idx - half)
    end = min(len(caption), token_end + half)
    if start == 0:
        end = min(len(caption), max_len)
    if end == len(caption):
        start = max(0, len(caption) - max_len)

    result = caption[start:end]
    if start > 0:
        result = "..." + result
    if end < len(caption):
        result = result + "..."
    return result


def get_object_nns(video_entry: dict, frame_idx: int, object_id: int,
                   patch_lookup: dict, video_name: str,
                   n_neighbors: int = N_NEIGHBORS) -> list:
    """Get top NN tokens for an object's patches in a specific frame."""
    frame_data = None
    for f in video_entry["frames"]:
        if f["frame_idx"] == frame_idx:
            frame_data = f
            break
    if frame_data is None:
        return []

    obj_patches = [a for a in frame_data["patch_assignments"]
                   if a["object_id"] == object_id]

    # Collect all NNs from all patches of this object
    all_nns = []
    for assignment in obj_patches:
        key = (video_name, frame_idx,
               assignment["patch_row"], assignment["patch_col"])
        patch_data = patch_lookup.get(key)
        if patch_data is None:
            continue
        nns = patch_data.get("nearest_contextual_neighbors", [])
        for nn in nns:
            all_nns.append(nn)

    # Deduplicate by token string, keep highest similarity
    seen = {}
    for nn in all_nns:
        tok = nn["token_str"].strip()
        if tok not in seen or nn["similarity"] > seen[tok]["similarity"]:
            seen[tok] = nn

    # Sort by similarity and take top N
    sorted_nns = sorted(seen.values(), key=lambda x: -x["similarity"])
    return sorted_nns[:n_neighbors]


def render_object_tracking(
    video_entry: dict,
    obj: dict,
    pos_result: dict,
    patch_lookup: dict,
    output_path: Path,
):
    """Render a figure showing one object tracked across frames."""
    video_name = video_entry["video_name"]
    video_id, dataset = resolve_symlink(video_name)
    if video_id is None:
        log.warning(f"Cannot resolve {video_name}")
        return

    video_path = VIDEOS_DIR / f"{video_name}.mp4"
    mask_dir = PVSG_ROOT / dataset / "masks" / video_id
    n_mask_files = len(list(mask_dir.glob("*.png")))

    n_frames = len(video_entry["frames"])
    obj_id = obj["object_id"]
    category = obj["category"]
    is_stable = pos_result["mode_stable"]
    dominant_pos = pos_result["dominant_pos"]

    # Object color for overlay
    hue = (obj_id * 0.618033988749895) % 1.0
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb([hue, 0.7, 0.9])
    color = tuple(int(c * 255) for c in rgb)

    # Compute timestamps for ALL input frames (before temporal compression)
    n_input_frames = video_entry["frames"][0].get("input_frames", [0])
    # Total input frames = max input_frame index + 1
    max_input = max(
        idx for f in video_entry["frames"]
        for idx in f.get("input_frames", [f["frame_idx"]])
    ) + 1
    all_timestamps = compute_timestamps(max_input, get_video_duration(video_path))

    # Collect frames + masks + NNs
    # Show ALL input frames (not just temporal steps)
    frame_images = []
    frame_nns = []

    for frame_data in video_entry["frames"]:
        frame_idx = frame_data["frame_idx"]
        input_frames = frame_data.get("input_frames", [frame_idx])
        grid_h = frame_data["grid_h"]
        grid_w = frame_data["grid_w"]

        # Get NNs once per temporal step (shared across merged input frames)
        nns = get_object_nns(video_entry, frame_idx, obj_id,
                             patch_lookup, video_name)

        for inp_idx in input_frames:
            ts = all_timestamps[inp_idx]

            # Extract frame from video
            frame_img = extract_frame_at_timestamp(video_path, ts)
            if frame_img is None:
                continue

            # Load mask and overlay (mask only, no bounding box)
            mask = load_mask_at_timestamp(video_id, dataset, ts, n_mask_files)
            if mask is not None:
                frame_img = overlay_object_mask(frame_img, mask, obj_id, color)

            frame_images.append((frame_img, ts))
            frame_nns.append(nns)

    if not frame_images:
        return

    # Render figure
    n_cols = len(frame_images)
    n_nn_lines = max(len(nns) for nns in frame_nns) if frame_nns else 0
    text_height = (n_nn_lines + 1) * 0.25 + 0.3
    img_height = 4.0
    header_height = 0.5
    fig_height = header_height + img_height + text_height

    fig = plt.figure(figsize=(4 * n_cols, fig_height))
    gs = GridSpec(3, n_cols,
                  height_ratios=[header_height, img_height, text_height],
                  hspace=0.05)

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    stability_text = "consistent" if is_stable else "inconsistent"
    stability_color = "#2e7d32" if is_stable else "#c62828"
    ax_header.text(0.0, 0.5,
                   f"{video_name}  |  {category} (id={obj_id})  |  "
                   f"dominant: {dominant_pos}",
                   fontsize=12, fontweight="bold",
                   transform=ax_header.transAxes, va="center")
    ax_header.text(0.99, 0.5, stability_text,
                   fontsize=11, fontstyle="italic", color=stability_color,
                   ha="right", transform=ax_header.transAxes, va="center")

    # Frame images
    for col_idx, (frame_img, ts) in enumerate(frame_images):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.imshow(frame_img)
        ax.set_title(f"t={ts:.1f}s", fontsize=9)
        ax.axis("off")

    # NN text below each frame
    for col_idx, nns in enumerate(frame_nns):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if not nns:
            continue

        line_h = 1.0 / max(n_nn_lines + 1, 1)
        y = 1.0 - line_h * 0.5

        for nn in nns:
            token_str = nn.get("token_str", "").strip()
            caption = nn.get("caption", "")
            sim = nn.get("similarity", 0.0)
            truncated = truncate_caption(caption, token_str)

            idx = truncated.lower().find(token_str.lower()) if token_str else -1
            if idx >= 0:
                before = truncated[:idx]
                bold = truncated[idx:idx + len(token_str)]
                after = truncated[idx + len(token_str):]
                line_str = f"[{sim:.2f}] {before}$\\bf{{{bold}}}${after}"
            else:
                line_str = f"[{sim:.2f}] {truncated}"

            ax.text(0.02, y, line_str, fontsize=7,
                    fontfamily="serif", color="#333333",
                    transform=ax.transAxes, va="center",
                    clip_on=True)
            y -= line_h

    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    log.info(f"  Saved {output_path} + {png_path}")


def select_examples(patch_map: dict, pos_results: dict,
                    n_examples: int = 6,
                    categories: list = None) -> list:
    """Select interesting examples: mix of stable/unstable, diverse categories.

    If categories is provided, only select objects from those categories.
    """
    # Build lookup from pos_results
    pos_lookup = {}
    for r in pos_results.get("per_object", []):
        key = (r["video_name"], r["object_id"])
        pos_lookup[key] = r

    candidates = []
    for video_entry in patch_map["videos"]:
        video_name = video_entry["video_name"]
        for obj in video_entry["objects_in_all_frames"]:
            key = (video_name, obj["object_id"])
            pos_result = pos_lookup.get(key)
            if pos_result is None:
                continue

            # Category filter
            if categories and obj["category"] not in categories:
                continue

            # Quality filter: need enough patches per frame
            min_patches = min(
                sum(1 for a in f["patch_assignments"]
                    if a["object_id"] == obj["object_id"])
                for f in video_entry["frames"]
            )
            if min_patches < 3:
                continue

            candidates.append({
                "video_entry": video_entry,
                "obj": obj,
                "pos_result": pos_result,
                "min_patches": min_patches,
                "is_stable": pos_result["mode_stable"],
                "category": obj["category"],
            })

    # Select diverse examples
    stable = [c for c in candidates if c["is_stable"]]
    unstable = [c for c in candidates if not c["is_stable"]]

    # Sort by patch count (prefer objects with more patches — more visible)
    stable.sort(key=lambda x: -x["min_patches"])
    unstable.sort(key=lambda x: -x["min_patches"])

    # Pick from diverse categories
    selected = []
    seen_categories = set()

    for pool in [stable, unstable]:
        for c in pool:
            if len(selected) >= n_examples:
                break
            if c["category"] not in seen_categories or len(seen_categories) >= n_examples:
                selected.append(c)
                seen_categories.add(c["category"])

    return selected[:n_examples]


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Generate object tracking example figures"
    )
    parser.add_argument("--patch-map", type=Path, required=True)
    parser.add_argument("--pos-results", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--model-key", type=str, required=True,
                        choices=["molmo2", "qwen25vl"])
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Filter to specific PVSG categories (e.g., adult child dog)")
    args = parser.parse_args()

    # Load data
    with open(args.patch_map) as f:
        patch_map = json.load(f)
    with open(args.pos_results) as f:
        pos_results = json.load(f)

    # Load LatentLens results for NN lookup
    results_path = args.results_dir / f"latentlens_layer{args.layer}.json"
    with open(results_path) as f:
        results_data = json.load(f)

    is_1f = (args.n_frames == 1)

    # Build patch lookup
    patch_lookup = {}
    for video_data in results_data["results"]:
        video_name = video_data.get("video_name",
                                     video_data["image_path"].replace(".jpg", ""))
        if is_1f:
            for patch in video_data.get("patches", []):
                key = (video_name, 0, patch["patch_row"], patch["patch_col"])
                patch_lookup[key] = patch
        else:
            for frame_entry in video_data.get("frames", []):
                frame_idx = frame_entry["frame_idx"]
                for patch in frame_entry.get("patches", []):
                    key = (video_name, frame_idx, patch["patch_row"], patch["patch_col"])
                    patch_lookup[key] = patch

    log.info(f"Patch lookup: {len(patch_lookup)} entries")

    # Select examples
    examples = select_examples(patch_map, pos_results, args.n_examples,
                               categories=args.categories)
    log.info(f"Selected {len(examples)} examples")

    # Render
    output_dir = Path("paper/Interpreting-VideoLLMs/figures/rq5_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(examples):
        video_name = ex["video_entry"]["video_name"]
        category = ex["category"]
        stem = f"{video_name}_{category}_{args.n_frames}f"
        output_path = output_dir / f"{stem}.pdf"

        log.info(f"[{i+1}/{len(examples)}] {video_name} / {category} "
                 f"(stable={ex['is_stable']}, patches={ex['min_patches']})")

        render_object_tracking(
            ex["video_entry"], ex["obj"], ex["pos_result"],
            patch_lookup, output_path,
        )

    log.info("Done!")


if __name__ == "__main__":
    main()
