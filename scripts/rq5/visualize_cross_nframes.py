#!/usr/bin/env python3
"""RQ5: Cross-frame-count visualization for tracked objects.

Shows the SAME object from the SAME video across multiple frame counts
(2f, 4f, 8f, 16f) in a single figure. Each row = one frame count,
columns = uniformly subsampled input frames.

Usage:
    python scripts/rq5/visualize_cross_nframes.py \
        --model-key qwen25vl \
        --categories adult child baby dog \
        --n-examples 4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"
VIDEOS_DIR = SCRATCH / "latentlens" / "pvsg_videos_100"

N_FRAMES_LIST = [2, 4, 8, 16]
MAX_DISPLAY_COLS = 6  # max frames shown per row
N_NEIGHBORS = 3
MAX_CAPTION_LEN = 60
FIGURES_DIR = Path("paper/Interpreting-VideoLLMs/figures/rq5_examples")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Utilities ────────────────────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def compute_timestamps(n_frames: int, duration: float) -> list:
    middle_idx = n_frames // 2
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    timestamps[middle_idx] = duration / 2
    return timestamps


def extract_frame(video_path: Path, timestamp: float) -> Image.Image:
    """Frame-accurate extraction (-ss after -i)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "frame.jpg"
        cmd = ["ffmpeg", "-i", str(video_path), "-ss", str(timestamp),
               "-vframes", "1", "-q:v", "2", "-y", str(fp)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if fp.exists():
            return Image.open(fp).convert("RGB").copy()
    return None


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


def load_mask(mask_dir: Path, timestamp: float, n_mask_files: int) -> np.ndarray:
    idx = min(max(0, round(timestamp * 5)), n_mask_files - 1)
    path = mask_dir / f"{idx:04d}.png"
    if path.exists():
        return np.array(Image.open(path))
    return None


def overlay_mask(image: Image.Image, mask: np.ndarray,
                 object_id: int, color=(255, 80, 80),
                 alpha: float = 0.35) -> Image.Image:
    """Semi-transparent mask fill + contour outline."""
    img = np.array(image).astype(np.float32)

    if mask.shape[:2] != img.shape[:2]:
        mask = np.array(
            Image.fromarray(mask).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST))

    obj = (mask == object_id)
    if not obj.any():
        return image

    # Fill
    for c in range(3):
        img[:, :, c] = np.where(
            obj, img[:, :, c] * (1 - alpha) + color[c] * alpha, img[:, :, c])

    # Contour
    eroded = ndimage.binary_erosion(obj, iterations=2)
    contour = obj & ~eroded
    lw = max(1, min(img.shape[:2]) // 200)
    if lw > 1:
        contour = ndimage.binary_dilation(contour, iterations=lw - 1)
    for c in range(3):
        img[:, :, c] = np.where(contour, color[c], img[:, :, c])

    return Image.fromarray(img.astype(np.uint8))


def truncate_caption(caption: str, token_str: str,
                     max_len: int = MAX_CAPTION_LEN) -> str:
    tok = token_str.strip()
    idx = caption.lower().find(tok.lower())
    if idx == -1:
        return caption[:max_len] + ("…" if len(caption) > max_len else "")
    end = idx + len(tok)
    half = (max_len - len(tok)) // 2
    s = max(0, idx - half)
    e = min(len(caption), end + half)
    if s == 0:
        e = min(len(caption), max_len)
    if e == len(caption):
        s = max(0, len(caption) - max_len)
    r = caption[s:e]
    if s > 0:
        r = "…" + r
    if e < len(caption):
        r = r + "…"
    return r


# ── Data loading ─────────────────────────────────────────────────────────

def load_all_data(model_key: str):
    """Load patch maps, POS results, and LatentLens results for all nframes."""
    data = {}
    for nf in N_FRAMES_LIST:
        pm_path = Path(f"results/rq5_patch_map_{model_key}_{nf}f.json")
        pos_path = Path(f"results/rq5_object_pos_{model_key}_{nf}f.json")
        res_path = Path(f"results/pvsg_100_{model_key}_{nf}f_allframes/latentlens_layer24.json")

        if not all(p.exists() for p in [pm_path, pos_path, res_path]):
            log.warning(f"Missing data for {nf}f, skipping")
            continue

        pm = json.load(open(pm_path))
        pos = json.load(open(pos_path))
        res = json.load(open(res_path))

        # Build patch lookup
        patch_lookup = {}
        for vd in res["results"]:
            vn = vd.get("video_name", vd["image_path"].replace(".jpg", ""))
            for fe in vd.get("frames", []):
                fi = fe["frame_idx"]
                for p in fe.get("patches", []):
                    patch_lookup[(vn, fi, p["patch_row"], p["patch_col"])] = p

        # Build POS lookup
        pos_lookup = {(r["video_name"], r["object_id"]): r
                      for r in pos.get("per_object", [])}

        # Build patch-map video lookup
        pm_lookup = {v["video_name"]: v for v in pm["videos"]}

        data[nf] = {
            "patch_map": pm,
            "pos_lookup": pos_lookup,
            "patch_lookup": patch_lookup,
            "pm_lookup": pm_lookup,
        }

    return data


def find_common_objects(data: dict, categories: list = None):
    """Find objects present across ALL frame counts with enough patches."""
    per_nf = {}
    for nf, d in data.items():
        objs = {}
        for v in d["patch_map"]["videos"]:
            for obj in v["objects_in_all_frames"]:
                if categories and obj["category"] not in categories:
                    continue
                key = (v["video_name"], obj["object_id"])
                pos_r = d["pos_lookup"].get(key)
                if pos_r is None:
                    continue
                min_p = min(
                    sum(1 for a in f["patch_assignments"]
                        if a["object_id"] == obj["object_id"])
                    for f in v["frames"])
                if min_p >= 3:
                    objs[key] = {
                        "video_name": v["video_name"],
                        "object_id": obj["object_id"],
                        "category": obj["category"],
                        "is_thing": obj["is_thing"],
                        "min_patches": min_p,
                        "pos_result": pos_r,
                    }
        per_nf[nf] = objs

    # Intersect keys
    common_keys = set.intersection(*(set(v.keys()) for v in per_nf.values()))
    log.info(f"Found {len(common_keys)} objects in all frame counts")

    # Return sorted by category then patch count
    results = []
    for key in common_keys:
        obj = per_nf[max(data.keys())][key]  # use largest nf for info
        obj["key"] = key
        results.append(obj)
    results.sort(key=lambda x: (-x["is_thing"], x["category"], -x["min_patches"]))
    return results


def get_object_nns(pm_video: dict, frame_idx: int, object_id: int,
                   patch_lookup: dict, video_name: str) -> list:
    """Get deduplicated top-N NNs for an object in a frame."""
    frame_data = None
    for f in pm_video["frames"]:
        if f["frame_idx"] == frame_idx:
            frame_data = f
            break
    if frame_data is None:
        return []

    all_nns = []
    for a in frame_data["patch_assignments"]:
        if a["object_id"] != object_id:
            continue
        key = (video_name, frame_idx, a["patch_row"], a["patch_col"])
        pd = patch_lookup.get(key)
        if pd is None:
            continue
        for nn in pd.get("nearest_contextual_neighbors", []):
            all_nns.append(nn)

    seen = {}
    for nn in all_nns:
        tok = nn["token_str"].strip()
        if tok not in seen or nn["similarity"] > seen[tok]["similarity"]:
            seen[tok] = nn
    return sorted(seen.values(), key=lambda x: -x["similarity"])[:N_NEIGHBORS]


# ── Rendering ────────────────────────────────────────────────────────────

def render_cross_nframes(obj_info: dict, data: dict, output_path: Path):
    """Render one figure: rows = frame counts, cols = subsampled input frames."""
    video_name = obj_info["video_name"]
    obj_id = obj_info["object_id"]
    category = obj_info["category"]

    video_id, dataset = resolve_symlink(video_name)
    if video_id is None:
        log.warning(f"Cannot resolve {video_name}")
        return
    video_path = VIDEOS_DIR / f"{video_name}.mp4"
    duration = get_video_duration(video_path)
    mask_dir = PVSG_ROOT / dataset / "masks" / video_id
    n_mask_files = len(list(mask_dir.glob("*.png")))

    # Object color
    from matplotlib.colors import hsv_to_rgb
    hue = (obj_id * 0.618033988749895) % 1.0
    color = tuple(int(c * 255) for c in hsv_to_rgb([hue, 0.7, 0.9]))

    nf_list = sorted(data.keys())
    n_rows = len(nf_list)

    # Determine display columns: use MAX_DISPLAY_COLS, subsample evenly
    n_cols = MAX_DISPLAY_COLS

    # Collect data per row
    row_data = []  # list of (nf, [(frame_img, ts, nns), ...])

    for nf in nf_list:
        d = data[nf]
        pm_video = d["pm_lookup"].get(video_name)
        if pm_video is None:
            continue

        # Compute all input-frame timestamps
        max_input = max(
            idx for f in pm_video["frames"]
            for idx in f.get("input_frames", [f["frame_idx"]])
        ) + 1
        all_ts = compute_timestamps(max_input, duration)

        # Collect all input frames with their temporal-step's NNs
        all_frames = []
        for frame_entry in pm_video["frames"]:
            fi = frame_entry["frame_idx"]
            input_frames = frame_entry.get("input_frames", [fi])
            nns = get_object_nns(pm_video, fi, obj_id,
                                 d["patch_lookup"], video_name)
            for inp_idx in input_frames:
                ts = all_ts[inp_idx]
                all_frames.append((ts, nns))

        # Subsample to n_cols
        if len(all_frames) <= n_cols:
            selected = all_frames
        else:
            indices = np.linspace(0, len(all_frames) - 1, n_cols, dtype=int)
            selected = [all_frames[i] for i in indices]

        # Extract frames and overlay masks
        frame_entries = []
        for ts, nns in selected:
            img = extract_frame(video_path, ts)
            if img is None:
                continue
            mask = load_mask(mask_dir, ts, n_mask_files)
            if mask is not None:
                img = overlay_mask(img, mask, obj_id, color)
            frame_entries.append((img, ts, nns))

        row_data.append((nf, frame_entries))

    if not row_data:
        return

    # Actual columns = max frames across rows
    actual_cols = max(len(entries) for _, entries in row_data)
    if actual_cols == 0:
        return

    # ── Layout ──
    col_width = 2.2
    img_height = 2.8
    text_height = 1.2
    row_height = img_height + text_height
    header_height = 0.5
    label_width = 0.6  # left margin for nf labels
    fig_width = label_width + col_width * actual_cols
    fig_height = header_height + row_height * n_rows + 0.2

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Header
    fig.text(0.01, 1 - header_height / (2 * fig_height),
             f"{video_name}  ·  {category} (id={obj_id})",
             fontsize=11, fontweight="bold", va="center", fontfamily="sans-serif")

    # Check stability across nf
    stab_texts = []
    for nf in nf_list:
        pr = data[nf]["pos_lookup"].get((video_name, obj_id))
        if pr:
            stab_texts.append(f"{nf}f:{'✓' if pr['mode_stable'] else '✗'}")
    fig.text(0.99, 1 - header_height / (2 * fig_height),
             "  ".join(stab_texts),
             fontsize=9, ha="right", va="center", fontfamily="monospace",
             color="#555555")

    for row_idx, (nf, entries) in enumerate(row_data):
        y_top = 1 - (header_height + row_idx * row_height) / fig_height
        y_img_bottom = y_top - img_height / fig_height
        y_text_bottom = y_img_bottom - text_height / fig_height

        # Row label (nf)
        fig.text(0.005, (y_top + y_img_bottom) / 2,
                 f"{nf}f", fontsize=10, fontweight="bold",
                 va="center", ha="left", rotation=0,
                 fontfamily="sans-serif", color="#333333")

        for col_idx, (img, ts, nns) in enumerate(entries):
            x_left = (label_width + col_idx * col_width) / fig_width
            w = col_width / fig_width * 0.92
            h_img = img_height / fig_height * 0.92

            # Image
            ax_img = fig.add_axes([x_left, y_img_bottom + 0.01,
                                   w, h_img])
            ax_img.imshow(img)
            ax_img.set_title(f"t={ts:.1f}s", fontsize=7, pad=2)
            ax_img.axis("off")

            # NNs text
            ax_txt = fig.add_axes([x_left, y_text_bottom, w,
                                   text_height / fig_height * 0.9])
            ax_txt.axis("off")
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)

            if nns:
                line_h = 1.0 / (N_NEIGHBORS + 0.5)
                y = 1.0 - line_h * 0.3
                for nn in nns:
                    tok = nn.get("token_str", "").strip()
                    cap = nn.get("caption", "")
                    sim = nn.get("similarity", 0.0)
                    trunc = truncate_caption(cap, tok)

                    idx = trunc.lower().find(tok.lower()) if tok else -1
                    if idx >= 0:
                        before = trunc[:idx]
                        bold = trunc[idx:idx + len(tok)]
                        after = trunc[idx + len(tok):]
                        line = f"[{sim:.2f}] {before}$\\bf{{{bold}}}${after}"
                    else:
                        line = f"[{sim:.2f}] {trunc}"

                    ax_txt.text(0.0, y, line, fontsize=5.5,
                                fontfamily="serif", color="#333333",
                                va="top", clip_on=True)
                    y -= line_h

    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_path.with_suffix(".png"), dpi=150,
                bbox_inches="tight", pad_inches=0.08)
    plt.close()
    log.info(f"Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Cross-frame-count object tracking visualization")
    parser.add_argument("--model-key", type=str, default="qwen25vl",
                        choices=["molmo2", "qwen25vl"])
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--video-id", type=str, default=None,
                        help="Generate for a specific video (e.g. pvsg_0058)")
    parser.add_argument("--object-id", type=int, default=None,
                        help="Specific object ID within the video")
    args = parser.parse_args()

    data = load_all_data(args.model_key)
    if not data:
        log.error("No data loaded")
        return

    common = find_common_objects(data, categories=args.categories)
    if not common:
        log.error("No common objects found")
        return

    # Filter by video-id if specified
    if args.video_id:
        common = [o for o in common if o["video_name"] == args.video_id]
        if args.object_id is not None:
            common = [o for o in common if o["object_id"] == args.object_id]
        if not common:
            log.error(f"No common objects found for {args.video_id}")
            return
        selected = common[:args.n_examples]
    else:
        # Select diverse examples
        selected = []
        seen_cats = set()
        for obj in common:
            if len(selected) >= args.n_examples:
                break
            if obj["category"] not in seen_cats:
                selected.append(obj)
                seen_cats.add(obj["category"])

    log.info(f"Selected {len(selected)} examples: "
             + ", ".join(f'{s["video_name"]}/{s["category"]}' for s in selected))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for i, obj in enumerate(selected):
        stem = f"rq5_cross_{obj['video_name']}_{obj['category']}"
        out = FIGURES_DIR / f"{stem}.pdf"
        log.info(f"[{i+1}/{len(selected)}] {obj['video_name']} / "
                 f"{obj['category']} (id={obj['object_id']})")
        render_cross_nframes(obj, data, out)

    log.info("Done!")


if __name__ == "__main__":
    main()
