#!/usr/bin/env python3
"""Run LatentLens with a sliding window across the full video duration.

For each target frame at time t, feeds up to 3 preceding frames + the target
frame (4f context) to the model. Extracts NNs for the LAST frame's visual
tokens only (the target frame). Slides across the entire video at a given fps.

Output: one JSON per layer with per-frame patches for every sampled timestamp.

Usage:
    python scripts/rq5/run_latentlens_sliding.py \
        --model allenai/Molmo2-8B \
        --index McGill-NLP/contextual_embeddings-molmo2-8b \
        --video $SCRATCH/latentlens/pvsg_videos_100/pvsg_0058.mp4 \
        --output-dir results/pvsg_demo_molmo2_sliding/ \
        --context-frames 3 --fps 1 --layer 24 --device cuda:0
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_num_threads(8)
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import latentlens
from latentlens import ContextualIndex

from run_latentlens_video import (
    get_video_duration,
    prepare_inputs_video,
    get_visual_token_mask,
)
from rq5.run_latentlens_allframes import get_all_frame_token_ranges

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_frame_at(video_path: Path, timestamp: float) -> Image.Image:
    """Extract a single frame at the given timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "frame.jpg"
        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            "-y", str(frame_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if frame_path.exists():
            return Image.open(frame_path).convert("RGB").copy()
    return None


def run_latentlens_last_frame(
    model, processor, index, frames, model_name, device, layer, n_input_frames, top_k=5,
):
    """Run model on frames, extract NNs for the LAST frame only.

    Returns {grid_h, grid_w, patches} for the last frame.
    """
    model_dtype = next(p.dtype for p in model.parameters())
    inputs = prepare_inputs_video(processor, frames, model_name, device, model_dtype)

    visual_mask = get_visual_token_mask(inputs, model_name)
    frame_ranges = get_all_frame_token_ranges(inputs, model_name, n_input_frames)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    hs = hidden_states[layer][0]
    all_visual_features = hs[visual_mask]

    # Last frame's range
    start, end, grid_h, grid_w, input_frames = frame_ranges[-1]
    frame_features = all_visual_features[start:end]
    frame_features = F.normalize(frame_features.float(), dim=-1)

    neighbors_per_token = index.search(
        frame_features.to(index.device), top_k=top_k
    )

    patches = []
    for patch_idx, neighbors in enumerate(neighbors_per_token):
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        nn_list = [
            {
                "token_str": nb.token_str,
                "similarity": nb.similarity,
                "caption": nb.caption,
                "contextual_layer": nb.contextual_layer,
                "position": nb.position,
            }
            for nb in neighbors
        ]
        patches.append({
            "patch_idx": patch_idx,
            "patch_row": row,
            "patch_col": col,
            "nearest_contextual_neighbors": nn_list,
        })

    del hidden_states, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"grid_h": grid_h, "grid_w": grid_w, "patches": patches}


def main():
    parser = argparse.ArgumentParser(
        description="Run LatentLens with sliding window across full video"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True,
                        help="Video file paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=3,
                        help="Number of preceding context frames (default: 3, so 4f total)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Sampling rate in frames per second")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--index-device", type=str, default="cpu",
                        help="Device for index (default: cpu to avoid OOM)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load model
    log.info(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)
    model = None
    for auto_cls in [AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq]:
        try:
            model = auto_cls.from_pretrained(args.model, **load_kwargs)
            log.info(f"Loaded with {auto_cls.__name__}")
            break
        except (ValueError, KeyError, TypeError):
            continue
    if model is None:
        raise RuntimeError(f"Could not load {args.model} with any Auto class")
    model = model.to(device).eval()

    # Load index (on separate device to avoid OOM)
    index_device = torch.device(args.index_device)
    log.info(f"Loading index: {args.index} (on {index_device})")
    index_path = Path(args.index)
    if index_path.exists():
        index = ContextualIndex.from_directory(str(index_path))
    else:
        index = ContextualIndex.from_pretrained(args.index)
    index = index.to(index_device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_context = args.context_frames
    n_total_frames = n_context + 1  # context + target

    for video_path in args.videos:
        video_name = video_path.stem
        log.info(f"Processing {video_name}...")

        duration = get_video_duration(video_path)
        # Sample timestamps at the given fps across the full duration
        n_samples = max(1, int(duration * args.fps))
        timestamps = [(i + 0.5) / args.fps for i in range(n_samples)]
        # Clamp to duration
        timestamps = [min(t, duration - 0.01) for t in timestamps]

        log.info(f"  Duration: {duration:.1f}s, sampling {n_samples} frames at {args.fps} fps")

        # Pre-extract all unique frames we'll need
        all_needed_ts = set()
        for t_idx, target_ts in enumerate(timestamps):
            # Context timestamps: preceding frames at the same fps
            for c in range(n_context, 0, -1):
                ctx_ts = timestamps[t_idx - c] if t_idx - c >= 0 else None
                if ctx_ts is not None:
                    all_needed_ts.add(ctx_ts)
            all_needed_ts.add(target_ts)

        log.info(f"  Extracting {len(all_needed_ts)} unique frames...")
        frame_cache = {}
        for ts in sorted(all_needed_ts):
            img = extract_frame_at(video_path, ts)
            if img is not None:
                frame_cache[ts] = img

        # Sliding window inference
        frame_results = []
        for t_idx, target_ts in enumerate(tqdm(timestamps, desc=f"  {video_name}")):
            if target_ts not in frame_cache:
                log.warning(f"  Skipping t={target_ts:.2f}s: no frame")
                continue

            # Build window: up to n_context preceding + target
            window_ts = []
            for c in range(n_context, 0, -1):
                prev_idx = t_idx - c
                if prev_idx >= 0 and timestamps[prev_idx] in frame_cache:
                    window_ts.append(timestamps[prev_idx])
            window_ts.append(target_ts)

            window_frames = [frame_cache[ts] for ts in window_ts]
            n_input = len(window_frames)

            try:
                result = run_latentlens_last_frame(
                    model, processor, index, window_frames,
                    args.model, device, args.layer, n_input, args.top_k,
                )
            except Exception as e:
                log.warning(f"  Failed at t={target_ts:.2f}s: {e}")
                continue

            frame_results.append({
                "frame_idx": t_idx,
                "timestamp": round(target_ts, 3),
                "context_timestamps": [round(t, 3) for t in window_ts[:-1]],
                "n_input_frames": n_input,
                "grid_h": result["grid_h"],
                "grid_w": result["grid_w"],
                "patches": result["patches"],
            })

        # Save
        out_path = args.output_dir / f"{video_name}_layer{args.layer}.json"
        out_data = {
            "video_name": video_name,
            "video_path": str(video_path),
            "duration": round(duration, 3),
            "fps": args.fps,
            "context_frames": n_context,
            "layer": args.layer,
            "n_frames_total": len(frame_results),
            "frames": frame_results,
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f)
        log.info(f"  Saved {out_path} ({len(frame_results)} frames)")

    log.info("Done!")


if __name__ == "__main__":
    main()
