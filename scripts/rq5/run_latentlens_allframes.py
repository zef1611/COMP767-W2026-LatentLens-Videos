#!/usr/bin/env python3
"""Run LatentLens on VideoLLMs extracting ALL frames' visual tokens.

Unlike run_latentlens_video.py (which extracts only the middle frame),
this script extracts visual tokens for every frame fed to the model,
with inline spatial correction (global crop only for Molmo2, correct
grid dims for Qwen2.5-VL).

Output format has per-frame patch lists instead of a flat patch list.

Usage:
    python scripts/rq5/run_latentlens_allframes.py \\
        --model allenai/Molmo2-8B \\
        --index McGill-NLP/contextual_embeddings-molmo2-8b \\
        --videos-dir $SCRATCH/latentlens/pvsg_videos_100/ \\
        --frames-dir $SCRATCH/latentlens/pvsg_frames_100/ \\
        --output-dir results/pvsg_100_molmo2_4f_allframes/ \\
        --n-frames 4 --layers 24 --device cuda:0
"""

import argparse
import json
import logging
import os
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
    extract_frames,
    get_video_duration,
    prepare_inputs_video,
    get_visual_token_mask,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_all_frame_token_ranges(
    inputs: dict, model_name: str, n_frames: int
) -> List[Tuple[int, int, int, int, List[int]]]:
    """Get token ranges for ALL frames (or temporal steps for Qwen).

    Returns list of (start, end, grid_h, grid_w, input_frame_indices) tuples.
    For Molmo2: one entry per input frame, start/end covers global crop only.
    For Qwen: one entry per temporal step (may merge multiple input frames).
    """
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        image_grids = inputs["image_grids"]
        grids = image_grids.tolist()

        visual_mask = inputs["token_type_ids"][0].bool()
        n_visual_total = visual_mask.sum().item()
        tokens_per_frame = n_visual_total // n_frames

        ranges = []
        for frame_idx in range(n_frames):
            frame_start = frame_idx * tokens_per_frame
            grid_h = int(grids[frame_idx][0])
            grid_w = int(grids[frame_idx][1])
            # Global crop = first grid_h * grid_w tokens of this frame's allocation
            n_global = grid_h * grid_w
            if n_global > tokens_per_frame:
                log.warning(f"Frame {frame_idx}: global crop {n_global} > "
                           f"tokens_per_frame {tokens_per_frame}, using all tokens")
                n_global = tokens_per_frame
            ranges.append((frame_start, frame_start + n_global,
                          grid_h, grid_w, [frame_idx]))
        return ranges

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        grid_thw = inputs["video_grid_thw"]
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        t, h, w = grid_thw[0].tolist()
        t, h, w = int(t), int(h), int(w)
        merge_size = 2
        merged_h = h // merge_size
        merged_w = w // merge_size
        tokens_per_step = merged_h * merged_w

        if t < n_frames:
            log.warning(f"Qwen temporal compression: {n_frames} input frames → "
                       f"{t} temporal steps")

        ranges = []
        for step in range(t):
            start = step * tokens_per_step
            end = start + tokens_per_step
            # Which input frames map to this temporal step
            input_frames = [i for i in range(n_frames) if i * t // n_frames == step]
            ranges.append((start, end, merged_h, merged_w, input_frames))
        return ranges

    else:
        raise ValueError(f"Unsupported video model: {model_name}")


def run_latentlens_allframes(
    model,
    processor,
    index: ContextualIndex,
    frames: List[Image.Image],
    model_name: str,
    device: torch.device,
    layers: List[int],
    n_frames: int,
    top_k: int = 5,
) -> Dict[int, List[dict]]:
    """Run LatentLens on ALL frames of a multi-frame video input.

    Returns {layer: [frame_data, ...]} where each frame_data has
    {frame_idx, grid_h, grid_w, input_frames, patches}.
    """
    model_dtype = next(p.dtype for p in model.parameters())
    inputs = prepare_inputs_video(processor, frames, model_name, device, model_dtype)

    visual_mask = get_visual_token_mask(inputs, model_name)
    frame_ranges = get_all_frame_token_ranges(inputs, model_name, n_frames)

    # Forward pass (same as single-frame — the expensive part)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    results_by_layer = {}

    for layer in layers:
        hs = hidden_states[layer][0]  # [seq_len, hidden_dim]
        all_visual_features = hs[visual_mask]  # [n_visual_total, hidden_dim]

        frame_data_list = []
        for range_idx, (start, end, grid_h, grid_w, input_frames) in enumerate(frame_ranges):
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

            frame_data_list.append({
                "frame_idx": range_idx,
                "input_frames": input_frames,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "patches": patches,
            })

        results_by_layer[layer] = frame_data_list

    del hidden_states, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results_by_layer


def compute_timestamps(video_path: Path, n_frames: int) -> List[float]:
    """Compute frame extraction timestamps (same formula as extract_frames)."""
    duration = get_video_duration(video_path)
    middle_idx = n_frames // 2
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    timestamps[middle_idx] = duration / 2
    return timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Run LatentLens on VideoLLMs — extract ALL frames"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--index-device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--layers", type=str, default="24",
                        help="Comma-separated layer indices")
    parser.add_argument("--index-layers", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    index_device = torch.device(args.index_device) if args.index_device else device
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

    # Load contextual index
    log.info(f"Loading contextual index: {args.index}")
    index_layers = [int(x) for x in args.index_layers.split(",")] if args.index_layers else None
    index_path = Path(args.index)
    if index_path.exists():
        index = ContextualIndex.from_directory(str(index_path), layers=index_layers)
    else:
        index = ContextualIndex.from_pretrained(args.index, layers=index_layers)
    index = index.to(index_device)
    log.info(f"Index layers: {index.available_layers}")

    layers = [int(x) for x in args.layers.split(",")]
    log.info(f"Extracting from model layers: {layers}")

    # Collect videos
    video_paths = sorted(
        p for p in args.videos_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mkv", ".webm")
    )[:args.num_videos]
    frame_paths = sorted(
        p for p in args.frames_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )[:args.num_videos]
    log.info(f"Found {len(video_paths)} videos, {len(frame_paths)} frames")

    n_process = min(len(video_paths), len(frame_paths))

    # Run
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_results: Dict[int, List[dict]] = {l: [] for l in layers}

    for vid_idx in tqdm(range(n_process), desc="Processing videos"):
        video_path = video_paths[vid_idx]
        frame_name = frame_paths[vid_idx].name
        video_name = video_path.stem

        try:
            frames = extract_frames(video_path, args.n_frames)
        except Exception as e:
            log.warning(f"Skipping {video_path.name}: frame extraction failed: {e}")
            continue

        if len(frames) != args.n_frames:
            log.warning(f"Skipping {video_path.name}: got {len(frames)} frames")
            continue

        timestamps = compute_timestamps(video_path, args.n_frames)

        try:
            results = run_latentlens_allframes(
                model, processor, index, frames, args.model, device,
                layers, args.n_frames, args.top_k
            )
        except Exception as e:
            log.warning(f"Skipping {video_path.name}: LatentLens failed: {e}")
            continue

        for layer, frame_data_list in results.items():
            layer_results[layer].append({
                "image_idx": len(layer_results[layer]),
                "image_path": frame_name,
                "video_name": video_name,
                "n_frames": args.n_frames,
                "frame_timestamps": timestamps,
                "frames": frame_data_list,
            })

    # Save
    for layer, results in layer_results.items():
        out_path = args.output_dir / f"latentlens_layer{layer}.json"
        with open(out_path, "w") as f:
            json.dump({"results": results}, f)
        log.info(f"Saved {out_path} ({len(results)} videos)")

    log.info("Done!")


if __name__ == "__main__":
    main()
