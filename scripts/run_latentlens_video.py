#!/usr/bin/env python3
"""Run LatentLens on VideoLLMs with multi-frame video input.

Feeds multiple frames from a video through a VideoLLM, extracts hidden states
at the MIDDLE frame's visual token positions, and searches them against a
contextual embedding index. Output format is identical to run_latentlens.py
for downstream compatibility (spatial extraction, judge, POS analysis).

Only supports video-capable models: Molmo2-8B, Qwen2.5-VL-7B.

Usage:
    python scripts/run_latentlens_video.py \
        --model allenai/Molmo2-8B \
        --index McGill-NLP/contextual_embeddings-molmo2-8b \
        --videos-dir data/molmo2cap_videos_500/ \
        --frames-dir data/molmo2cap_frames_500/ \
        --output-dir results/molmo2cap_videos_500_molmo2/ \
        --n-frames 4 \
        --device cuda:0
"""

import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Limit CPU threads so multiple jobs can coexist
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import torch
torch.set_num_threads(8)
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import latentlens
from latentlens import ContextualIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction from video
# ---------------------------------------------------------------------------

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def extract_frames(video_path: Path, n_frames: int = 4) -> List[Image.Image]:
    """Extract N uniformly-spaced frames from a video.

    The middle frame (index n_frames // 2) is forced to land at exactly
    duration / 2, matching the midpoint used by download_molmo2cap_frames.py
    for the RQ2 single-frame baseline.
    """
    duration = get_video_duration(video_path)
    middle_idx = n_frames // 2

    # Uniform timestamps with forced midpoint
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    timestamps[middle_idx] = duration / 2

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            frame_path = Path(tmpdir) / f"frame_{i:03d}.jpg"
            cmd = [
                "ffmpeg", "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1", "-q:v", "2",
                "-y", str(frame_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if frame_path.exists():
                frames.append(Image.open(frame_path).convert("RGB").copy())
            else:
                log.warning(f"Failed to extract frame {i} at t={ts:.2f}s from {video_path.name}")
                return []

    return frames


# ---------------------------------------------------------------------------
# Model-specific video input preparation
# ---------------------------------------------------------------------------

def prepare_inputs_video(
    processor, frames: List[Image.Image], model_name: str,
    device: torch.device, model_dtype: torch.dtype
) -> dict:
    """Prepare multi-frame input for a VideoLLM."""
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        # Molmo2: multiple image entries in chat content
        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": "Describe this video."})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=frames, text=text, return_tensors="pt")

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        # Qwen2.5-VL: native video content type
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": "Describe this video."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(videos=[frames], text=[text], return_tensors="pt",
                          padding=True, max_pixels=1024 * 28 * 28)

    else:
        raise ValueError(f"Unsupported video model: {model_name}")

    # Move to device and cast
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
            inputs[k] = v
    return inputs


def get_visual_token_mask(inputs: dict, model_name: str) -> torch.Tensor:
    """Return boolean mask marking ALL visual token positions (all frames)."""
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        return inputs["token_type_ids"][0].bool()

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        # Video uses token ID 151656 (video_pad), not 151655 (image_pad)
        return (inputs["input_ids"][0] == 151656).bool()

    else:
        raise ValueError(f"Unsupported video model: {model_name}")


def get_middle_frame_token_range(
    inputs: dict, model_name: str, n_frames: int, middle_idx: int
) -> Tuple[int, int, int, int]:
    """Get the token range for the middle frame within the visual token sequence.

    Returns (start_idx, end_idx, grid_h, grid_w) where start/end are indices
    into the visual-tokens-only array (not full sequence).
    """
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        # image_grids has shape [N_frames, 4] giving [h1, w1, h2, w2] per frame
        # (multi-crop: global crop is h1*w1, but total tokens per frame includes
        # all crops). We determine tokens_per_frame from total visual tokens / N.
        image_grids = inputs["image_grids"]
        grids = image_grids.tolist()

        # Total visual tokens across all frames
        visual_mask = inputs["token_type_ids"][0].bool()
        n_visual_total = visual_mask.sum().item()
        tokens_per_frame = n_visual_total // n_frames

        start = middle_idx * tokens_per_frame
        end = start + tokens_per_frame

        # Global crop grid dimensions from first two values of image_grids
        grid_h = int(grids[middle_idx][0])
        grid_w = int(grids[middle_idx][1])
        return start, end, grid_h, grid_w

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        # Video uses video_grid_thw (not image_grid_thw)
        # Shape [1, 3] giving (t, h, w) where t is the number of temporal steps
        # after the processor's temporal compression (t <= n_frames).
        grid_thw = inputs["video_grid_thw"]
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        t, h, w = grid_thw[0].tolist()
        t, h, w = int(t), int(h), int(w)
        merge_size = 2
        merged_h = h // merge_size
        merged_w = w // merge_size
        tokens_per_step = merged_h * merged_w

        # The processor compresses n_frames → t temporal steps.
        # Map the middle frame index to the corresponding temporal step.
        # With uniform compression: temporal_step = middle_idx * t // n_frames
        temporal_step = middle_idx * t // n_frames

        start = temporal_step * tokens_per_step
        end = start + tokens_per_step
        return start, end, merged_h, merged_w

    else:
        raise ValueError(f"Unsupported video model: {model_name}")


# ---------------------------------------------------------------------------
# Main LatentLens pipeline for video
# ---------------------------------------------------------------------------

def run_latentlens_on_video(
    model,
    processor,
    index: ContextualIndex,
    frames: List[Image.Image],
    model_name: str,
    device: torch.device,
    layers: List[int],
    n_frames: int,
    top_k: int = 5,
) -> Dict[int, Tuple[List[dict], int, int]]:
    """Run LatentLens on a multi-frame video input, extracting middle frame only.

    Returns {layer: (patches, grid_h, grid_w)} where patches are for the middle
    frame only.
    """
    middle_idx = n_frames // 2
    model_dtype = next(p.dtype for p in model.parameters())
    inputs = prepare_inputs_video(processor, frames, model_name, device, model_dtype)

    visual_mask = get_visual_token_mask(inputs, model_name)
    start, end, grid_h, grid_w = get_middle_frame_token_range(
        inputs, model_name, n_frames, middle_idx
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    results_by_layer = {}

    for layer in layers:
        hs = hidden_states[layer][0]  # [seq_len, hidden_dim]
        all_visual_features = hs[visual_mask]  # [n_visual_total, hidden_dim]

        # Extract only the middle frame's visual tokens
        mid_features = all_visual_features[start:end]  # [n_mid, hidden_dim]
        mid_features = F.normalize(mid_features.float(), dim=-1)

        # Search against contextual index
        neighbors_per_token = index.search(mid_features.to(index.device), top_k=top_k)

        # Build patch data (same format as run_latentlens.py)
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

        results_by_layer[layer] = (patches, grid_h, grid_w)

    del hidden_states, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results_by_layer


# ---------------------------------------------------------------------------
# Validation: verify frame-to-token mapping
# ---------------------------------------------------------------------------

def validate_token_mapping(processor, frames, model_name, device, model_dtype):
    """Validate that middle-frame token isolation is correct.

    Checks that the middle-frame slice has the expected number of tokens
    and that the grid dimensions are consistent.
    """
    n_frames = len(frames)
    middle_idx = n_frames // 2

    # Multi frame
    multi_inputs = prepare_inputs_video(processor, frames, model_name, device, model_dtype)
    multi_mask = get_visual_token_mask(multi_inputs, model_name)
    n_multi_total = multi_mask.sum().item()
    start, end, grid_h, grid_w = get_middle_frame_token_range(
        multi_inputs, model_name, n_frames, middle_idx
    )
    n_mid = end - start

    log.info(f"Validation for {model_name}:")
    log.info(f"  Multi-frame total visual tokens: {n_multi_total}")
    log.info(f"  Middle-frame slice: [{start}:{end}] = {n_mid} tokens")
    log.info(f"  Middle-frame grid: {grid_h}x{grid_w} = {grid_h * grid_w}")
    log.info(f"  Slice within bounds: {start >= 0 and end <= n_multi_total}")

    if end > n_multi_total:
        log.error(f"  OUT OF BOUNDS: end {end} > total {n_multi_total}")
        return False
    if n_mid != grid_h * grid_w:
        # For Molmo2, n_mid includes all crops while grid is global-crop only.
        # This is expected — extract_spatial_patches.py will trim to global crop.
        log.info(f"  Note: token count {n_mid} != grid {grid_h}x{grid_w} = {grid_h*grid_w} "
                 f"(multi-crop — spatial extraction will handle this)")

    log.info(f"  OK: validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run LatentLens on VideoLLMs with multi-frame input"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace VLM model ID (Molmo2-8B or Qwen2.5-VL-7B)")
    parser.add_argument("--index", type=str, required=True,
                        help="HuggingFace repo ID or local path for contextual index")
    parser.add_argument("--videos-dir", type=Path, required=True,
                        help="Directory containing MP4 video files")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="Directory containing single-frame JPEGs (for image_path in output)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for JSON results")
    parser.add_argument("--num-videos", type=int, default=500,
                        help="Number of videos to process")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of frames to sample per video")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--index-device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: auto from index)")
    parser.add_argument("--index-layers", type=str, default=None,
                        help="Comma-separated contextual index layers to load")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true",
                        help="Run token mapping validation on first video and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    index_device = torch.device(args.index_device) if args.index_device else device
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load model and processor
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

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = index.available_layers
    log.info(f"Extracting from model layers: {layers}")

    # Collect videos and corresponding frame filenames
    video_paths = sorted(
        p for p in args.videos_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mkv", ".webm")
    )[:args.num_videos]
    frame_paths = sorted(
        p for p in args.frames_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )[:args.num_videos]
    log.info(f"Found {len(video_paths)} videos, {len(frame_paths)} frames")

    if len(video_paths) != len(frame_paths):
        log.warning(f"Video/frame count mismatch: {len(video_paths)} videos vs {len(frame_paths)} frames")
    n_process = min(len(video_paths), len(frame_paths))

    # Validation mode
    if args.validate_only:
        log.info("Running validation on first video...")
        frames = extract_frames(video_paths[0], args.n_frames)
        if not frames:
            log.error("Failed to extract frames from first video")
            return
        model_dtype = next(p.dtype for p in model.parameters())
        ok = validate_token_mapping(processor, frames, args.model, device, model_dtype)
        if ok:
            log.info("Validation PASSED")
        else:
            log.error("Validation FAILED")
        return

    # Run LatentLens on each video
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_results: Dict[int, List[dict]] = {l: [] for l in layers}

    for vid_idx in tqdm(range(n_process), desc="Processing videos"):
        video_path = video_paths[vid_idx]
        frame_name = frame_paths[vid_idx].name

        try:
            frames = extract_frames(video_path, args.n_frames)
        except Exception as e:
            log.warning(f"Skipping {video_path.name}: frame extraction failed: {e}")
            continue

        if len(frames) != args.n_frames:
            log.warning(f"Skipping {video_path.name}: got {len(frames)} frames, expected {args.n_frames}")
            continue

        try:
            results = run_latentlens_on_video(
                model, processor, index, frames, args.model, device,
                layers, args.n_frames, args.top_k
            )
        except Exception as e:
            log.warning(f"Skipping {video_path.name}: LatentLens failed: {e}")
            continue

        for layer, (patches, grid_h, grid_w) in results.items():
            layer_results[layer].append({
                "image_idx": len(layer_results[layer]),
                "image_path": frame_name,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "n_frames": args.n_frames,
                "patches": patches,
            })

    # Save per-layer JSON files
    for layer, results in layer_results.items():
        out_path = args.output_dir / f"latentlens_layer{layer}.json"
        with open(out_path, "w") as f:
            json.dump({"results": results}, f)
        log.info(f"Saved {out_path} ({len(results)} videos)")

    log.info("Done!")


if __name__ == "__main__":
    main()
