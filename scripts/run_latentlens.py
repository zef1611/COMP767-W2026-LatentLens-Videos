#!/usr/bin/env python3
"""Run LatentLens on a VLM with static images.

Feeds images through a VLM, extracts hidden states at visual token positions,
and searches them against a contextual embedding index. Outputs JSON files
compatible with the demo viewer.

Usage:
    python scripts/run_latentlens.py \
        --model allenai/Molmo2-8B \
        --index McGill-NLP/contextual_embeddings-molmo2-8b \
        --images-dir data/images/ \
        --output-dir results/molmo2/ \
        --num-images 100 \
        --device cuda:0
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import latentlens
from latentlens import ContextualIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-specific helpers for extracting visual token positions
# ---------------------------------------------------------------------------

def get_visual_token_mask(inputs: dict, model_name: str) -> torch.Tensor:
    """Return a boolean mask of shape [seq_len] marking visual token positions.

    Handles different VLM architectures:
    - Molmo2: uses token_type_ids (True = image token)
    - Molmo-7B-D: uses image_input_idx (>= 0 = valid image token position)
    - Idefics3: image_token_id = 128257
    - Qwen2.5-VL: image_token_id = 151655
    """
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        return inputs["token_type_ids"][0].bool()

    elif "molmo" in model_lower:
        image_input_idx = inputs["image_input_idx"]
        seq_len = inputs["input_ids"].shape[1]
        mask = torch.zeros(seq_len, dtype=torch.bool)
        valid = image_input_idx[image_input_idx >= 0].long()
        mask[valid] = True
        return mask

    elif "idefics3" in model_lower:
        return (inputs["input_ids"][0] == 128257).bool()

    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        return (inputs["input_ids"][0] == 151655).bool()

    else:
        raise ValueError(
            f"Unsupported model {model_name}. Add visual token extraction logic."
        )


def prepare_inputs(
    processor, image: Image.Image, model_name: str, device: torch.device, model_dtype: torch.dtype
) -> dict:
    """Process an image through the model's processor and prepare inputs."""
    model_lower = model_name.lower()

    if "molmo2" in model_lower:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=[image], text=text, return_tensors="pt")
    elif "molmo" in model_lower:
        inputs = processor.process(images=[image], text="Describe this image.")
        inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    elif "idefics3" in model_lower:
        text = "User:<image>Describe this image.\nAssistant:"
        inputs = processor(text=text, images=[image], return_tensors="pt")
    elif "qwen2.5-vl" in model_lower or "qwen2_5_vl" in model_lower:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Constrain resolution to avoid OOM on large images (max ~1000 visual tokens)
        inputs = processor(images=[image], text=[text], return_tensors="pt", padding=True,
                          max_pixels=1024 * 28 * 28)
    else:
        raise ValueError(f"Unsupported model {model_name}. Add processor logic.")

    # Move to device and cast float tensors to model dtype
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
            inputs[k] = v
    return inputs


def get_grid_size(inputs: dict, model_name: str, visual_mask: torch.Tensor) -> int:
    """Infer the patch grid size from the number of visual tokens.

    Molmo uses multi-crop (e.g. 2 crops × 576 patches = 1152 tokens).
    We return the per-crop grid size (e.g. 24 for 576 patches).
    """
    n_visual = visual_mask.sum().item()

    model_lower = model_name.lower()
    if "molmo" in model_lower and "molmo2" not in model_lower:
        # Original Molmo: image_input_idx has shape [num_crops, patches_per_crop]
        if "image_input_idx" in inputs:
            patches_per_crop = inputs["image_input_idx"].shape[-1]
            grid = int(math.sqrt(patches_per_crop))
            return grid

    grid = int(math.sqrt(n_visual))
    if grid * grid != n_visual:
        log.warning(f"Non-square grid: {n_visual} visual tokens, using grid_size={grid}")
    return grid


# ---------------------------------------------------------------------------
# Main LatentLens pipeline
# ---------------------------------------------------------------------------

def run_latentlens_on_image(
    model,
    processor,
    index: ContextualIndex,
    image: Image.Image,
    model_name: str,
    device: torch.device,
    layers: List[int],
    top_k: int = 5,
) -> Dict[int, List[dict]]:
    """Run LatentLens on a single image.

    Returns {layer: [patch_data, ...]} where each patch_data has:
    - patch_idx, patch_row, patch_col
    - nearest_contextual_neighbors: [{token_str, similarity, caption, contextual_layer, position}]
    """
    model_dtype = next(p.dtype for p in model.parameters())
    inputs = prepare_inputs(processor, image, model_name, device, model_dtype)
    visual_mask = get_visual_token_mask(inputs, model_name)
    grid_size = get_grid_size(inputs, model_name, visual_mask)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of [1, seq_len, hidden_dim]

    results_by_layer = {}

    for layer in layers:
        # Extract visual token features at this layer
        hs = hidden_states[layer][0]  # [seq_len, hidden_dim]
        visual_features = hs[visual_mask]  # [n_visual, hidden_dim]
        visual_features = F.normalize(visual_features.float(), dim=-1)

        # Search against contextual index (move to index device if different)
        neighbors_per_token = index.search(visual_features.to(index.device), top_k=top_k)

        # Build patch data
        patches = []
        for patch_idx, neighbors in enumerate(neighbors_per_token):
            row = patch_idx // grid_size
            col = patch_idx % grid_size
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

        results_by_layer[layer] = patches

    del hidden_states, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results_by_layer


def main():
    parser = argparse.ArgumentParser(description="Run LatentLens on a VLM with static images")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace VLM model ID")
    parser.add_argument("--index", type=str, required=True,
                        help="HuggingFace repo ID or local path for contextual index")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory containing images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for JSON results")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to process")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for the model")
    parser.add_argument("--index-device", type=str, default=None,
                        help="Device for the index (default: same as --device)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: auto from index)")
    parser.add_argument("--index-layers", type=str, default=None,
                        help="Comma-separated contextual index layers to load (default: all)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    index_device = torch.device(args.index_device) if args.index_device else device
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load model and processor
    log.info(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Try loading in order of specificity
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

    # Load contextual index for cross-layer search
    log.info(f"Loading contextual index: {args.index}")
    index_layers = [int(x) for x in args.index_layers.split(",")] if args.index_layers else None
    index_path = Path(args.index)
    if index_path.exists():
        index = ContextualIndex.from_directory(str(index_path), layers=index_layers)
    else:
        index = ContextualIndex.from_pretrained(args.index, layers=index_layers)
    index = index.to(index_device)
    log.info(f"Index layers: {index.available_layers}")

    # Determine which MODEL layers to extract hidden states from
    # (independent of which layers the index has)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = index.available_layers
    log.info(f"Extracting from model layers: {layers}")

    # Collect images
    image_paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )[:args.num_images]
    log.info(f"Processing {len(image_paths)} images from {args.images_dir}")

    # Run LatentLens on each image
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate results per layer
    layer_results: Dict[int, List[dict]] = {l: [] for l in layers}

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"Skipping {img_path.name}: {e}")
            continue
        results = run_latentlens_on_image(
            model, processor, index, image, args.model, device, layers, args.top_k
        )
        for layer, patches in results.items():
            layer_results[layer].append({
                "image_idx": len(layer_results[layer]),
                "image_path": str(img_path.name),
                "patches": patches,
            })

    # Save per-layer JSON files
    for layer, results in layer_results.items():
        out_path = args.output_dir / f"latentlens_layer{layer}.json"
        with open(out_path, "w") as f:
            json.dump({"results": results}, f)
        log.info(f"Saved {out_path} ({len(results)} images)")

    log.info("Done!")


if __name__ == "__main__":
    main()
