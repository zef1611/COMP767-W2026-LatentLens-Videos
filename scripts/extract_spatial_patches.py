#!/usr/bin/env python3
"""Extract spatially-correct patches from LatentLens results for all models.

For each model, produces a _spatial/ directory where patch row/col maps
correctly to image spatial positions. This is the ONLY source of truth
for anything that draws bounding boxes on images (judge, demo, viz).

Model-specific handling:
- Molmo-7B-D: global crop = first 144 patches (12x12), from multi-crop sequence
- Molmo2-8B: global crop = first N patches (varies per image), from multi-crop sequence
- Idefics3-8B: global image = LAST tile in the sequence, after pixel shuffle
- Qwen2.5-VL-7B: single spatial grid, just needs correct grid_h x grid_w from processor

Usage:
    python scripts/extract_spatial_patches.py \
        --model allenai/Molmo-7B-D-0924 \
        --model-key molmo-7b-d \
        --results-dir results/pixmo100_molmo-7b-d/ \
        --images-dir data/pixmo_cap_100/ \
        --output-dir results/pixmo100_molmo-7b-d_spatial/
"""

import argparse
import json
import logging
import math
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_spatial_info(processor, image: Image.Image, model_key: str) -> dict:
    """Get spatial patch info for the global/primary view of the image.

    Returns dict with:
        mode: "first_n" (take first N patches) or "last_n" (take last N) or "all" (all patches are spatial)
        n_patches: number of patches in the global view
        grid_h, grid_w: spatial grid dimensions
    """
    if "molmo2" in model_key:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=[image], text=text, return_tensors="pt")
        grids = inputs["image_grids"][0].tolist()
        h, w = int(grids[0]), int(grids[1])
        return {"mode": "first_n", "n_patches": h * w, "grid_h": h, "grid_w": w}

    elif "molmo" in model_key:
        inputs = processor.process(images=[image], text="Describe.")
        patches_per_crop = inputs["image_input_idx"].shape[1]
        g = int(math.sqrt(patches_per_crop))
        return {"mode": "first_n", "n_patches": patches_per_crop, "grid_h": g, "grid_w": g}

    elif "idefics3" in model_key:
        text = "User:<image>Describe this image.\nAssistant:"
        inputs = processor(text=text, images=[image], return_tensors="pt")
        pv = inputs.get("pixel_values")
        # pixel_values shape: [batch, n_tiles, channels, h, w]
        n_tiles = pv.shape[1]  # includes tiles + 1 global image
        tokens_per_tile = 169  # (364/14)^2 / scale_factor^2 = 26^2/4
        total_tokens = n_tiles * tokens_per_tile
        g = int(math.sqrt(tokens_per_tile))  # 13x13
        return {"mode": "last_n", "n_patches": tokens_per_tile, "grid_h": g, "grid_w": g,
                "total_tokens": total_tokens}

    elif "qwen" in model_key:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=[text], return_tensors="pt",
                          padding=True, max_pixels=1024 * 28 * 28)
        grid_thw = inputs["image_grid_thw"][0].tolist()
        # Qwen2.5-VL uses a 2x2 merge: visual tokens are grid_h/2 x grid_w/2
        merge_size = 2
        h = int(grid_thw[1]) // merge_size
        w = int(grid_thw[2]) // merge_size
        return {"mode": "all", "n_patches": h * w, "grid_h": h, "grid_w": w}

    raise ValueError(f"Unsupported model_key: {model_key}")


def extract_spatial(img_data, info):
    """Extract spatially-correct patches from raw LatentLens results."""
    mode = info["mode"]
    n = info["n_patches"]
    gh, gw = info["grid_h"], info["grid_w"]
    patches = img_data["patches"]

    if mode == "first_n":
        # Global crop is first N patches
        selected = [p for p in patches if p["patch_idx"] < n]
        for p in selected:
            p["patch_row"] = p["patch_idx"] // gw
            p["patch_col"] = p["patch_idx"] % gw

    elif mode == "last_n":
        # Global image is last N patches (Idefics3)
        total = info.get("total_tokens", len(patches))
        start_idx = total - n
        selected = [p for p in patches if p["patch_idx"] >= start_idx]
        for p in selected:
            local_idx = p["patch_idx"] - start_idx
            p["patch_row"] = local_idx // gw
            p["patch_col"] = local_idx % gw

    elif mode == "all":
        # All patches are spatial, just fix row/col with correct grid dims
        selected = list(patches)
        for p in selected:
            p["patch_row"] = p["patch_idx"] // gw
            p["patch_col"] = p["patch_idx"] % gw

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "image_idx": img_data.get("image_idx", 0),
        "image_path": img_data.get("image_path", ""),
        "grid_h": gh,
        "grid_w": gw,
        "patches": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID for loading processor")
    parser.add_argument("--model-key", type=str, required=True,
                        help="Short key: molmo-7b-d, idefics3, molmo2, qwen25vl")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=100)
    args = parser.parse_args()

    log.info(f"Loading processor for {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    all_image_paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )[:args.num_images]
    log.info(f"Found {len(all_image_paths)} image files")

    # Filter to only images that can be opened (matching what run_latentlens.py does)
    image_paths = []
    for img_path in all_image_paths:
        try:
            Image.open(img_path).verify()
            image_paths.append(img_path)
        except Exception:
            log.warning(f"Skipping unreadable image: {img_path.name}")
    log.info(f"Processing {len(image_paths)} valid images")

    # Get spatial info for each valid image
    # If results contain grid_h/grid_w (e.g. from run_latentlens_video.py),
    # use those directly instead of recomputing from the processor.
    log.info("Computing spatial patch info per image...")
    spatial_infos = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        info = get_spatial_info(processor, img, args.model_key)
        spatial_infos.append(info)
    log.info(f"  Example: img0 = {spatial_infos[0]}")

    # Process each layer file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_files = sorted(args.results_dir.glob("latentlens_layer*.json"))
    log.info(f"Processing {len(layer_files)} layer files...")

    for lf in layer_files:
        with open(lf) as f:
            data = json.load(f)

        new_results = []
        n_results = len(data["results"])
        if n_results != len(spatial_infos):
            log.warning(f"Results have {n_results} images but spatial_infos has {len(spatial_infos)}; using min")
        for img_idx in range(min(n_results, len(spatial_infos))):
            img_data = data["results"][img_idx]
            info = spatial_infos[img_idx]
            # Override grid from JSON if present (e.g. multi-frame video results)
            if "grid_h" in img_data and "grid_w" in img_data:
                info = dict(info)
                info["grid_h"] = img_data["grid_h"]
                info["grid_w"] = img_data["grid_w"]
                info["n_patches"] = img_data["grid_h"] * img_data["grid_w"]
            result = extract_spatial(img_data, info)
            result["image_idx"] = img_idx
            new_results.append(result)

        out_path = args.output_dir / lf.name
        with open(out_path, "w") as f:
            json.dump({"results": new_results}, f)
        log.info(f"  {lf.name}: {len(new_results)} images, "
                 f"~{len(new_results[0]['patches'])} patches/img")

    log.info("Done!")


if __name__ == "__main__":
    main()
