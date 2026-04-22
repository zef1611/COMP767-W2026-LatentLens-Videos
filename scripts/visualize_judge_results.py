#!/usr/bin/env python3
"""Generate per-patch visualization PNGs from judge evaluation results.

For each evaluated patch, creates a PNG showing:
- Image with red bounding box
- Cropped region
- Top-5 candidate words with judge classifications
- Interpretable/not verdict

Usage:
    python scripts/visualize_judge_results.py \
        --model "Molmo2-8B" \
        --output-dir visualizations/molmo2
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path("results")
IMAGES_DIR = Path("data/pixmo_cap_100")
BBOX_SIZE = 3


def resize_and_pad_pil(pil_img, size=512):
    w, h = pil_img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return padded


MODEL_RESULTS_DIRS = {
    "Molmo-7B-D": "pixmo100_molmo-7b-d",
    "Idefics3-8B": "pixmo100_idefics3",
    "Molmo2-8B": "pixmo100_molmo2",
    "Qwen2.5-VL-7B": "pixmo100_qwen25vl",
}


def get_grid_for_image(results_dir, layer, img_idx):
    """Get grid dimensions for a specific image from LatentLens results."""
    path = RESULTS_DIR / results_dir / f"latentlens_layer{layer}.json"
    with open(path) as f:
        data = json.load(f)
    img = data["results"][img_idx]
    patches = img["patches"]
    grid_h = max(p["patch_row"] for p in patches) + 1
    grid_w = max(p["patch_col"] for p in patches) + 1
    return grid_h, grid_w


def render_patch_viz(image_path, patch_row, patch_col, grid_h, grid_w,
                     candidates, gemini_response, layer, img_idx):
    """Render a single patch visualization. Returns PIL Image."""
    # Load and preprocess image
    processed = resize_and_pad_pil(Image.open(image_path).convert("RGB"))

    patch_h = 512 / grid_h
    patch_w = 512 / grid_w
    left = max(0, patch_col * patch_w)
    top = max(0, patch_row * patch_h)
    right = min(512, (patch_col + BBOX_SIZE) * patch_w)
    bottom = min(512, (patch_row + BBOX_SIZE) * patch_h)

    # Image with bbox
    img_bbox = processed.copy()
    ImageDraw.Draw(img_bbox).rectangle([left, top, right, bottom], outline="red", width=3)

    # Crop (scaled up for visibility)
    crop = processed.crop((int(left), int(top), int(right), int(bottom)))
    crop_display = crop.resize((150, 150), Image.LANCZOS)

    # Build the visualization canvas
    canvas_w = 512 + 20 + 350  # image + gap + text panel
    canvas_h = 512
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(img_bbox, (0, 0))
    canvas.paste(crop_display, (522, 10))

    draw = ImageDraw.Draw(canvas)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
        font_small = font

    # Verdict
    is_interp = gemini_response.get("interpretable", False)
    verdict_color = (0, 128, 0) if is_interp else (200, 0, 0)
    verdict_text = "INTERPRETABLE" if is_interp else "NOT INTERPRETABLE"
    draw.text((522, 170), verdict_text, fill=verdict_color, font=font_bold)

    # Info line
    draw.text((522, 190), f"Layer {layer} | Img {img_idx} | Patch [{patch_row},{patch_col}]",
              fill=(100, 100, 100), font=font_small)
    draw.text((522, 205), f"Grid {grid_h}x{grid_w}",
              fill=(100, 100, 100), font=font_small)

    # Candidates with classifications
    concrete = set(gemini_response.get("concrete_words", []))
    abstract = set(gemini_response.get("abstract_words", []))
    global_w = set(gemini_response.get("global_words", []))

    y = 230
    draw.text((522, y), "Candidates:", fill=(0, 0, 0), font=font_bold)
    y += 20
    for i, word in enumerate(candidates):
        if word in concrete:
            label = "[concrete]"
            color = (0, 128, 0)
        elif word in abstract:
            label = "[abstract]"
            color = (0, 0, 180)
        elif word in global_w:
            label = "[global]"
            color = (180, 120, 0)
        else:
            label = ""
            color = (150, 150, 150)
        line = f"{i+1}. \"{word}\" {label}"
        draw.text((522, y), line, fill=color, font=font)
        y += 18

    # Reasoning (truncated)
    reasoning = gemini_response.get("reasoning", "")
    if reasoning:
        y += 10
        draw.text((522, y), "Reasoning:", fill=(0, 0, 0), font=font_bold)
        y += 18
        # Word wrap
        words = reasoning.split()
        line = ""
        for w in words:
            test = line + " " + w if line else w
            if len(test) > 42:
                draw.text((522, y), line, fill=(80, 80, 80), font=font_small)
                y += 14
                line = w
                if y > 500:
                    draw.text((522, y), "...", fill=(80, 80, 80), font=font_small)
                    break
            else:
                line = test
        else:
            if line:
                draw.text((522, y), line, fill=(80, 80, 80), font=font_small)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_RESULTS_DIRS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--judge-results", type=str, default="results/judge_evaluation.json")
    args = parser.parse_args()

    with open(args.judge_results) as f:
        all_results = json.load(f)

    model_data = all_results.get(args.model)
    if not model_data:
        print(f"No results for {args.model}")
        sys.exit(1)

    results_subdir = MODEL_RESULTS_DIRS[args.model]
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    out_root = Path(args.output_dir)
    total = 0

    for layer_key in sorted(model_data, key=int):
        layer = int(layer_key)
        layer_data = model_data[layer_key]
        patches = layer_data.get("patches", [])

        layer_dir = out_root / f"layer{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        grid_h, grid_w = get_grid_for_image(results_subdir, layer, 0)

        for patch_result in patches:
            img_idx = patch_result.get("image_idx")
            if img_idx is None or img_idx >= len(image_paths):
                continue
            if "error" in patch_result:
                continue

            row = patch_result["patch_row"]
            col = patch_result["patch_col"]
            candidates = patch_result.get("candidates", [])
            resp = patch_result.get("gemini_response", {})

            # Get per-image grid if it varies
            try:
                grid_h, grid_w = get_grid_for_image(results_subdir, layer, img_idx)
            except (IndexError, KeyError):
                pass

            viz = render_patch_viz(
                image_paths[img_idx], row, col, grid_h, grid_w,
                candidates, resp, layer, img_idx
            )

            is_interp = "yes" if resp.get("interpretable") else "no"
            out_path = layer_dir / f"img{img_idx:03d}_r{row}_c{col}_{is_interp}.png"
            viz.save(out_path)
            total += 1

        n_interp = layer_data.get("n_interpretable", 0)
        n_total = layer_data.get("n_patches", 0)
        print(f"  Layer {layer}: {len(patches)} patches saved ({n_interp}/{n_total} interpretable)")

    print(f"\nDone! {total} visualizations saved to {out_root}")


if __name__ == "__main__":
    main()
