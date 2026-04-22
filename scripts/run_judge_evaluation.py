#!/usr/bin/env python3
"""Run Gemini LLM judge evaluation across models and layers.

Follows the protocol from krojer2026latentlens:
- 100 patches (1 per image, 100 images)
- Layers: {0, 1, 4, 8, 16, 24, L-2, L-1}
- Patch = 3x3 bounding box
- Judge classifies top-5 LatentLens neighbors as concrete/abstract/global
- Metric: % interpretable (at least one word classified)

Key invariant: each image gets ONE normalized (x, y) coordinate, sampled
once and reused across all models and all layers. This ensures we compare
the same image region everywhere.

Usage:
    python scripts/run_judge_evaluation.py --api-key-file gemini_key.txt
"""

import argparse
import json
import random
import sys
import time
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                        "vl_embedding_spaces/third_party/molmo/llm_judge"))
from prompts import IMAGE_PROMPT_WITH_CROP


def resize_natural(pil_img, max_side=512):
    """Resize image preserving aspect ratio, no padding."""
    w, h = pil_img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


# ── Config ──────────────────────────────────────────────────────────────────

MODELS = {
    "Molmo-7B-D": {
        "results_dir": "results/pixmo100_molmo-7b-d_spatial",
        "layers": [0, 1, 2, 4, 8, 16, 24, 26, 27],
        "n_layers": 28,
    },
    "Idefics3-8B": {
        "results_dir": "results/pixmo100_idefics3_spatial",
        "layers": [0, 1, 2, 4, 8, 16, 24, 30, 31],
        "n_layers": 32,
    },
    "Molmo2-8B": {
        "results_dir": "results/pixmo100_molmo2_spatial",
        "layers": [0, 1, 2, 4, 8, 16, 24, 34, 35],
        "n_layers": 36,
    },
    "Qwen2.5-VL-7B": {
        "results_dir": "results/pixmo100_qwen25vl_spatial",
        "layers": [0, 1, 2, 4, 8, 16, 24, 26, 27],
        "n_layers": 28,
    },
}

IMAGES_DIR = Path("data/pixmo_cap_100")
NUM_PATCHES = 100  # 1 per image
BBOX_SIZE = 3      # 3x3 patch bounding box
SEED = 42
MAX_WORKERS = 20


def sample_normalized_coordinates(n_images, seed=42):
    """Pre-sample one (norm_x, norm_y) per image, in [0, 1).

    Coordinates are in normalized content space — (0,0) is top-left of
    the image content, (1,1) is bottom-right. Image-independent and
    model-independent. The grid mapping (norm_to_patch) handles
    converting these to model-specific patch positions.
    """
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n_images)]


def norm_to_patch(norm_x, norm_y, grid_h, grid_w):
    """Map normalized (x, y) to the nearest valid patch row, col.

    Ensures a BBOX_SIZE x BBOX_SIZE bbox centered on (row, col) fits within
    the grid, i.e. row in [half, grid_h-1-half], col in [half, grid_w-1-half].
    """
    half = BBOX_SIZE // 2
    row = int(norm_y * grid_h)
    col = int(norm_x * grid_w)
    row = max(half, min(row, grid_h - 1 - half))
    col = max(half, min(col, grid_w - 1 - half))
    return row, col


def get_grid_for_image(img_data):
    """Get grid dimensions from spatial result metadata."""
    if "grid_h" in img_data and "grid_w" in img_data:
        return img_data["grid_h"], img_data["grid_w"]
    patches = img_data.get("patches", [])
    if not patches:
        return 1, 1
    return (max(p["patch_row"] for p in patches) + 1,
            max(p["patch_col"] for p in patches) + 1)


def find_patch_at(img_data, row, col):
    """Find the patch at (row, col), or nearest if exact match doesn't exist."""
    patches = img_data.get("patches", [])
    # Exact match
    for p in patches:
        if p["patch_row"] == row and p["patch_col"] == col:
            return p
    # Nearest (by Manhattan distance)
    best, best_dist = None, float("inf")
    for p in patches:
        d = abs(p["patch_row"] - row) + abs(p["patch_col"] - col)
        if d < best_dist:
            best, best_dist = p, d
    return best


def get_available_layers(results_dir):
    available = []
    for f in Path(results_dir).glob("latentlens_layer*.json"):
        layer = int(f.stem.replace("latentlens_layer", ""))
        available.append(layer)
    return sorted(available)


def load_layer_results(results_dir, layer):
    path = Path(results_dir) / f"latentlens_layer{layer}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def prepare_judge_input(image_path, patch, grid_h, grid_w):
    """Prepare image with bbox + crop + candidate words for the judge.

    Uses the natural (unpadded) image. Box position derived from
    grid proportions: col/grid_w and row/grid_h map to image width/height.
    """
    pil_img = Image.open(image_path).convert("RGB")
    display = resize_natural(pil_img, 512)
    disp_w, disp_h = display.size

    row, col = patch["patch_row"], patch["patch_col"]
    half = BBOX_SIZE // 2

    # Grid proportions -> pixel positions on the natural image (centered on patch)
    left = (col - half) / grid_w * disp_w
    top = (row - half) / grid_h * disp_h
    right = (col - half + BBOX_SIZE) / grid_w * disp_w
    bottom = (row - half + BBOX_SIZE) / grid_h * disp_h

    # Clamp
    left, top = max(0, left), max(0, top)
    right, bottom = min(disp_w, right), min(disp_h, bottom)

    img_with_bbox = display.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

    cropped = display.crop((int(left), int(top), int(right), int(bottom)))

    nbs = patch.get("nearest_contextual_neighbors", [])[:5]
    candidates = [nb.get("token_str", "").strip() for nb in nbs
                  if nb.get("token_str", "").strip()]

    return img_with_bbox, cropped, candidates


def call_gemini(model, img_with_bbox, cropped, prompt):
    """Call Gemini API, return parsed JSON response."""
    parts = [img_with_bbox, cropped, prompt]
    response = model.generate_content(parts)
    text = response.text

    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    return {
        "interpretable": False,
        "concrete_words": [], "abstract_words": [], "global_words": [],
        "reasoning": text
    }


def evaluate_layer(gemini_model, results_dir, layer, image_paths,
                   norm_coords, existing=None):
    """Evaluate 100 patches for one model+layer. Returns list of results."""
    data = load_layer_results(results_dir, layer)
    if data is None:
        print(f"    Layer {layer}: no results file, skipping")
        return []

    images = data["results"]
    n_images = min(NUM_PATCHES, len(images), len(image_paths))

    # Build tasks
    results = list(existing or [])
    tasks = []
    for img_idx in range(n_images):
        if existing and any(r["image_idx"] == img_idx for r in existing):
            continue

        img_data = images[img_idx]
        grid_h, grid_w = get_grid_for_image(img_data)

        # Map normalized coordinate to this model's grid
        norm_x, norm_y = norm_coords[img_idx]
        row, col = norm_to_patch(norm_x, norm_y, grid_h, grid_w)

        patch = find_patch_at(img_data, row, col)
        if patch is None:
            continue

        img_with_bbox, cropped, candidates = prepare_judge_input(
            image_paths[img_idx], patch, grid_h, grid_w
        )

        if not candidates:
            # All neighbors are whitespace — not interpretable by definition
            results.append({
                "image_idx": img_idx,
                "patch_row": patch["patch_row"],
                "patch_col": patch["patch_col"],
                "norm_x": norm_x,
                "norm_y": norm_y,
                "candidates": [],
                "interpretable": False,
                "note": "all-whitespace neighbors",
            })
            continue

        prompt = IMAGE_PROMPT_WITH_CROP.format(
            candidate_words=json.dumps(candidates)
        )

        tasks.append({
            "image_idx": img_idx,
            "patch_row": patch["patch_row"],
            "patch_col": patch["patch_col"],
            "norm_x": norm_x,
            "norm_y": norm_y,
            "candidates": candidates,
            "img_with_bbox": img_with_bbox,
            "cropped": cropped,
            "prompt": prompt,
        })

    if not tasks:
        return results

    def run_task(task):
        for attempt in range(3):
            try:
                resp = call_gemini(
                    gemini_model, task["img_with_bbox"], task["cropped"], task["prompt"]
                )
                is_interp = resp.get("interpretable", False)
                return {
                    "image_idx": task["image_idx"],
                    "patch_row": task["patch_row"],
                    "patch_col": task["patch_col"],
                    "norm_x": task["norm_x"],
                    "norm_y": task["norm_y"],
                    "candidates": task["candidates"],
                    "gemini_response": resp,
                    "interpretable": is_interp,
                }
            except Exception as e:
                if attempt == 2:
                    return {
                        "image_idx": task["image_idx"],
                        "patch_row": task["patch_row"],
                        "patch_col": task["patch_col"],
                        "norm_x": task["norm_x"],
                        "norm_y": task["norm_y"],
                        "candidates": task["candidates"],
                        "error": str(e),
                        "interpretable": False,
                    }
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=str, default="gemini_key.txt")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output", type=str, default="results/judge_evaluation.json")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    import google.generativeai as genai
    api_key = open(args.api_key_file).read().strip()
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(args.model_name)

    # Load image paths
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:NUM_PATCHES]
    print(f"Found {len(image_paths)} images in {IMAGES_DIR}")

    # Pre-sample normalized coordinates (constant across all models and layers)
    norm_coords = sample_normalized_coordinates(len(image_paths), seed=SEED)
    print(f"Sampled {len(norm_coords)} normalized coordinates (seed={SEED})")

    # Load existing results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Resuming from {output_path}")

    total_calls = 0
    t_start = time.time()

    for model_name, config in MODELS.items():
        results_dir = config["results_dir"]
        requested_layers = config["layers"]

        available = get_available_layers(results_dir)
        layers = [l for l in requested_layers if l in available]
        missing = [l for l in requested_layers if l not in available]
        if missing:
            print(f"\n{model_name}: layers {missing} not available, using {layers}")
        else:
            print(f"\n{model_name}: evaluating layers {layers}")

        if model_name not in all_results:
            all_results[model_name] = {}

        for layer in layers:
            layer_key = str(layer)
            existing = all_results[model_name].get(layer_key, {}).get("patches", [])
            n_existing = len(existing)

            if n_existing >= NUM_PATCHES:
                n_interp = sum(1 for r in existing if r.get("interpretable"))
                print(f"  Layer {layer}: already done ({n_interp}/{n_existing} interpretable)")
                continue

            results = evaluate_layer(
                gemini_model, results_dir, layer, image_paths,
                norm_coords, existing
            )

            n_new = len(results) - n_existing
            total_calls += n_new
            n_interp = sum(1 for r in results if r.get("interpretable"))
            pct = 100 * n_interp / len(results) if results else 0

            all_results[model_name][layer_key] = {
                "n_patches": len(results),
                "n_interpretable": n_interp,
                "pct_interpretable": round(pct, 1),
                "patches": results,
            }

            print(f"  Layer {layer}: {n_interp}/{len(results)} interpretable ({pct:.1f}%) [{n_new} new calls]")

            # Save after each layer
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone! {total_calls} API calls in {elapsed:.0f}s ({total_calls/max(elapsed,1):.1f} calls/sec)")
    print(f"Results saved to {output_path}")

    # Print summary table
    print(f"\n{'Model':<18} {'Layer':>6} {'Interp%':>8}")
    print("-" * 35)
    for model_name in MODELS:
        for layer_key in sorted(all_results.get(model_name, {}), key=int):
            d = all_results[model_name][layer_key]
            print(f"{model_name:<18} {layer_key:>6} {d['pct_interpretable']:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
