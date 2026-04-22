#!/usr/bin/env python3
"""Run Gemini LLM judge evaluation for RQ2.

Evaluates 1000 patches (1 per image) across 4 models at a single layer.
Uses 500 PixMo-Cap + 500 Molmo2-Cap images.
Shares the same judge protocol as run_judge_evaluation.py.

Usage:
    python scripts/run_judge_rq2.py \
        --api-key-file gemini_key.txt \
        --layer 24 \
        --output results/judge_evaluation_rq2.json
"""

import argparse
import os
import json
import random
import sys
import time
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                        "latentlens/reproduce/scripts/evaluate"))
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
        "results_dirs": {
            "pixmo": "results/pixmo_cap_500_molmo-7b-d_spatial",
            "vidframes": "results/molmo2cap_frames_500_molmo-7b-d_spatial",
        },
    },
    "Idefics3-8B": {
        "results_dirs": {
            "pixmo": "results/pixmo_cap_500_idefics3_spatial",
            "vidframes": "results/molmo2cap_frames_500_idefics3_spatial",
        },
    },
    "Molmo2-8B": {
        "results_dirs": {
            "pixmo": "results/pixmo_cap_500_molmo2_spatial",
            "vidframes": "results/molmo2cap_frames_500_molmo2_spatial",
        },
    },
    "Qwen2.5-VL-7B": {
        "results_dirs": {
            "pixmo": "results/pixmo_cap_500_qwen25vl_spatial",
            "vidframes": "results/molmo2cap_frames_500_qwen25vl_spatial",
        },
    },
}

_scratch = Path(os.environ.get("SCRATCH", "data")) / "latentlens"
IMAGES_DIRS = {
    "pixmo": _scratch / "pixmo_cap_500",
    "vidframes": _scratch / "molmo2cap_frames_500",
}

BBOX_SIZE = 3
SEED = 42
MAX_WORKERS = 20


def sample_normalized_coordinates(n_images, seed=42):
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n_images)]


def norm_to_patch(norm_x, norm_y, grid_h, grid_w):
    half = BBOX_SIZE // 2
    row = int(norm_y * grid_h)
    col = int(norm_x * grid_w)
    row = max(half, min(row, grid_h - 1 - half))
    col = max(half, min(col, grid_w - 1 - half))
    return row, col


def get_grid_for_image(img_data):
    if "grid_h" in img_data and "grid_w" in img_data:
        return img_data["grid_h"], img_data["grid_w"]
    patches = img_data.get("patches", [])
    if not patches:
        return 1, 1
    return (max(p["patch_row"] for p in patches) + 1,
            max(p["patch_col"] for p in patches) + 1)


def find_patch_at(img_data, row, col):
    patches = img_data.get("patches", [])
    for p in patches:
        if p["patch_row"] == row and p["patch_col"] == col:
            return p
    best, best_dist = None, float("inf")
    for p in patches:
        d = abs(p["patch_row"] - row) + abs(p["patch_col"] - col)
        if d < best_dist:
            best, best_dist = p, d
    return best


def prepare_judge_input(image_path, patch, grid_h, grid_w):
    pil_img = Image.open(image_path).convert("RGB")
    display = resize_natural(pil_img, 512)
    disp_w, disp_h = display.size

    row, col = patch["patch_row"], patch["patch_col"]
    half = BBOX_SIZE // 2
    left = (col - half) / grid_w * disp_w
    top = (row - half) / grid_h * disp_h
    right = (col - half + BBOX_SIZE) / grid_w * disp_w
    bottom = (row - half + BBOX_SIZE) / grid_h * disp_h
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


def build_tasks_for_dataset(ds_key, results_dir, layer, image_dir, norm_coords, existing_keys):
    """Build judge tasks for one dataset split."""
    layer_file = Path(results_dir) / f"latentlens_layer{layer}.json"
    if not layer_file.exists():
        print(f"    WARNING: {layer_file} not found, skipping")
        return []

    with open(layer_file) as f:
        data = json.load(f)

    images = data["results"]
    n_images = min(len(images), len(norm_coords))
    tasks = []

    for img_idx in range(n_images):
        task_key = f"{ds_key}_{img_idx}"
        if task_key in existing_keys:
            continue

        img_data = images[img_idx]

        # Use image_path field from results to find actual file
        img_filename = img_data.get("image_path", "")
        img_path = image_dir / img_filename
        if not img_path.exists():
            continue

        grid_h, grid_w = get_grid_for_image(img_data)
        norm_x, norm_y = norm_coords[img_idx]
        row, col = norm_to_patch(norm_x, norm_y, grid_h, grid_w)
        patch = find_patch_at(img_data, row, col)
        if patch is None:
            continue

        img_with_bbox, cropped, candidates = prepare_judge_input(
            img_path, patch, grid_h, grid_w
        )

        tasks.append({
            "ds_key": ds_key,
            "image_idx": img_idx,
            "image_path": img_filename,
            "patch_row": patch["patch_row"],
            "patch_col": patch["patch_col"],
            "norm_x": norm_x,
            "norm_y": norm_y,
            "candidates": candidates,
            "neighbors": patch.get("nearest_contextual_neighbors", [])[:5],
            "img_with_bbox": img_with_bbox,
            "cropped": cropped,
        })

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=str, default="gemini_key.txt")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--output", type=str, default="results/judge_evaluation_rq2.json")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    import google.generativeai as genai
    api_key = open(args.api_key_file).read().strip()
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(args.model_name)

    # Pre-sample normalized coordinates per dataset
    # Use different seeds for different datasets to get different patch positions
    # Sample more than needed; we'll only use as many as there are results
    norm_coords_pixmo = sample_normalized_coordinates(1000, seed=SEED)
    norm_coords_vidframes = sample_normalized_coordinates(1000, seed=SEED + 1)

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
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if model_name not in all_results:
            all_results[model_name] = {"layer": args.layer, "patches": []}

        existing = all_results[model_name].get("patches", [])
        existing_keys = {f"{p['ds_key']}_{p['image_idx']}" for p in existing
                         if "ds_key" in p}

        # Build tasks for both datasets
        all_tasks = []
        for ds_key, results_dir in config["results_dirs"].items():
            nc = norm_coords_pixmo if ds_key == "pixmo" else norm_coords_vidframes
            tasks = build_tasks_for_dataset(
                ds_key, results_dir, args.layer,
                IMAGES_DIRS[ds_key], nc, existing_keys
            )
            all_tasks.extend(tasks)
            print(f"  {ds_key}: {len(tasks)} new tasks")

        if not all_tasks:
            n_interp = sum(1 for p in existing if p.get("interpretable"))
            print(f"  Already done: {n_interp}/{len(existing)} interpretable")
            continue

        # Separate tasks with candidates from those without
        no_candidates = []
        with_candidates = []
        for t in all_tasks:
            if not t["candidates"]:
                no_candidates.append({
                    "ds_key": t["ds_key"],
                    "image_idx": t["image_idx"],
                    "image_path": t["image_path"],
                    "patch_row": t["patch_row"],
                    "patch_col": t["patch_col"],
                    "norm_x": t["norm_x"],
                    "norm_y": t["norm_y"],
                    "candidates": [],
                    "neighbors": t["neighbors"],
                    "interpretable": False,
                    "note": "all-whitespace neighbors",
                })
            else:
                with_candidates.append(t)

        existing.extend(no_candidates)

        def run_task(task):
            prompt = IMAGE_PROMPT_WITH_CROP.format(
                candidate_words=json.dumps(task["candidates"])
            )
            for attempt in range(3):
                try:
                    resp = call_gemini(
                        gemini_model, task["img_with_bbox"], task["cropped"], prompt
                    )
                    is_interp = resp.get("interpretable", False)
                    return {
                        "ds_key": task["ds_key"],
                        "image_idx": task["image_idx"],
                        "image_path": task["image_path"],
                        "patch_row": task["patch_row"],
                        "patch_col": task["patch_col"],
                        "norm_x": task["norm_x"],
                        "norm_y": task["norm_y"],
                        "candidates": task["candidates"],
                        "neighbors": task["neighbors"],
                        "gemini_response": resp,
                        "interpretable": is_interp,
                    }
                except Exception as e:
                    if attempt == 2:
                        return {
                            "ds_key": task["ds_key"],
                            "image_idx": task["image_idx"],
                            "image_path": task["image_path"],
                            "patch_row": task["patch_row"],
                            "patch_col": task["patch_col"],
                            "norm_x": task["norm_x"],
                            "norm_y": task["norm_y"],
                            "candidates": task["candidates"],
                            "neighbors": task["neighbors"],
                            "error": str(e),
                            "interpretable": False,
                        }
                    time.sleep(2 ** attempt)

        # Run judge in parallel
        print(f"  Running {len(with_candidates)} API calls (max {MAX_WORKERS} workers)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(run_task, t) for t in with_candidates]
            done = 0
            for f in concurrent.futures.as_completed(futures):
                existing.append(f.result())
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(with_candidates)} done")
                    # Periodic save
                    all_results[model_name]["patches"] = existing
                    with open(output_path, "w") as fout:
                        json.dump(all_results, fout, indent=2)

        total_calls += len(with_candidates)

        # Final counts
        n_interp = sum(1 for p in existing if p.get("interpretable"))
        n_total = len(existing)
        pct = 100 * n_interp / n_total if n_total else 0

        # Per-dataset breakdown
        for ds_key in ["pixmo", "vidframes"]:
            ds_patches = [p for p in existing if p.get("ds_key") == ds_key]
            ds_interp = sum(1 for p in ds_patches if p.get("interpretable"))
            ds_pct = 100 * ds_interp / len(ds_patches) if ds_patches else 0
            print(f"  {ds_key}: {ds_interp}/{len(ds_patches)} interpretable ({ds_pct:.1f}%)")

        all_results[model_name] = {
            "layer": args.layer,
            "n_patches": n_total,
            "n_interpretable": n_interp,
            "pct_interpretable": round(pct, 1),
            "patches": existing,
        }

        # Save after each model
        with open(output_path, "w") as fout:
            json.dump(all_results, fout, indent=2)
        print(f"  Total: {n_interp}/{n_total} interpretable ({pct:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\nDone! {total_calls} API calls in {elapsed:.0f}s")
    print(f"Results saved to {output_path}")

    # Summary
    print(f"\n{'Model':<18} {'Total':>6} {'Interp':>7} {'%':>7}")
    print("-" * 40)
    for model_name in MODELS:
        d = all_results.get(model_name, {})
        print(f"{model_name:<18} {d.get('n_patches',0):>6} {d.get('n_interpretable',0):>7} "
              f"{d.get('pct_interpretable',0):>6.1f}%")


if __name__ == "__main__":
    main()
