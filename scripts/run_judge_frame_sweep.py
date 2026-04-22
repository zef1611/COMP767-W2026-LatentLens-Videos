#!/usr/bin/env python3
"""Run Gemini LLM judge evaluation across frame counts for RQ3.

Evaluates patches from multi-frame video LatentLens results at different
frame counts (2, 8, 16) for Molmo2-8B and Qwen2.5-VL-7B.

Uses the SAME normalized coordinates and seeds as RQ2/RQ3 (SEED=43),
enabling comparison across all frame counts.

Existing judge results:
  - 1f: results/judge_evaluation_rq2.json (vidframes split)
  - 4f: results/judge_evaluation_rq3.json

This script fills in: 2f, 8f, 16f.

Usage:
    python scripts/run_judge_frame_sweep.py \
        --api-key-file gemini_key.txt \
        --output results/judge_frame_sweep.json
"""

import argparse
import json
import os
import random
import sys
import time
import concurrent.futures
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                        "latentlens/reproduce/scripts/evaluate"))
from prompts import IMAGE_PROMPT_WITH_CROP


def resize_natural(pil_img, max_side=512):
    w, h = pil_img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


# ── Config ──────────────────────────────────────────────────────────────────

FRAME_COUNTS = [2, 8, 16]

MODELS = {
    "Molmo2-8B": {
        "key": "molmo2",
        "frame_dirs": {
            2: "results/molmo2cap_videos_500_molmo2_2f_spatial",
            8: "results/molmo2cap_videos_500_molmo2_8f_spatial",
            16: "results/molmo2cap_videos_500_molmo2_16f_spatial",
        },
    },
    "Qwen2.5-VL-7B": {
        "key": "qwen25vl",
        "frame_dirs": {
            2: "results/molmo2cap_videos_500_qwen25vl_2f_spatial",
            8: "results/molmo2cap_videos_500_qwen25vl_8f_spatial",
            16: "results/molmo2cap_videos_500_qwen25vl_16f_spatial",
        },
    },
}

_scratch = Path(os.environ.get("SCRATCH", "data")) / "latentlens"
IMAGES_DIR = _scratch / "molmo2cap_frames_500"

BBOX_SIZE = 3
SEED = 43
MAX_WORKERS = 8
MAX_IMAGES = 100


def sample_normalized_coordinates(n_images, seed=43):
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n_images)]


def norm_to_patch(norm_x, norm_y, grid_h, grid_w):
    row = int(norm_y * grid_h)
    col = int(norm_x * grid_w)
    row = max(0, min(row, grid_h - BBOX_SIZE))
    col = max(0, min(col, grid_w - BBOX_SIZE))
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
    left = col / grid_w * disp_w
    top = row / grid_h * disp_h
    right = (col + BBOX_SIZE) / grid_w * disp_w
    bottom = (row + BBOX_SIZE) / grid_h * disp_h
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


def build_tasks(results_dir, layer, image_dir, norm_coords, existing_keys,
                max_images=100):
    layer_file = Path(results_dir) / f"latentlens_layer{layer}.json"
    if not layer_file.exists():
        print(f"    WARNING: {layer_file} not found, skipping")
        return []

    with open(layer_file) as f:
        data = json.load(f)

    images = data["results"]
    n_images = min(len(images), len(norm_coords), max_images)
    tasks = []

    for img_idx in range(n_images):
        task_key = f"{img_idx}"
        if task_key in existing_keys:
            continue

        img_data = images[img_idx]
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
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--output", type=str,
                        default="results/judge_frame_sweep.json")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    import google.generativeai as genai
    api_key = open(args.api_key_file).read().strip()
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(args.model_name)

    norm_coords = sample_normalized_coordinates(1000, seed=SEED)

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
        for nf in FRAME_COUNTS:
            results_dir = config["frame_dirs"].get(nf)
            if results_dir is None:
                continue

            result_key = f"{model_name}_{nf}f"
            print(f"\n{'='*60}")
            print(f"  {result_key}")
            print(f"{'='*60}")

            if result_key not in all_results:
                all_results[result_key] = {"model": model_name, "n_frames": nf,
                                           "layer": args.layer, "patches": []}

            existing = all_results[result_key].get("patches", [])
            existing_keys = {str(p["image_idx"]) for p in existing}

            tasks = build_tasks(
                results_dir, args.layer,
                IMAGES_DIR, norm_coords, existing_keys,
                max_images=args.max_images
            )
            print(f"  {len(tasks)} new tasks")

            if not tasks:
                n_interp = sum(1 for p in existing if p.get("interpretable"))
                print(f"  Already done: {n_interp}/{len(existing)} interpretable")
                continue

            # Separate tasks with/without candidates
            no_candidates = []
            with_candidates = []
            for t in tasks:
                if not t["candidates"]:
                    no_candidates.append({
                        "image_idx": t["image_idx"],
                        "image_path": t["image_path"],
                        "patch_row": t["patch_row"],
                        "patch_col": t["patch_col"],
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
                            gemini_model, task["img_with_bbox"],
                            task["cropped"], prompt
                        )
                        return {
                            "image_idx": task["image_idx"],
                            "image_path": task["image_path"],
                            "patch_row": task["patch_row"],
                            "patch_col": task["patch_col"],
                            "candidates": task["candidates"],
                            "neighbors": task["neighbors"],
                            "gemini_response": resp,
                            "interpretable": resp.get("interpretable", False),
                        }
                    except Exception as e:
                        if attempt == 2:
                            return {
                                "image_idx": task["image_idx"],
                                "image_path": task["image_path"],
                                "patch_row": task["patch_row"],
                                "patch_col": task["patch_col"],
                                "candidates": task["candidates"],
                                "neighbors": task["neighbors"],
                                "error": str(e),
                                "interpretable": False,
                            }
                        time.sleep(2 ** attempt)

            print(f"  Running {len(with_candidates)} API calls "
                  f"(max {MAX_WORKERS} workers)...")
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(run_task, t) for t in with_candidates]
                done = 0
                for f in concurrent.futures.as_completed(futures):
                    existing.append(f.result())
                    done += 1
                    if done % 25 == 0:
                        print(f"    {done}/{len(with_candidates)} done")
                        all_results[result_key]["patches"] = existing
                        with open(output_path, "w") as fout:
                            json.dump(all_results, fout, indent=2)

            total_calls += len(with_candidates)

            n_interp = sum(1 for p in existing if p.get("interpretable"))
            n_total = len(existing)
            pct = 100 * n_interp / n_total if n_total else 0

            all_results[result_key] = {
                "model": model_name,
                "n_frames": nf,
                "layer": args.layer,
                "n_patches": n_total,
                "n_interpretable": n_interp,
                "pct_interpretable": round(pct, 1),
                "patches": existing,
            }

            with open(output_path, "w") as fout:
                json.dump(all_results, fout, indent=2)
            print(f"  {n_interp}/{n_total} interpretable ({pct:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\nDone! {total_calls} API calls in {elapsed:.0f}s")
    print(f"Results saved to {output_path}")

    # Summary table
    print(f"\n{'Condition':<25} {'Total':>6} {'Interp':>7} {'%':>7}")
    print("-" * 47)
    for key in sorted(all_results.keys()):
        d = all_results[key]
        print(f"{key:<25} {d.get('n_patches',0):>6} "
              f"{d.get('n_interpretable',0):>7} "
              f"{d.get('pct_interpretable',0):>6.1f}%")


if __name__ == "__main__":
    main()
