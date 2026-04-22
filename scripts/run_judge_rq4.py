#!/usr/bin/env python3
"""RQ4: Judge interpretability across layers, single-frame vs multi-frame.

Extends RQ1 interpretability curve to the cross-layer × cross-frame comparison.
For each model × condition × layer, samples 1 patch per image (100 images),
asks Gemini judge whether the top-5 NN words are interpretable.

Uses the SAME normalized coordinates across all conditions and layers (seed=43,
matching RQ2/RQ3 vidframes split) so that each image gets the same patch
position everywhere — enabling paired McNemar tests.

Output structure:
{
  "Molmo2-8B": {
    "single": {"1": {"n_patches": 100, "n_interpretable": 45, ...}, "16": ..., "24": ...},
    "multi":  {"1": {...}, "16": {...}, "24": {...}}
  },
  "Qwen2.5-VL-7B": {...}
}

Usage:
    python scripts/run_judge_rq4.py \
        --api-key-file gemini_key.txt \
        --layers 1,16,24 \
        --output results/judge_evaluation_rq4.json
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


# ── Config ──────────────────────────────────────────────────────────────────

_scratch = Path(os.environ.get("SCRATCH", "data")) / "latentlens"
IMAGES_DIR = _scratch / "molmo2cap_frames_500"

MODELS = {
    "Molmo2-8B": {
        "single": "results/molmo2cap_frames_500_molmo2_spatial",
        "multi":  "results/molmo2cap_videos_500_molmo2_spatial",
    },
    "Qwen2.5-VL-7B": {
        "single": "results/molmo2cap_frames_500_qwen25vl_spatial",
        "multi":  "results/molmo2cap_videos_500_qwen25vl_spatial",
    },
}

BBOX_SIZE = 3
SEED = 43  # Same as RQ2 vidframes / RQ3 for paired comparison
MAX_WORKERS = 10


# ── Helpers (reused from run_judge_evaluation.py) ───────────────────────────

def resize_natural(pil_img, max_side=512):
    w, h = pil_img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


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


# ── Core evaluation ─────────────────────────────────────────────────────────

def evaluate_condition(gemini_model, results_dir, layer, image_paths,
                       norm_coords, existing_patches=None):
    """Evaluate patches for one model+condition+layer. Returns list of results."""
    layer_file = Path(results_dir) / f"latentlens_layer{layer}.json"
    if not layer_file.exists():
        print(f"      WARNING: {layer_file} not found, skipping")
        return []

    with open(layer_file) as f:
        data = json.load(f)

    images = data["results"]
    n_images = min(len(images), len(image_paths), len(norm_coords))

    results = list(existing_patches or [])
    existing_idxs = {r["image_idx"] for r in results}
    tasks = []

    for img_idx in range(n_images):
        if img_idx in existing_idxs:
            continue

        img_data = images[img_idx]
        img_filename = img_data.get("image_path", "")
        img_path = image_paths[img_idx] if not img_filename else IMAGES_DIR / img_filename
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

        if not candidates:
            results.append({
                "image_idx": img_idx,
                "image_path": img_filename,
                "patch_row": patch["patch_row"],
                "patch_col": patch["patch_col"],
                "norm_x": norm_x,
                "norm_y": norm_y,
                "candidates": [],
                "interpretable": False,
                "note": "all-whitespace neighbors",
            })
            continue

        tasks.append({
            "image_idx": img_idx,
            "image_path": img_filename,
            "patch_row": patch["patch_row"],
            "patch_col": patch["patch_col"],
            "norm_x": norm_x,
            "norm_y": norm_y,
            "candidates": candidates,
            "img_with_bbox": img_with_bbox,
            "cropped": cropped,
        })

    if not tasks:
        return results

    def run_task(task):
        prompt = IMAGE_PROMPT_WITH_CROP.format(
            candidate_words=json.dumps(task["candidates"])
        )
        for attempt in range(3):
            try:
                resp = call_gemini(
                    gemini_model, task["img_with_bbox"], task["cropped"], prompt
                )
                return {
                    "image_idx": task["image_idx"],
                    "image_path": task["image_path"],
                    "patch_row": task["patch_row"],
                    "patch_col": task["patch_col"],
                    "norm_x": task["norm_x"],
                    "norm_y": task["norm_y"],
                    "candidates": task["candidates"],
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
    parser = argparse.ArgumentParser(
        description="RQ4: Judge interpretability across layers, single vs multi-frame")
    parser.add_argument("--api-key-file", type=str, default="gemini_key.txt")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--layers", type=str, default="1,16,24",
                        help="Comma-separated layer indices")
    parser.add_argument("--output", type=str,
                        default="results/judge_evaluation_rq4.json")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    import google.generativeai as genai
    api_key = open(args.api_key_file).read().strip()
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(args.model_name)

    # Load image paths (sorted for reproducibility)
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Found {len(image_paths)} images in {IMAGES_DIR}")

    # Same coordinates as RQ2/RQ3 vidframes for paired comparison
    norm_coords = sample_normalized_coordinates(len(image_paths), seed=SEED)
    print(f"Sampled {len(norm_coords)} normalized coordinates (seed={SEED})")
    print(f"Layers: {layers}")

    # Load existing results for resume
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Resuming from {output_path}")

    total_calls = 0
    t_start = time.time()

    for model_name, conditions in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        if model_name not in all_results:
            all_results[model_name] = {}

        for condition, results_dir in conditions.items():
            print(f"\n  --- {condition} ---")

            if not Path(results_dir).exists():
                print(f"    {results_dir} not found, skipping")
                continue

            if condition not in all_results[model_name]:
                all_results[model_name][condition] = {}

            for layer in layers:
                layer_key = str(layer)
                existing = all_results[model_name][condition].get(
                    layer_key, {}).get("patches", [])

                if len(existing) >= len(image_paths):
                    n_interp = sum(1 for r in existing if r.get("interpretable"))
                    pct = 100 * n_interp / len(existing)
                    print(f"    Layer {layer}: already done "
                          f"({n_interp}/{len(existing)} = {pct:.1f}% interpretable)")
                    continue

                results = evaluate_condition(
                    gemini_model, results_dir, layer, image_paths,
                    norm_coords, existing
                )

                n_new = len(results) - len(existing)
                total_calls += n_new
                n_interp = sum(1 for r in results if r.get("interpretable"))
                n_total = len(results)
                pct = 100 * n_interp / n_total if n_total else 0

                all_results[model_name][condition][layer_key] = {
                    "n_patches": n_total,
                    "n_interpretable": n_interp,
                    "pct_interpretable": round(pct, 1),
                    "patches": results,
                }

                print(f"    Layer {layer}: {n_interp}/{n_total} interpretable "
                      f"({pct:.1f}%) [{n_new} new calls]")

                # Save after each layer (resume-safe)
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone! {total_calls} API calls in {elapsed:.0f}s "
          f"({total_calls/max(elapsed,1):.1f} calls/sec)")
    print(f"Results saved to {output_path}")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'Model':<18} {'Cond':<8} {'Layer':>6} {'Interp%':>8}")
    print("-" * 45)
    for model_name in MODELS:
        for condition in ("single", "multi"):
            cond_data = all_results.get(model_name, {}).get(condition, {})
            for layer_key in sorted(cond_data, key=lambda x: int(x)):
                d = cond_data[layer_key]
                if isinstance(d, dict) and "pct_interpretable" in d:
                    print(f"{model_name:<18} {condition:<8} {layer_key:>6} "
                          f"{d['pct_interpretable']:>7.1f}%")
        print()

    # ── Paired McNemar tests ────────────────────────────────────────────────
    print("Paired McNemar tests (single vs multi, same image+patch):")
    print("-" * 60)
    for model_name in MODELS:
        single_data = all_results.get(model_name, {}).get("single", {})
        multi_data = all_results.get(model_name, {}).get("multi", {})

        for layer_key in sorted(set(single_data) & set(multi_data), key=int):
            s_patches = {r["image_idx"]: r.get("interpretable", False)
                         for r in single_data[layer_key].get("patches", [])}
            m_patches = {r["image_idx"]: r.get("interpretable", False)
                         for r in multi_data[layer_key].get("patches", [])}

            common = sorted(set(s_patches) & set(m_patches))
            if len(common) < 10:
                continue

            # McNemar contingency: b = single-yes & multi-no, c = single-no & multi-yes
            b = sum(1 for i in common if s_patches[i] and not m_patches[i])
            c = sum(1 for i in common if not s_patches[i] and m_patches[i])
            n_both_yes = sum(1 for i in common if s_patches[i] and m_patches[i])
            n_both_no = sum(1 for i in common if not s_patches[i] and not m_patches[i])

            # McNemar exact (binomial test)
            from scipy.stats import binomtest
            if b + c > 0:
                p_value = binomtest(b, b + c, 0.5).pvalue
            else:
                p_value = 1.0

            s_pct = 100 * sum(s_patches[i] for i in common) / len(common)
            m_pct = 100 * sum(m_patches[i] for i in common) / len(common)

            print(f"  {model_name} layer {layer_key}: "
                  f"single={s_pct:.1f}% multi={m_pct:.1f}% "
                  f"delta={m_pct-s_pct:+.1f}%  "
                  f"b={b} c={c} p={p_value:.4e} (n={len(common)})")


if __name__ == "__main__":
    main()
