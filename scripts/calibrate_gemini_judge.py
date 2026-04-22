#!/usr/bin/env python3
"""
Calibrate Gemini as an LLM judge for LatentLens interpretability evaluation.

Runs the same evaluation prompt on the same instances that have both:
- Human annotations (ground truth)
- GPT-5 judge results (reference baseline)

Compares Gemini's judgements against both to assess calibration.
Supports iterating on the prompt until Gemini matches GPT-5/human agreement.

Usage:
    python scripts/calibrate_gemini_judge.py \
        --api-key YOUR_GEMINI_KEY \
        --data-type nn \
        --num-samples 50
"""

import os
import sys
import json
import argparse
import base64
import io
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import requests
from io import BytesIO

# Add the old project's llm_judge to path for shared utilities
OLD_PROJECT = Path("/home/nlp/users/bkroje/vl_embedding_spaces/third_party/molmo")
sys.path.insert(0, str(OLD_PROJECT / "llm_judge"))
sys.path.insert(0, str(OLD_PROJECT / "human_correlations"))
from utils import calculate_square_bbox_from_patch, draw_bbox_on_image, resize_and_pad

HUMAN_CORR_DIR = OLD_PROJECT / "human_correlations"

# ── Prompt ──────────────────────────────────────────────────────────────────
# Start with the exact GPT-5 prompt (IMAGE_PROMPT_WITH_CROP from prompts.py).
# We import it so any tweaks to the original are picked up, but also allow
# overriding it here for Gemini-specific calibration experiments.

from prompts import IMAGE_PROMPT_WITH_CROP as GPT5_PROMPT

# Gemini prompt — start identical to GPT-5, tweak as needed during calibration
GEMINI_PROMPT = GPT5_PROMPT


def call_gemini(model, image_with_bbox, cropped_image, prompt, api_key):
    """Call Gemini API with image(s) + text prompt, return parsed JSON."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    gen_model = genai.GenerativeModel(model)

    parts = [image_with_bbox, prompt]
    if cropped_image is not None:
        parts.append(cropped_image)

    response = gen_model.generate_content(parts)
    response_text = response.text

    # Extract JSON from response
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)

    print(f"No JSON found in response: {response_text[:200]}")
    return {
        "interpretable": False,
        "concrete_words": [],
        "abstract_words": [],
        "global_words": [],
        "reasoning": response_text
    }


def load_human_labels(data_type="nn"):
    """Load human annotations, return dict: instance_id -> bool (majority vote)."""
    if data_type == "nn":
        data_dir = HUMAN_CORR_DIR / "interp_data_nn"
    else:
        data_dir = HUMAN_CORR_DIR / "interp_data_contextual"

    # Load data.json for candidate info
    with open(data_dir / "data.json") as f:
        data_items = {item["id"]: item for item in json.load(f)}

    # Load all human annotation files
    results_dir = data_dir / "results"
    human_votes = defaultdict(list)  # instance_id -> list of bool

    for result_file in results_dir.glob("evaluation_*.json"):
        with open(result_file) as f:
            data = json.load(f)
        for result in data.get("results", []):
            iid = result.get("instanceId")
            if not iid or iid not in data_items:
                continue
            selected = result.get("selectedWords", [])
            none_selected = result.get("noneSelected", False)
            is_interp = len(selected) > 0 or not none_selected
            human_votes[iid].append(is_interp)

    # Majority vote
    human_labels = {}
    for iid, votes in human_votes.items():
        human_labels[iid] = sum(votes) > len(votes) / 2

    return human_labels, data_items


def load_gpt5_labels(data_type="nn"):
    """Load GPT-5 judge results, return dict: instance_id -> bool."""
    if data_type == "nn":
        results_file = HUMAN_CORR_DIR / "llm_judge_results" / "human_study_llm_results.json"
    else:
        results_file = HUMAN_CORR_DIR / "llm_judge_results_contextual" / "human_study_llm_results.json"

    with open(results_file) as f:
        data = json.load(f)

    gpt5_labels = {}
    gpt5_full = {}
    for r in data.get("results", []):
        iid = r.get("instance_id")
        if not iid or "gpt_response" not in r:
            continue
        gpt5_labels[iid] = r["gpt_response"].get("interpretable", False)
        gpt5_full[iid] = r

    return gpt5_labels, gpt5_full


def prepare_image(image_url, patch_row, patch_col):
    """Download image, preprocess, draw bbox, crop region. Returns (image_with_bbox, cropped, processed)."""
    for attempt in range(3):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    pil_image = Image.open(BytesIO(response.content))

    image_np = np.array(pil_image.convert("RGB"))
    processed_np, image_mask = resize_and_pad(image_np, (512, 512), normalize=False)
    processed_np = (processed_np * 255).astype(np.uint8)
    processed_image = Image.fromarray(processed_np)

    actual_patch_size = 512 / 24
    bbox = calculate_square_bbox_from_patch(patch_row, patch_col, patch_size=actual_patch_size, size=3)

    image_with_bbox = draw_bbox_on_image(processed_image, bbox)
    cropped = processed_image.crop(bbox)

    return image_with_bbox, cropped


def compute_metrics(labels_a, labels_b, name_a="A", name_b="B"):
    """Compute agreement metrics between two binary label dicts (on shared keys)."""
    from sklearn.metrics import cohen_kappa_score

    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        return {"n": 0, "error": "no shared instances"}

    a = np.array([int(labels_a[k]) for k in shared])
    b = np.array([int(labels_b[k]) for k in shared])

    accuracy = np.mean(a == b)
    kappa = cohen_kappa_score(a, b)

    # Confusion matrix
    tp = int(np.sum((a == 1) & (b == 1)))
    tn = int(np.sum((a == 0) & (b == 0)))
    fp = int(np.sum((a == 0) & (b == 1)))
    fn = int(np.sum((a == 1) & (b == 0)))

    return {
        "n": len(shared),
        f"{name_a}_positive": int(np.sum(a)),
        f"{name_b}_positive": int(np.sum(b)),
        "accuracy": float(accuracy),
        "cohens_kappa": float(kappa),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Gemini judge against GPT-5 + human labels")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model name")
    parser.add_argument("--data-type", type=str, default="nn", choices=["nn", "contextual"])
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of instances to evaluate (0 = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/gemini_calibration_{data_type}.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing partial results")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set GEMINI_API_KEY")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path(f"results/gemini_calibration_{args.data_type}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load reference labels
    print("Loading human annotations...")
    human_labels, data_items = load_human_labels(args.data_type)
    print(f"  {len(human_labels)} instances with human labels")

    print("Loading GPT-5 judge results...")
    gpt5_labels, gpt5_full = load_gpt5_labels(args.data_type)
    print(f"  {len(gpt5_labels)} instances with GPT-5 labels")

    # Instances that have BOTH human and GPT-5 labels
    shared_ids = sorted(set(human_labels) & set(gpt5_labels) & set(data_items))
    print(f"  {len(shared_ids)} instances with both human + GPT-5 labels")

    # Print baseline: GPT-5 vs human
    print("\n=== Baseline: GPT-5 vs Human ===")
    baseline = compute_metrics(human_labels, gpt5_labels, "human", "gpt5")
    print(f"  n={baseline['n']}, accuracy={baseline['accuracy']:.3f}, kappa={baseline['cohens_kappa']:.3f}")
    print(f"  Human positive: {baseline['human_positive']}/{baseline['n']}, GPT-5 positive: {baseline['gpt5_positive']}/{baseline['n']}")
    print(f"  Confusion: {baseline['confusion']}")

    # Sample instances
    rng = np.random.RandomState(args.seed)
    if args.num_samples > 0 and args.num_samples < len(shared_ids):
        sample_ids = list(rng.choice(shared_ids, size=args.num_samples, replace=False))
    else:
        sample_ids = shared_ids

    # Load existing results if resuming
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
        existing = {r["instance_id"]: r for r in existing_data.get("results", [])}
        print(f"  Resuming: {len(existing)} already evaluated")

    # Run Gemini on samples
    print(f"\n=== Running Gemini ({args.model}) on {len(sample_ids)} instances ===")
    gemini_results = []
    gemini_labels = {}

    for i, iid in enumerate(tqdm(sample_ids)):
        # Use existing result if available
        if iid in existing:
            gemini_results.append(existing[iid])
            resp = existing[iid].get("gemini_response", {})
            gemini_labels[iid] = resp.get("interpretable", False)
            continue

        item = data_items[iid]
        image_url = item.get("image_url")
        patch_row = item.get("patch_row")
        patch_col = item.get("patch_col")
        candidates = item.get("candidates", [])

        # For contextual data, extract full words from [sentence, token] tuples
        if args.data_type == "contextual":
            sys.path.insert(0, str(OLD_PROJECT / "llm_judge"))
            from run_single_model_with_viz_contextual import extract_full_word_from_token
            word_candidates = []
            for c in candidates:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    word = extract_full_word_from_token(c[0], c[1])
                    word_candidates.append(word if word else c[1])
                else:
                    word_candidates.append(str(c))
            candidates = word_candidates

        # Prepare image
        image_with_bbox, cropped = prepare_image(image_url, patch_row, patch_col)

        # Format prompt
        prompt = GEMINI_PROMPT.format(candidate_words=json.dumps(candidates))

        # Call Gemini (with retry)
        for attempt in range(3):
            try:
                gemini_response = call_gemini(args.model, image_with_bbox, cropped, prompt, api_key)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\nFailed after 3 attempts on {iid}: {e}")
                    gemini_response = {"interpretable": False, "concrete_words": [], "abstract_words": [], "global_words": [], "reasoning": f"ERROR: {e}"}
                    break
                time.sleep(2 ** attempt)

        is_interp = gemini_response.get("interpretable", False)
        gemini_labels[iid] = is_interp

        result = {
            "instance_id": iid,
            "candidates": candidates,
            "patch_row": patch_row,
            "patch_col": patch_col,
            "image_url": image_url,
            "gemini_response": gemini_response,
            "human_label": human_labels.get(iid),
            "gpt5_label": gpt5_labels.get(iid),
            "gemini_label": is_interp,
        }
        gemini_results.append(result)

        # Save intermediate results every 10 instances
        if (i + 1) % 10 == 0:
            _save(output_path, gemini_results, sample_ids, args, baseline)

        # Rate limiting
        time.sleep(0.3)

    # Final save + metrics
    # Compute Gemini vs Human
    gemini_vs_human = compute_metrics(human_labels, gemini_labels, "human", "gemini")
    # Compute Gemini vs GPT-5
    gemini_vs_gpt5 = compute_metrics(gpt5_labels, gemini_labels, "gpt5", "gemini")

    _save(output_path, gemini_results, sample_ids, args, baseline,
          gemini_vs_human=gemini_vs_human, gemini_vs_gpt5=gemini_vs_gpt5)

    # Print results
    print(f"\n=== Results ({len(gemini_labels)} instances) ===")
    print(f"\nGemini vs Human:")
    print(f"  accuracy={gemini_vs_human['accuracy']:.3f}, kappa={gemini_vs_human['cohens_kappa']:.3f}")
    print(f"  Human positive: {gemini_vs_human['human_positive']}/{gemini_vs_human['n']}")
    print(f"  Gemini positive: {gemini_vs_human['gemini_positive']}/{gemini_vs_human['n']}")
    print(f"  Confusion: {gemini_vs_human['confusion']}")

    print(f"\nGemini vs GPT-5:")
    print(f"  accuracy={gemini_vs_gpt5['accuracy']:.3f}, kappa={gemini_vs_gpt5['cohens_kappa']:.3f}")
    print(f"  GPT-5 positive: {gemini_vs_gpt5['gpt5_positive']}/{gemini_vs_gpt5['n']}")
    print(f"  Gemini positive: {gemini_vs_gpt5['gemini_positive']}/{gemini_vs_gpt5['n']}")
    print(f"  Confusion: {gemini_vs_gpt5['confusion']}")

    print(f"\nBaseline (GPT-5 vs Human):")
    print(f"  accuracy={baseline['accuracy']:.3f}, kappa={baseline['cohens_kappa']:.3f}")

    # Show disagreement examples
    print(f"\n=== Disagreement Examples (Gemini vs GPT-5) ===")
    disagree_count = 0
    for r in gemini_results:
        iid = r["instance_id"]
        if r.get("gemini_label") != r.get("gpt5_label") and disagree_count < 5:
            print(f"\n  Instance: {iid}")
            print(f"  Candidates: {r['candidates']}")
            print(f"  Human={r.get('human_label')}, GPT-5={r.get('gpt5_label')}, Gemini={r.get('gemini_label')}")
            gemini_resp = r.get("gemini_response", {})
            print(f"  Gemini reasoning: {gemini_resp.get('reasoning', 'N/A')[:200]}")
            gpt5_resp = gpt5_full.get(iid, {}).get("gpt_response", {})
            print(f"  GPT-5 reasoning: {gpt5_resp.get('reasoning', 'N/A')[:200]}")
            disagree_count += 1

    print(f"\nResults saved to {output_path}")


def _save(output_path, results, sample_ids, args, baseline,
          gemini_vs_human=None, gemini_vs_gpt5=None):
    output = {
        "model": args.model,
        "data_type": args.data_type,
        "num_samples": len(sample_ids),
        "evaluated": len(results),
        "baseline_gpt5_vs_human": baseline,
        "gemini_vs_human": gemini_vs_human,
        "gemini_vs_gpt5": gemini_vs_gpt5,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
