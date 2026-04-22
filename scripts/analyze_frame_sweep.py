#!/usr/bin/env python3
"""Analyze LatentLens results across different frame counts (1, 2, 4, 8, 16).

Compares nearest neighbors for the same middle frame across frame-count
conditions. Produces JSON + matplotlib plot showing how metrics change
with increasing temporal context.

Metrics per frame count:
  - Dynamic verb % of top-5 NNs
  - Top-1 cosine similarity (mean)
  - Jaccard overlap with single-frame (1f) baseline
  - Noun %, Adj %

Usage:
    python scripts/analyze_frame_sweep.py \
        --output results/frame_sweep_rq3.json \
        --plot paper/Interpreting-VideoLLMs/figures/rq3_frame_sweep.pdf
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import spacy

# POS taxonomy (same as RQ2/RQ3)
STATIVE_VERBS = {
    "know", "believe", "think", "understand", "realize", "recognize",
    "remember", "forget", "imagine", "suppose", "doubt", "mean",
    "feel", "hear", "see", "smell", "taste", "notice", "perceive",
    "like", "love", "hate", "prefer", "want", "wish", "need", "desire",
    "fear", "envy", "mind", "care", "appreciate", "dislike",
    "have", "own", "possess", "belong", "owe", "contain", "include",
    "consist", "involve", "lack",
    "be", "exist", "seem", "appear", "look", "sound", "resemble",
    "remain", "stay", "weigh", "measure", "cost", "equal", "fit",
    "depend", "deserve", "matter", "owe", "suit", "tend",
    "agree", "disagree", "deny", "promise", "refuse",
    "concern", "impress", "please", "satisfy", "surprise",
}


def classify_token(nlp, token_str, caption):
    """Classify a single NN token using sentence-context POS."""
    token_str = token_str.strip()
    if not token_str or not caption or len(token_str) <= 1:
        return "other"

    doc = nlp(caption)
    token_lower = token_str.lower().strip()

    matched = None
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() == token_lower:
            matched = tok
            break
    if matched is None:
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            if token_lower in tok.text.lower() and len(token_lower) >= 3:
                matched = tok
                break
    if matched is None:
        return "other"

    lemma_lower = matched.lemma_.lower()
    if matched.pos_ == "VERB":
        return "stative_verb" if lemma_lower in STATIVE_VERBS else "dynamic_verb"
    if matched.pos_ in ("NOUN", "PROPN"):
        return "noun"
    if matched.pos_ == "ADJ":
        return "adj"
    return "other"


# Result directory patterns
MODELS = {
    "Molmo2-8B": {
        "key": "molmo2",
        "single_frame_dir": "results/molmo2cap_frames_500_molmo2_spatial",
        "frame_dirs": {
            2: "results/molmo2cap_videos_500_molmo2_2f_spatial",
            4: "results/molmo2cap_videos_500_molmo2_spatial",
            8: "results/molmo2cap_videos_500_molmo2_8f_spatial",
            16: "results/molmo2cap_videos_500_molmo2_16f_spatial",
        },
    },
    "Qwen2.5-VL-7B": {
        "key": "qwen25vl",
        "single_frame_dir": "results/molmo2cap_frames_500_qwen25vl_spatial",
        "frame_dirs": {
            2: "results/molmo2cap_videos_500_qwen25vl_2f_spatial",
            4: "results/molmo2cap_videos_500_qwen25vl_spatial",
            8: "results/molmo2cap_videos_500_qwen25vl_8f_spatial",
            16: "results/molmo2cap_videos_500_qwen25vl_16f_spatial",
        },
    },
}


def load_results(results_dir, layer=24):
    """Load spatial LatentLens results."""
    path = Path(results_dir) / f"latentlens_layer{layer}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["results"]


def get_nn_tokens(patches_list, img_idx, max_patches=10, seed=42):
    """Get top-5 NN tokens from a sample of patches for an image.

    Samples up to max_patches uniformly to keep POS tagging tractable
    (~5000 tokens per condition instead of ~500K).
    """
    if img_idx >= len(patches_list):
        return [], []
    img_data = patches_list[img_idx]
    patches = img_data.get("patches", [])
    if not patches:
        return [], []

    # Deterministic sample of patches
    import random
    rng = random.Random(seed + img_idx)
    if len(patches) > max_patches:
        patches = rng.sample(patches, max_patches)

    tokens = []
    sims = []
    for p in patches:
        nbs = p.get("nearest_contextual_neighbors", [])[:5]
        for nb in nbs:
            tok = nb.get("token_str", "").strip()
            cap = nb.get("caption", "")
            sim = nb.get("similarity", 0.0)
            if tok and len(tok) > 1:
                tokens.append((tok, cap))
                sims.append(sim)
    return tokens, sims


def compute_jaccard_per_image(results_1f, results_nf, n_images, max_patches=10):
    """Compute per-image Jaccard of top-5 NNs (sampled patches).

    Matches patches by (patch_row, patch_col) instead of patch_idx to handle
    cases where grid dimensions differ between conditions (e.g., Qwen2.5-VL
    adaptive grids). Skips images with mismatched grids.
    """
    import random
    jaccards = []
    n_skipped = 0
    for i in range(min(n_images, len(results_1f), len(results_nf))):
        patches_1 = results_1f[i].get("patches", [])
        patches_n = results_nf[i].get("patches", [])

        # Check grid consistency — skip if grids differ
        grid_h_1 = results_1f[i].get("grid_h")
        grid_w_1 = results_1f[i].get("grid_w")
        grid_h_n = results_nf[i].get("grid_h")
        grid_w_n = results_nf[i].get("grid_w")
        if (grid_h_1 is not None and grid_h_n is not None
                and (grid_h_1 != grid_h_n or grid_w_1 != grid_w_n)):
            n_skipped += 1
            continue

        # Key by (row, col) for spatial alignment
        nn_1 = {(p["patch_row"], p["patch_col"]): {nb.get("token_str", "").strip()
                for nb in p.get("nearest_contextual_neighbors", [])[:5]}
                for p in patches_1}
        nn_n = {(p["patch_row"], p["patch_col"]): {nb.get("token_str", "").strip()
                for nb in p.get("nearest_contextual_neighbors", [])[:5]}
                for p in patches_n}

        common = sorted(set(nn_1.keys()) & set(nn_n.keys()))
        if not common:
            continue

        # Sample patches for speed
        rng = random.Random(42 + i)
        if len(common) > max_patches:
            common = rng.sample(common, max_patches)

        img_jaccards = []
        for key in common:
            s1, sn = nn_1[key], nn_n[key]
            if not s1 and not sn:
                continue
            j = len(s1 & sn) / len(s1 | sn) if (s1 | sn) else 1.0
            img_jaccards.append(j)
        if img_jaccards:
            jaccards.append(np.mean(img_jaccards))

    if n_skipped:
        print(f"    (skipped {n_skipped} images with mismatched grids)")
    return jaccards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--max-images", type=int, default=100,
                        help="Use first N images from each condition (ensures consistency)")
    parser.add_argument("--output", type=str, default="results/frame_sweep_rq3.json")
    parser.add_argument("--plot", type=str, default="paper/Interpreting-VideoLLMs/figures/rq3_frame_sweep.pdf")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    frame_counts = [1, 2, 4, 8, 16]
    output = {}

    for model_name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # Load single-frame baseline
        results_1f = load_results(config["single_frame_dir"], args.layer)
        if results_1f is None:
            print(f"  WARNING: no single-frame results, skipping")
            continue

        n_images = min(len(results_1f), args.max_images)
        print(f"  Using first {n_images} images")
        model_data = {}

        for nf in frame_counts:
            if nf == 1:
                results_nf = results_1f
            else:
                if nf not in config["frame_dirs"]:
                    continue
                results_nf = load_results(config["frame_dirs"][nf], args.layer)
                if results_nf is None:
                    print(f"  {nf}f: no results, skipping")
                    continue

            # POS analysis (sample up to 5000 tokens)
            counts = Counter()
            all_sims = []
            for i in range(min(n_images, len(results_nf))):
                tokens, sims = get_nn_tokens(results_nf, i)
                all_sims.extend(sims)
                for tok, cap in tokens:
                    cat = classify_token(nlp, tok, cap)
                    counts[cat] += 1

            total = sum(counts.values())
            if total == 0:
                continue

            dyn_pct = counts.get("dynamic_verb", 0) / total * 100
            sta_pct = counts.get("stative_verb", 0) / total * 100
            noun_pct = counts.get("noun", 0) / total * 100
            adj_pct = counts.get("adj", 0) / total * 100
            mean_sim = float(np.mean(all_sims)) if all_sims else 0.0

            # Jaccard with 1-frame baseline
            if nf == 1:
                mean_jaccard = 1.0
            else:
                jaccards = compute_jaccard_per_image(results_1f, results_nf, n_images)
                mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

            model_data[str(nf)] = {
                "n_images": min(n_images, len(results_nf)),
                "n_tokens": total,
                "dynamic_verb_pct": round(dyn_pct, 2),
                "stative_verb_pct": round(sta_pct, 2),
                "noun_pct": round(noun_pct, 2),
                "adj_pct": round(adj_pct, 2),
                "mean_top1_sim": round(mean_sim, 4),
                "mean_jaccard_vs_1f": round(mean_jaccard, 4),
            }

            print(f"  {nf:>2}f: dynV={dyn_pct:5.1f}%  noun={noun_pct:5.1f}%  "
                  f"sim={mean_sim:.4f}  jaccard={mean_jaccard:.4f}  ({total} tokens)")

        output[model_name] = model_data

    # Save JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        colors = {"Molmo2-8B": "#1f77b4", "Qwen2.5-VL-7B": "#ff7f0e"}
        markers = {"Molmo2-8B": "o", "Qwen2.5-VL-7B": "s"}

        # Map total frames N → preceding frames (N//2) for x-axis.
        # The studied frame is always the middle frame (index N//2); due to causal
        # attention in the decoder, only the N//2 preceding frames provide context.
        preceding_counts = [n // 2 for n in frame_counts]  # [0, 1, 2, 4, 8]

        for model_name, model_data in output.items():
            nfs = sorted([int(k) for k in model_data.keys()])
            preceding = [n // 2 for n in nfs]
            dyn = [model_data[str(n)]["dynamic_verb_pct"] for n in nfs]
            sim = [model_data[str(n)]["mean_top1_sim"] for n in nfs]
            jac = [model_data[str(n)]["mean_jaccard_vs_1f"] for n in nfs]
            c = colors[model_name]
            m = markers[model_name]

            axes[0].plot(preceding, dyn, f"-{m}", color=c, label=model_name, markersize=7)
            axes[1].plot(preceding, sim, f"-{m}", color=c, label=model_name, markersize=7)
            axes[2].plot(preceding, jac, f"-{m}", color=c, label=model_name, markersize=7)

        axes[0].set_xlabel("Preceding frames")
        axes[0].set_ylabel("Dynamic verb (%)")
        axes[0].set_title("(a) Dynamic verb fraction")
        axes[0].set_xticks(preceding_counts)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Preceding frames")
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_title("(b) Top-1 NN similarity")
        axes[1].set_xticks(preceding_counts)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel("Preceding frames")
        axes[2].set_ylabel("Jaccard with 0-preceding baseline")
        axes[2].set_title("(c) NN overlap vs. no-context baseline")
        axes[2].set_xticks(preceding_counts)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {args.plot}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
