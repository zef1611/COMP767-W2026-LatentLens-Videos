#!/usr/bin/env python3
"""Paired comparison of RQ2 (single-frame) vs RQ3 (multi-frame) LatentLens results.

Compares nearest neighbors for the same image/patch between single-frame and
multi-frame conditions. Produces JSON with:
  1. POS distribution delta per model
  2. Top-5 NN overlap (Jaccard) per model
  3. Top-1 cosine similarity shift per model
  4. Statistical tests (Wilcoxon signed-rank)
  5. Qualitative examples where NNs shift from nouns to verbs

Usage:
    python scripts/compare_rq2_rq3.py \
        --rq2-judge results/judge_evaluation_rq2.json \
        --rq3-judge results/judge_evaluation_rq3.json \
        --rq2-pos results/pos_analysis_rq2.json \
        --rq3-pos results/pos_analysis_rq3.json \
        --output results/comparison_rq2_rq3.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

import spacy

# Reuse taxonomy
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

TEMPORAL_ADVERBS = {
    "slowly", "quickly", "rapidly", "fast", "swiftly", "briskly",
    "gradually", "steadily", "hastily", "hurriedly",
    "suddenly", "immediately", "instantly", "eventually", "finally",
    "already", "soon", "recently", "previously", "afterwards",
    "meanwhile", "simultaneously", "continuously", "repeatedly",
    "temporarily", "permanently", "briefly", "momentarily",
    "often", "always", "never", "sometimes", "rarely", "frequently",
    "occasionally", "constantly", "regularly", "periodically",
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
    if matched.pos_ == "ADV":
        return "temporal_adv" if (lemma_lower in TEMPORAL_ADVERBS or token_lower in TEMPORAL_ADVERBS) else "other_adv"
    if matched.pos_ in ("NOUN", "PROPN"):
        return "noun"
    if matched.pos_ == "ADJ":
        return "adj"
    return "other"


def get_nn_tokens(patch):
    """Extract top-5 NN token strings from a patch."""
    nbs = patch.get("neighbors", [])[:5]
    return [nb.get("token_str", "").strip() for nb in nbs if nb.get("token_str", "").strip()]


def compute_jaccard(tokens_a, tokens_b):
    """Jaccard similarity between two lists of tokens."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def get_top1_similarity(patch):
    """Get the cosine similarity of the top-1 NN."""
    nbs = patch.get("neighbors", [])
    if nbs:
        return nbs[0].get("similarity", 0.0)
    return 0.0


def pair_patches(rq2_patches, rq3_patches):
    """Pair patches by image_idx (both use same normalized coordinates)."""
    rq2_by_idx = {p["image_idx"]: p for p in rq2_patches}
    rq3_by_idx = {p["image_idx"]: p for p in rq3_patches}
    common = sorted(set(rq2_by_idx.keys()) & set(rq3_by_idx.keys()))
    return [(rq2_by_idx[i], rq3_by_idx[i]) for i in common]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq2-judge", type=str, default="results/judge_evaluation_rq2.json")
    parser.add_argument("--rq3-judge", type=str, default="results/judge_evaluation_rq3.json")
    parser.add_argument("--rq2-pos", type=str, default="results/pos_analysis_rq2.json")
    parser.add_argument("--rq3-pos", type=str, default="results/pos_analysis_rq3.json")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Only use patches from the first N images for consistency")
    parser.add_argument("--output", type=str, default="results/comparison_rq2_rq3.json")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

    with open(args.rq2_judge) as f:
        rq2_data = json.load(f)
    with open(args.rq3_judge) as f:
        rq3_data = json.load(f)

    # Load POS summaries if available
    rq2_pos, rq3_pos = {}, {}
    if Path(args.rq2_pos).exists():
        with open(args.rq2_pos) as f:
            rq2_pos = json.load(f)
    if Path(args.rq3_pos).exists():
        with open(args.rq3_pos) as f:
            rq3_pos = json.load(f)

    # Only compare models present in both
    common_models = sorted(set(rq2_data.keys()) & set(rq3_data.keys()))
    print(f"Comparing models: {common_models}")

    output = {}

    for model_name in common_models:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # Get vidframes patches from RQ2 (not pixmo — RQ3 only has video)
        # Filter to first max_images for consistency across conditions
        rq2_patches = [p for p in rq2_data[model_name].get("patches", [])
                       if p.get("ds_key") == "vidframes" and p["image_idx"] < args.max_images]
        rq3_patches = [p for p in rq3_data[model_name].get("patches", [])
                       if p["image_idx"] < args.max_images]

        pairs = pair_patches(rq2_patches, rq3_patches)
        print(f"  Paired patches: {len(pairs)}")

        if not pairs:
            print("  No paired patches found, skipping")
            continue

        # 1. NN overlap (Jaccard)
        jaccards = []
        for p2, p3 in pairs:
            t2 = get_nn_tokens(p2)
            t3 = get_nn_tokens(p3)
            jaccards.append(compute_jaccard(t2, t3))
        mean_jaccard = float(np.mean(jaccards))
        print(f"  Mean Jaccard overlap: {mean_jaccard:.3f}")

        # 2. Top-1 similarity shift
        sim2 = [get_top1_similarity(p2) for p2, _ in pairs]
        sim3 = [get_top1_similarity(p3) for _, p3 in pairs]
        mean_sim2 = float(np.mean(sim2))
        mean_sim3 = float(np.mean(sim3))
        sim_stat, sim_p = stats.wilcoxon([s3 - s2 for s2, s3 in zip(sim2, sim3)])
        print(f"  Top-1 sim: single={mean_sim2:.4f}, multi={mean_sim3:.4f}, "
              f"delta={mean_sim3-mean_sim2:+.4f}, p={sim_p:.4e}")

        # 3. Per-patch POS classification and delta
        rq2_verb_counts = []
        rq3_verb_counts = []
        rq2_total_counts = Counter()
        rq3_total_counts = Counter()

        for p2, p3 in pairs:
            # Classify NNs for RQ2 patch
            n2_verbs = 0
            for nb in p2.get("neighbors", [])[:5]:
                cat = classify_token(nlp, nb.get("token_str", ""), nb.get("caption", ""))
                rq2_total_counts[cat] += 1
                if cat == "dynamic_verb":
                    n2_verbs += 1
            rq2_verb_counts.append(n2_verbs)

            # Classify NNs for RQ3 patch
            n3_verbs = 0
            for nb in p3.get("neighbors", [])[:5]:
                cat = classify_token(nlp, nb.get("token_str", ""), nb.get("caption", ""))
                rq3_total_counts[cat] += 1
                if cat == "dynamic_verb":
                    n3_verbs += 1
            rq3_verb_counts.append(n3_verbs)

        # POS distribution percentages
        rq2_total = sum(rq2_total_counts.values())
        rq3_total = sum(rq3_total_counts.values())

        def pct(counts, key, total):
            return counts.get(key, 0) / total * 100 if total > 0 else 0

        pos_categories = ["dynamic_verb", "stative_verb", "temporal_adv", "noun", "adj"]
        pos_delta = {}
        for cat in pos_categories:
            p2 = pct(rq2_total_counts, cat, rq2_total)
            p3 = pct(rq3_total_counts, cat, rq3_total)
            pos_delta[cat] = {"single_frame": round(p2, 2), "multi_frame": round(p3, 2),
                              "delta": round(p3 - p2, 2)}
            print(f"  {cat}: {p2:.1f}% → {p3:.1f}% (delta={p3-p2:+.1f}pp)")

        # Statistical test on per-patch verb counts
        verb_diffs = [v3 - v2 for v2, v3 in zip(rq2_verb_counts, rq3_verb_counts)]
        if any(d != 0 for d in verb_diffs):
            verb_stat, verb_p = stats.wilcoxon(verb_diffs)
        else:
            verb_stat, verb_p = 0.0, 1.0
        print(f"  Verb count Wilcoxon: stat={verb_stat:.1f}, p={verb_p:.4e}")

        # 4. Qualitative examples: patches where NNs shifted toward verbs
        examples = []
        for (p2, p3), j in zip(pairs, jaccards):
            t2 = get_nn_tokens(p2)
            t3 = get_nn_tokens(p3)
            # Look for cases where multi-frame has more verbs
            cats2 = [classify_token(nlp, nb.get("token_str", ""), nb.get("caption", ""))
                     for nb in p2.get("neighbors", [])[:5]]
            cats3 = [classify_token(nlp, nb.get("token_str", ""), nb.get("caption", ""))
                     for nb in p3.get("neighbors", [])[:5]]
            n_verbs_2 = sum(1 for c in cats2 if c == "dynamic_verb")
            n_verbs_3 = sum(1 for c in cats3 if c == "dynamic_verb")
            if n_verbs_3 > n_verbs_2 and j < 0.5:
                examples.append({
                    "image_idx": p2["image_idx"],
                    "image_path": p2.get("image_path", ""),
                    "single_frame_nns": t2,
                    "multi_frame_nns": t3,
                    "single_frame_verbs": n_verbs_2,
                    "multi_frame_verbs": n_verbs_3,
                    "jaccard": round(j, 3),
                })
        examples.sort(key=lambda x: x["multi_frame_verbs"] - x["single_frame_verbs"], reverse=True)
        print(f"  Verb-gain examples: {len(examples)} (showing top 5)")
        for ex in examples[:5]:
            print(f"    img={ex['image_path']}: {ex['single_frame_nns']} → {ex['multi_frame_nns']}")

        output[model_name] = {
            "n_paired": len(pairs),
            "mean_jaccard": mean_jaccard,
            "top1_similarity": {
                "single_frame": mean_sim2,
                "multi_frame": mean_sim3,
                "delta": mean_sim3 - mean_sim2,
                "wilcoxon_p": sim_p,
            },
            "pos_delta": pos_delta,
            "verb_count_test": {
                "wilcoxon_stat": float(verb_stat),
                "wilcoxon_p": float(verb_p),
                "mean_single": float(np.mean(rq2_verb_counts)),
                "mean_multi": float(np.mean(rq3_verb_counts)),
            },
            "rq2_pos_counts": dict(rq2_total_counts),
            "rq3_pos_counts": dict(rq3_total_counts),
            "verb_gain_examples": examples[:10],
        }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
