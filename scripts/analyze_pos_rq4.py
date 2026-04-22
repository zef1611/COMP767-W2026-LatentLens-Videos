#!/usr/bin/env python3
"""RQ4 Option B: POS emergence curves across layers.

For each model × condition (single-frame vs multi-frame), computes POS
distribution at every indexed layer. Produces per-layer percentages and
paired Wilcoxon signed-rank tests at each layer (single vs multi-frame).

Reuses POS taxonomy from analyze_pos_all.py (Vendler stative verbs,
TimeML temporal adverbs).

Usage:
    python scripts/analyze_pos_rq4.py \
        --single-molmo2 results/molmo2cap_frames_500_molmo2/ \
        --multi-molmo2 results/molmo2cap_videos_500_molmo2/ \
        --single-qwen results/molmo2cap_frames_500_qwen25vl/ \
        --multi-qwen results/molmo2cap_videos_500_qwen25vl/ \
        --output results/pos_emergence_rq4.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

import spacy

# ── Reuse POS taxonomy from analyze_pos_all.py ──────────────────────────────

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

POS_CATEGORIES = ["dynamic_verb", "stative_verb", "temporal_adv", "noun", "adj", "other"]


def find_token_in_sentence(nlp, sentence: str, token_str: str):
    """POS-tag token_str within its source sentence context.

    Returns (full_word, pos_tag) or None.
    """
    token_str = token_str.strip()
    if not token_str or not sentence:
        return None

    doc = nlp(sentence)
    token_lower = token_str.lower().strip()

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() == token_lower:
            return (tok.text, tok.pos_)

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if token_lower in tok.text.lower() and len(token_lower) >= 3:
            return (tok.text, tok.pos_)

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() in token_lower and len(tok.text) >= 3:
            return (tok.text, tok.pos_)

    low_sent = sentence.lower()
    idx = low_sent.find(token_lower)
    if idx >= 0:
        for tok in doc:
            if tok.idx <= idx < tok.idx + len(tok.text):
                if not tok.is_punct and not tok.is_space:
                    return (tok.text, tok.pos_)

    return None


def classify_word(word: str, pos: str) -> str:
    """Classify a word into POS category."""
    lemma = word.lower()
    if pos == "VERB":
        return "stative_verb" if lemma in STATIVE_VERBS else "dynamic_verb"
    if pos == "ADV":
        return "temporal_adv" if lemma in TEMPORAL_ADVERBS else "other_adv"
    if pos in ("NOUN", "PROPN"):
        return "noun"
    if pos == "ADJ":
        return "adj"
    return "other"


def discover_layers(results_dir: Path) -> list:
    """Find all available layer indices from JSON filenames."""
    layers = []
    for p in results_dir.glob("latentlens_layer*.json"):
        try:
            layer = int(p.stem.replace("latentlens_layer", ""))
            layers.append(layer)
        except ValueError:
            continue
    return sorted(layers)


def analyze_pos_per_layer(results_dir: Path, nlp, sample_size: int = 5000,
                          seed: int = 42, top_k: int = 5):
    """Compute POS distribution at each available layer.

    Returns {layer: Counter({category: count})}.
    """
    layers = discover_layers(results_dir)
    if not layers:
        print(f"  WARNING: no layer files in {results_dir}")
        return {}

    pos_by_layer = {}
    for layer in layers:
        random.seed(seed)
        path = results_dir / f"latentlens_layer{layer}.json"
        with open(path) as f:
            data = json.load(f)

        token_caption_pairs = []
        for img in data["results"]:
            for patch in img["patches"]:
                nbs = patch.get("nearest_contextual_neighbors", [])
                for nb in nbs[:top_k]:
                    tok = nb.get("token_str", "").strip()
                    cap = nb.get("caption", "")
                    if tok and len(tok) > 1:
                        token_caption_pairs.append((tok, cap))

        sampled = random.sample(token_caption_pairs, min(sample_size, len(token_caption_pairs)))
        counts = Counter()
        for tok, cap in sampled:
            result = find_token_in_sentence(nlp, cap, tok)
            if result is not None:
                word, pos = result
                counts[classify_word(word, pos)] += 1
        pos_by_layer[layer] = counts

    return pos_by_layer


def analyze_pos_per_image_per_layer(results_dir: Path, nlp, top_k: int = 5,
                                    patches_per_image: int = 10, seed: int = 42):
    """Compute per-image dynamic verb fraction at each layer for paired tests.

    Samples `patches_per_image` patches per image (not all) to keep runtime
    manageable (~500 images × 10 patches × 5 NNs = 25K spaCy calls per layer).

    Returns {layer: {image_idx: dynamic_verb_fraction}}.
    """
    layers = discover_layers(results_dir)
    result = {}

    for layer in layers:
        random.seed(seed)
        path = results_dir / f"latentlens_layer{layer}.json"
        with open(path) as f:
            data = json.load(f)

        per_image = {}
        for img in data["results"]:
            img_idx = img["image_idx"]
            patches = img["patches"]

            # Sample patches for this image
            sampled_patches = random.sample(patches, min(patches_per_image, len(patches)))

            n_dynamic = 0
            n_total = 0
            for patch in sampled_patches:
                nbs = patch.get("nearest_contextual_neighbors", [])
                for nb in nbs[:top_k]:
                    tok = nb.get("token_str", "").strip()
                    cap = nb.get("caption", "")
                    if tok and len(tok) > 1:
                        res = find_token_in_sentence(nlp, cap, tok)
                        if res is not None:
                            n_total += 1
                            if classify_word(res[0], res[1]) == "dynamic_verb":
                                n_dynamic += 1
            per_image[img_idx] = n_dynamic / max(n_total, 1)
        result[layer] = per_image

    return result


def counts_to_pct(counts: Counter) -> dict:
    """Convert counts to percentages."""
    total = sum(counts.values())
    if total == 0:
        return {cat: 0.0 for cat in POS_CATEGORIES}
    return {cat: round(counts.get(cat, 0) / total * 100, 2) for cat in POS_CATEGORIES}


def main():
    parser = argparse.ArgumentParser(description="RQ4: POS emergence curves across layers")
    parser.add_argument("--single-molmo2", type=Path, default=Path("results/molmo2cap_frames_500_molmo2"))
    parser.add_argument("--multi-molmo2", type=Path, default=Path("results/molmo2cap_videos_500_molmo2"))
    parser.add_argument("--single-qwen", type=Path, default=Path("results/molmo2cap_frames_500_qwen25vl"))
    parser.add_argument("--multi-qwen", type=Path, default=Path("results/molmo2cap_videos_500_qwen25vl"))
    parser.add_argument("--output", type=Path, default=Path("results/pos_emergence_rq4.json"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

    model_configs = {
        "molmo2": {"single": args.single_molmo2, "multi": args.multi_molmo2},
        "qwen25vl": {"single": args.single_qwen, "multi": args.multi_qwen},
    }

    output = {}

    for model_name, dirs in model_configs.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        model_output = {}

        for condition, results_dir in dirs.items():
            if not results_dir.exists():
                print(f"  {condition}: {results_dir} not found, skipping")
                continue

            print(f"\n  --- {condition} ---")
            pos_by_layer = analyze_pos_per_layer(
                results_dir, nlp, sample_size=args.sample_size,
                seed=args.seed, top_k=args.top_k,
            )

            layer_pcts = {}
            for layer in sorted(pos_by_layer.keys()):
                pcts = counts_to_pct(pos_by_layer[layer])
                layer_pcts[str(layer)] = pcts
                total = sum(pos_by_layer[layer].values())
                print(f"  Layer {layer:>3}: dyn_verb={pcts['dynamic_verb']:>5.1f}%  "
                      f"noun={pcts['noun']:>5.1f}%  adj={pcts['adj']:>5.1f}%  "
                      f"(n={total})")

            model_output[condition] = layer_pcts

        # Paired statistical tests at each common layer
        single_layers = set(model_output.get("single", {}).keys())
        multi_layers = set(model_output.get("multi", {}).keys())
        common_layers = sorted(single_layers & multi_layers, key=int)

        if common_layers and "single" in dirs and "multi" in dirs:
            print(f"\n  --- Paired tests (per-image dynamic verb count) ---")
            per_img_single = analyze_pos_per_image_per_layer(
                dirs["single"], nlp, top_k=args.top_k)
            per_img_multi = analyze_pos_per_image_per_layer(
                dirs["multi"], nlp, top_k=args.top_k)

            stats_by_layer = {}
            for layer_str in common_layers:
                layer = int(layer_str)
                if layer not in per_img_single or layer not in per_img_multi:
                    continue

                s_data = per_img_single[layer]
                m_data = per_img_multi[layer]
                common_imgs = sorted(set(s_data.keys()) & set(m_data.keys()))

                if len(common_imgs) < 10:
                    continue

                diffs = [m_data[i] - s_data[i] for i in common_imgs]
                if any(d != 0 for d in diffs):
                    stat, p = stats.wilcoxon(diffs)
                else:
                    stat, p = 0.0, 1.0

                mean_single = float(np.mean([s_data[i] for i in common_imgs]))
                mean_multi = float(np.mean([m_data[i] for i in common_imgs]))

                stats_by_layer[layer_str] = {
                    "n_paired": len(common_imgs),
                    "mean_single": round(mean_single, 3),
                    "mean_multi": round(mean_multi, 3),
                    "delta": round(mean_multi - mean_single, 3),
                    "wilcoxon_stat": float(stat),
                    "wilcoxon_p": float(p),
                }
                print(f"  Layer {layer:>3}: single={mean_single:.2f} multi={mean_multi:.2f} "
                      f"delta={mean_multi-mean_single:+.2f} p={p:.4e}")

            model_output["paired_tests"] = stats_by_layer

        output[model_name] = model_output

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
