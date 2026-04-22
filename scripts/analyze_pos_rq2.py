#!/usr/bin/env python3
"""POS analysis for RQ2: dynamic/stative verbs and temporal adverbs.

Analyzes the LatentLens nearest neighbors from judge-evaluated patches.
Two analyses:
  (a) ALL top-5 NNs regardless of interpretability (raw POS distribution)
  (b) Only NNs from INTERPRETABLE patches (meaningful verb/temporal content)

Uses sentence-context POS tagging (spaCy on full captions).

Usage:
    python scripts/analyze_pos_rq2.py \
        --judge-results results/judge_evaluation_rq2.json \
        --output results/pos_analysis_rq2.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import spacy

# ── Stative verbs (Vendler 1957; Comrie 1976) ────────────────────────────────
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


def find_token_in_sentence(nlp, sentence: str, token_str: str):
    """POS-tag token_str within its source sentence context."""
    token_str = token_str.strip()
    if not token_str or not sentence:
        return None

    doc = nlp(sentence)
    token_lower = token_str.lower().strip()

    # Pass 1: exact match
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() == token_lower:
            return (tok.text, tok.pos_, tok.lemma_)

    # Pass 2: query is substring of spaCy token (BPE subword → full word)
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if token_lower in tok.text.lower() and len(token_lower) >= 3:
            return (tok.text, tok.pos_, tok.lemma_)

    # Pass 3: spaCy token is substring of query (with length guard)
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() in token_lower and len(tok.text) >= 3:
            return (tok.text, tok.pos_, tok.lemma_)

    # Pass 4: character offset fallback
    low_sent = sentence.lower()
    idx = low_sent.find(token_lower)
    if idx >= 0:
        for tok in doc:
            if tok.idx <= idx < tok.idx + len(tok.text):
                if not tok.is_punct and not tok.is_space:
                    return (tok.text, tok.pos_, tok.lemma_)

    return None


def classify_word(word: str, pos: str, lemma: str) -> str:
    """Classify into: dynamic_verb, stative_verb, temporal_adv, other_adv, noun, adj, other."""
    lemma_lower = lemma.lower()
    if pos == "VERB":
        if lemma_lower in STATIVE_VERBS:
            return "stative_verb"
        return "dynamic_verb"
    if pos == "ADV":
        if lemma_lower in TEMPORAL_ADVERBS or word.lower() in TEMPORAL_ADVERBS:
            return "temporal_adv"
        return "other_adv"
    if pos in ("NOUN", "PROPN"):
        return "noun"
    if pos == "ADJ":
        return "adj"
    return "other"


def analyze_patches(patches, nlp, interpretable_only=False):
    """Analyze POS distribution of top-5 NNs from judge-evaluated patches."""
    counts = Counter()
    n_patches_used = 0

    for patch in patches:
        if interpretable_only and not patch.get("interpretable", False):
            continue

        neighbors = patch.get("neighbors", [])
        if not neighbors:
            continue

        n_patches_used += 1
        for nb in neighbors[:5]:
            tok = nb.get("token_str", "").strip()
            cap = nb.get("caption", "")
            if not tok or len(tok) <= 1:
                continue

            result = find_token_in_sentence(nlp, cap, tok)
            if result is not None:
                word, pos, lemma = result
                category = classify_word(word, pos, lemma)
                counts[category] += 1

    return counts, n_patches_used


def print_table(title, rows):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Model':<18} {'Type':<7} {'N':>5} {'DynV%':>7} {'StatV%':>7} "
          f"{'TmpAdv%':>8} {'Noun%':>7} {'Adj%':>6}")
    print("-" * 70)
    for row in rows:
        c = row["counts"]
        total = sum(c.values())
        if total == 0:
            continue
        dyn = c.get("dynamic_verb", 0) / total * 100
        sta = c.get("stative_verb", 0) / total * 100
        tmp = c.get("temporal_adv", 0) / total * 100
        noun = c.get("noun", 0) / total * 100
        adj = c.get("adj", 0) / total * 100
        mtype = row.get("type", "")
        print(f"{row['model']:<18} {mtype:<7} {row['n_patches']:>5} "
              f"{dyn:>6.1f}% {sta:>6.1f}% {tmp:>7.1f}% {noun:>6.1f}% {adj:>5.1f}%")


MODEL_TYPES = {
    "Molmo-7B-D": "image",
    "Idefics3-8B": "image",
    "Molmo2-8B": "video",
    "Qwen2.5-VL-7B": "video",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-results", type=str, default="results/judge_evaluation_rq2.json")
    parser.add_argument("--output", type=str, default="results/pos_analysis_rq2.json")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

    with open(args.judge_results) as f:
        judge_data = json.load(f)

    output = {}

    # Analysis (a): ALL patches, all top-5 NNs
    rows_all = []
    for model_name in judge_data:
        patches = judge_data[model_name].get("patches", [])
        counts, n_used = analyze_patches(patches, nlp, interpretable_only=False)
        rows_all.append({
            "model": model_name,
            "type": MODEL_TYPES.get(model_name, ""),
            "n_patches": n_used,
            "counts": dict(counts),
        })
        output.setdefault(model_name, {})["all_patches"] = {
            "n_patches": n_used,
            "counts": dict(counts),
        }

    print_table("(a) ALL patches — raw POS distribution of top-5 NNs", rows_all)

    # Analysis (b): INTERPRETABLE patches only
    rows_interp = []
    for model_name in judge_data:
        patches = judge_data[model_name].get("patches", [])
        counts, n_used = analyze_patches(patches, nlp, interpretable_only=True)
        rows_interp.append({
            "model": model_name,
            "type": MODEL_TYPES.get(model_name, ""),
            "n_patches": n_used,
            "counts": dict(counts),
        })
        output[model_name]["interpretable_patches"] = {
            "n_patches": n_used,
            "counts": dict(counts),
        }

    print_table("(b) INTERPRETABLE patches only — POS of top-5 NNs", rows_interp)

    # Analysis (c): Per-dataset breakdown (interpretable only)
    print(f"\n{'='*80}")
    print(f"  (c) Per-dataset breakdown (interpretable only)")
    print(f"{'='*80}")
    print(f"{'Model':<18} {'Dataset':<12} {'N':>5} {'DynV%':>7} {'StatV%':>7} "
          f"{'TmpAdv%':>8} {'Noun%':>7}")
    print("-" * 70)

    for model_name in judge_data:
        patches = judge_data[model_name].get("patches", [])
        for ds_key in ["pixmo", "vidframes"]:
            ds_patches = [p for p in patches if p.get("ds_key") == ds_key]
            counts, n_used = analyze_patches(ds_patches, nlp, interpretable_only=True)
            total = sum(counts.values())
            if total == 0:
                continue
            dyn = counts.get("dynamic_verb", 0) / total * 100
            sta = counts.get("stative_verb", 0) / total * 100
            tmp = counts.get("temporal_adv", 0) / total * 100
            noun = counts.get("noun", 0) / total * 100
            ds_label = "PixMo" if ds_key == "pixmo" else "VidFrames"
            print(f"{model_name:<18} {ds_label:<12} {n_used:>5} "
                  f"{dyn:>6.1f}% {sta:>6.1f}% {tmp:>7.1f}% {noun:>6.1f}%")

            output[model_name].setdefault("per_dataset", {})[ds_key] = {
                "n_patches": n_used,
                "counts": dict(counts),
            }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
