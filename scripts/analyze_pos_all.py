#!/usr/bin/env python3
"""POS analysis across all 4 models × 2 data types.

Compares dynamic verb / stative verb / temporal adverb frequencies.
Uses top-5 LatentLens neighbors and POS-tags tokens in sentence context.

Usage:
    python scripts/analyze_pos_all.py
"""

import json
import random
from collections import Counter
from pathlib import Path

import spacy


# ── Stative verbs (Vendler 1957; Comrie 1976) ────────────────────────────────
# These describe states rather than actions/events. Canonical list from
# linguistics literature. Used to distinguish "is/has/knows" from dynamic
# verbs like "run/build/throw".
STATIVE_VERBS = {
    # cognition / perception
    "know", "believe", "think", "understand", "realize", "recognize",
    "remember", "forget", "imagine", "suppose", "doubt", "mean",
    "feel", "hear", "see", "smell", "taste", "notice", "perceive",
    # emotion / attitude
    "like", "love", "hate", "prefer", "want", "wish", "need", "desire",
    "fear", "envy", "mind", "care", "appreciate", "dislike",
    # possession / relation
    "have", "own", "possess", "belong", "owe", "contain", "include",
    "consist", "involve", "lack",
    # being / appearance
    "be", "exist", "seem", "appear", "look", "sound", "resemble",
    "remain", "stay", "weigh", "measure", "cost", "equal", "fit",
    # other statives
    "depend", "deserve", "matter", "owe", "suit", "tend",
    "agree", "disagree", "deny", "promise", "refuse",
    "concern", "impress", "please", "satisfy", "surprise",
}

# ── Temporal / dynamic adverbs ───────────────────────────────────────────────
# Adverbs that specifically denote temporal or manner-of-motion qualities.
# Curated from TimeML-style temporal expressions and motion linguistics.
TEMPORAL_ADVERBS = {
    # speed / manner of motion
    "slowly", "quickly", "rapidly", "fast", "swiftly", "briskly",
    "gradually", "steadily", "hastily", "hurriedly",
    # temporal sequence
    "suddenly", "immediately", "instantly", "eventually", "finally",
    "already", "soon", "recently", "previously", "afterwards",
    "meanwhile", "simultaneously", "continuously", "repeatedly",
    "temporarily", "permanently", "briefly", "momentarily",
    # frequency
    "often", "always", "never", "sometimes", "rarely", "frequently",
    "occasionally", "constantly", "regularly", "periodically",
}


def find_token_in_sentence(nlp, sentence: str, token_str: str):
    """POS-tag `token_str` within its source sentence context.

    Returns (full_word, pos_tag) or None if token cannot be located.
    spaCy POS tagging is much more accurate with sentence context than
    tagging isolated words.
    """
    token_str = token_str.strip()
    if not token_str or not sentence:
        return None

    doc = nlp(sentence)

    # Find the spaCy token that best matches the LatentLens token_str
    token_lower = token_str.lower().strip()

    # Pass 1: exact match
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() == token_lower:
            return (tok.text, tok.pos_)

    # Pass 2: query is substring of spaCy token (BPE subword → full word)
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if token_lower in tok.text.lower() and len(token_lower) >= 3:
            return (tok.text, tok.pos_)

    # Pass 3: spaCy token is substring of query (with length guard)
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() in token_lower and len(tok.text) >= 3:
            return (tok.text, tok.pos_)

    # Pass 4: character offset fallback
    low_sent = sentence.lower()
    idx = low_sent.find(token_lower)
    if idx >= 0:
        for tok in doc:
            if tok.idx <= idx < tok.idx + len(tok.text):
                if not tok.is_punct and not tok.is_space:
                    return (tok.text, tok.pos_)

    return None


def classify_word(word: str, pos: str) -> str:
    """Classify a word into: dynamic_verb, stative_verb, temporal_adv, other_adv, noun, adj, other."""
    lemma = word.lower()
    if pos == "VERB":
        if lemma in STATIVE_VERBS:
            return "stative_verb"
        return "dynamic_verb"
    if pos == "ADV":
        if lemma in TEMPORAL_ADVERBS:
            return "temporal_adv"
        return "other_adv"
    if pos in ("NOUN", "PROPN"):
        return "noun"
    if pos == "ADJ":
        return "adj"
    return "other"


def analyze_pos(results_dir: Path, layers: list, nlp, sample_size: int = 5000,
                seed: int = 42, top_k: int = 5):
    """Returns {layer: Counter({category: count})} using top-k NNs and sentence-context POS."""
    random.seed(seed)
    pos_by_layer = {}

    for layer in layers:
        path = results_dir / f"latentlens_layer{layer}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        # Collect (token_str, caption) pairs from top-k neighbors
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
                category = classify_word(word, pos)
                counts[category] += 1
        pos_by_layer[layer] = counts

    return pos_by_layer


def main():
    nlp = spacy.load("en_core_web_sm")

    configs = [
        # (label, type, dir, layers)
        ("Molmo", "image", "results/pixmo100_molmo-7b-d", [1,2,4,8,16,24,27]),
        ("Idefics3", "image", "results/pixmo100_idefics3", [1,2,4,8,16,24,31]),
        ("Molmo2", "video", "results/pixmo100_molmo2", [1,2,4,8,16,24,35]),
        ("Qwen2.5-VL", "video", "results/pixmo100_qwen25vl", [1,2,4,8,16,24,27]),
        ("Molmo", "image", "results/molmo2cap100_molmo-7b-d", [1,2,4,8,16,24,27]),
        ("Idefics3", "image", "results/molmo2cap100_idefics3", [1,2,4,8,16,24,31]),
        ("Molmo2", "video", "results/molmo2cap100_molmo2", [1,2,4,8,16,24,35]),
        ("Qwen2.5-VL", "video", "results/molmo2cap100_qwen25vl", [1,2,4,8,16,24,27]),
    ]

    print(f"\n{'Model':<14} {'Type':<7} {'Data':<12} {'Layer':>6} "
          f"{'DynVerb%':>9} {'StatVerb%':>10} {'TmpAdv%':>8} {'Noun%':>7} {'Adj%':>6}")
    print("-" * 95)

    for label, mtype, results_dir, layers in configs:
        rdir = Path(results_dir)
        if not rdir.exists():
            continue

        data_name = "PixMo" if "pixmo" in results_dir else "VidFrames"
        pos_data = analyze_pos(rdir, layers, nlp)
        if not pos_data:
            continue

        for i, layer in enumerate(sorted(pos_data.keys())):
            c = pos_data[layer]
            total = sum(c.values())
            if total == 0:
                continue
            dyn = c.get("dynamic_verb", 0) / total * 100
            sta = c.get("stative_verb", 0) / total * 100
            tmp = c.get("temporal_adv", 0) / total * 100
            noun = c.get("noun", 0) / total * 100
            adj = c.get("adj", 0) / total * 100
            prefix = f"{label:<14} {mtype:<7} {data_name:<12}" if i == 0 else " " * 35
            print(f"{prefix} {layer:>6} {dyn:>8.1f}% {sta:>9.1f}% {tmp:>7.1f}% {noun:>6.1f}% {adj:>5.1f}%")
        print()


if __name__ == "__main__":
    main()
