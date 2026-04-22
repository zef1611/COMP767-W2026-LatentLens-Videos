#!/usr/bin/env python3
"""RQ5: Per-object POS consistency analysis across frames.

For each tracked object (present in ALL frames), collects NN tokens from
the LatentLens results, POS-tags them, and measures whether the POS
distribution is consistent across frames.

Usage:
    python scripts/rq5/analyze_object_pos.py \\
        --patch-map results/rq5_patch_map_molmo2_4f.json \\
        --results-dir results/pvsg_100_molmo2_4f_allframes/ \\
        --n-frames 4 --layer 24 \\
        --output results/rq5_object_pos_molmo2_4f.json
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Import POS utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_pos_rq3 import classify_word


# Cached spaCy parse — avoids re-parsing the same caption thousands of times
_nlp_global = None

@lru_cache(maxsize=65536)
def _parse_sentence(sentence: str):
    """Cache spaCy doc parse by sentence string."""
    return _nlp_global(sentence)


def find_token_in_sentence_cached(nlp, sentence: str, token_str: str):
    """POS-tag token_str within its source sentence context, with cached parsing."""
    global _nlp_global
    _nlp_global = nlp

    token_str = token_str.strip()
    if not token_str or not sentence:
        return None

    doc = _parse_sentence(sentence)
    token_lower = token_str.lower().strip()

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() == token_lower:
            return (tok.text, tok.pos_, tok.lemma_)

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if token_lower in tok.text.lower() and len(token_lower) >= 3:
            return (tok.text, tok.pos_, tok.lemma_)

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.text.lower() in token_lower and len(tok.text) >= 3:
            return (tok.text, tok.pos_, tok.lemma_)

    return None

POS_CATEGORIES = ["noun", "adj", "dynamic_verb", "stative_verb",
                   "temporal_adv", "other_adv", "other"]


def js_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two POS distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0

    total_p = sum(p.values())
    total_q = sum(q.values())
    if total_p == 0 or total_q == 0:
        return 0.0

    jsd = 0.0
    for k in all_keys:
        pk = p.get(k, 0) / total_p
        qk = q.get(k, 0) / total_q
        mk = (pk + qk) / 2
        if pk > 0 and mk > 0:
            jsd += 0.5 * pk * np.log2(pk / mk)
        if qk > 0 and mk > 0:
            jsd += 0.5 * qk * np.log2(qk / mk)

    return float(jsd)


def get_dominant_pos(counts: dict) -> str:
    """Return the POS category with highest count."""
    if not counts:
        return "none"
    return max(counts, key=counts.get)


def build_patch_lookup(results_data: dict, is_1f: bool) -> dict:
    """Build lookup: (video_name, frame_idx, patch_row, patch_col) -> patch_data.

    For 1f spatial results (flat format), frame_idx is always 0.
    For allframes results, uses the nested frames structure.
    """
    lookup = {}
    for video_data in results_data["results"]:
        video_name = video_data.get("video_name",
                                     video_data["image_path"].replace(".jpg", ""))

        if is_1f:
            # Flat patch format
            for patch in video_data.get("patches", []):
                key = (video_name, 0, patch["patch_row"], patch["patch_col"])
                lookup[key] = patch
        else:
            # Allframes nested format
            for frame_entry in video_data.get("frames", []):
                frame_idx = frame_entry["frame_idx"]
                for patch in frame_entry.get("patches", []):
                    key = (video_name, frame_idx, patch["patch_row"], patch["patch_col"])
                    lookup[key] = patch

    return lookup


def analyze_object(
    video_entry: dict,
    obj: dict,
    patch_lookup: dict,
    nlp,
) -> dict:
    """Analyze POS consistency for one tracked object across frames."""
    video_name = video_entry["video_name"]
    obj_id = obj["object_id"]

    per_frame_pos = {}
    per_frame_tokens = {}
    total_patches = 0

    for frame_data in video_entry["frames"]:
        frame_idx = frame_data["frame_idx"]
        frame_counts = Counter()
        frame_tokens = []

        # Find patches assigned to this object in this frame
        obj_patches = [a for a in frame_data["patch_assignments"]
                       if a["object_id"] == obj_id]

        for assignment in obj_patches:
            key = (video_name, frame_idx,
                   assignment["patch_row"], assignment["patch_col"])
            patch_data = patch_lookup.get(key)
            if patch_data is None:
                continue

            total_patches += 1
            neighbors = patch_data.get("nearest_contextual_neighbors", [])
            for nb in neighbors:
                token_str = nb.get("token_str", "")
                caption = nb.get("caption", "")
                result = find_token_in_sentence_cached(nlp, caption, token_str)
                if result is None:
                    continue
                word, pos, lemma = result
                category = classify_word(word, pos, lemma)
                frame_counts[category] += 1
                frame_tokens.append({
                    "token": token_str.strip(),
                    "pos": category,
                })

        per_frame_pos[str(frame_idx)] = dict(frame_counts)
        per_frame_tokens[str(frame_idx)] = frame_tokens

    if total_patches == 0:
        return None

    # Aggregate POS across all frames
    aggregate = Counter()
    for counts in per_frame_pos.values():
        aggregate.update(counts)

    dominant_pos = get_dominant_pos(dict(aggregate))

    # Mode stability: is dominant POS the same in every frame?
    frame_dominants = [get_dominant_pos(counts)
                       for counts in per_frame_pos.values() if counts]
    mode_stable = len(set(frame_dominants)) <= 1 if frame_dominants else True

    # JS divergence between all pairs of frames
    frame_dists = [counts for counts in per_frame_pos.values() if counts]
    js_values = []
    for i in range(len(frame_dists)):
        for j in range(i + 1, len(frame_dists)):
            js_values.append(js_divergence(frame_dists[i], frame_dists[j]))

    js_mean = float(np.mean(js_values)) if js_values else 0.0

    return {
        "video_name": video_name,
        "object_id": obj_id,
        "category": obj["category"],
        "is_thing": obj["is_thing"],
        "n_patches_total": total_patches,
        "per_frame_pos": per_frame_pos,
        "aggregate_pos": dict(aggregate),
        "dominant_pos": dominant_pos,
        "mode_stable": mode_stable,
        "js_divergence_mean": round(js_mean, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Per-object POS consistency analysis"
    )
    parser.add_argument("--patch-map", type=Path, required=True,
                        help="Patch-object mapping JSON from map_masks_to_patches.py")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="LatentLens results directory")
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")

    # Load patch map
    with open(args.patch_map) as f:
        patch_map = json.load(f)
    log.info(f"Loaded patch map: {patch_map['n_videos']} videos, "
             f"{patch_map['total_trackable_objects']} objects")

    # Load LatentLens results
    results_path = args.results_dir / f"latentlens_layer{args.layer}.json"
    with open(results_path) as f:
        results_data = json.load(f)

    is_1f = (args.n_frames == 1)
    patch_lookup = build_patch_lookup(results_data, is_1f)
    log.info(f"Built patch lookup with {len(patch_lookup)} entries")

    # Analyze each tracked object
    per_object_results = []
    by_category = defaultdict(lambda: {
        "count": 0, "mode_stable_count": 0,
        "js_divs": [], "pos_totals": Counter()
    })
    thing_counts = Counter()
    stuff_counts = Counter()
    thing_total = 0
    stuff_total = 0

    for video_entry in patch_map["videos"]:
        for obj in video_entry["objects_in_all_frames"]:
            result = analyze_object(video_entry, obj, patch_lookup, nlp)
            if result is None:
                continue

            per_object_results.append(result)

            # Aggregate by category
            cat = result["category"]
            by_category[cat]["count"] += 1
            if result["mode_stable"]:
                by_category[cat]["mode_stable_count"] += 1
            by_category[cat]["js_divs"].append(result["js_divergence_mean"])
            by_category[cat]["pos_totals"].update(result["aggregate_pos"])

            # Thing vs stuff
            agg = result["aggregate_pos"]
            agg_total = sum(agg.values())
            if agg_total > 0:
                if result["is_thing"]:
                    thing_counts.update(agg)
                    thing_total += agg_total
                else:
                    stuff_counts.update(agg)
                    stuff_total += agg_total

    # Build category summary
    category_summary = {}
    for cat, info in sorted(by_category.items(), key=lambda x: -x[1]["count"]):
        n = info["count"]
        category_summary[cat] = {
            "count": n,
            "dominant_pos": get_dominant_pos(dict(info["pos_totals"])),
            "mode_stability": round(info["mode_stable_count"] / n, 3) if n else 0,
            "mean_js_div": round(float(np.mean(info["js_divs"])), 4) if info["js_divs"] else 0,
        }

    # Thing vs stuff summary
    thing_pcts = {k: round(v / thing_total * 100, 1) for k, v in thing_counts.items()} if thing_total else {}
    stuff_pcts = {k: round(v / stuff_total * 100, 1) for k, v in stuff_counts.items()} if stuff_total else {}

    # Overall summary
    n_total = len(per_object_results)
    n_stable = sum(1 for r in per_object_results if r["mode_stable"])
    all_js = [r["js_divergence_mean"] for r in per_object_results]

    output = {
        "n_frames": args.n_frames,
        "layer": args.layer,
        "per_object": per_object_results,
        "by_category": category_summary,
        "by_is_thing": {
            "thing": {"pos_pct": thing_pcts, "n_tokens": thing_total},
            "stuff": {"pos_pct": stuff_pcts, "n_tokens": stuff_total},
        },
        "summary": {
            "n_objects_tracked": n_total,
            "overall_mode_stability": round(n_stable / n_total, 3) if n_total else 0,
            "overall_mean_js_div": round(float(np.mean(all_js)), 4) if all_js else 0,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {args.output}")
    log.info(f"  {n_total} objects analyzed")
    log.info(f"  Mode stability: {output['summary']['overall_mode_stability']:.1%}")
    log.info(f"  Mean JS divergence: {output['summary']['overall_mean_js_div']:.4f}")

    # Print category table
    print(f"\n{'Category':<20} {'N':>5} {'Dominant':>12} {'Stability':>10} {'JS-div':>8}")
    print("-" * 60)
    for cat, info in sorted(category_summary.items(), key=lambda x: -x[1]["count"])[:15]:
        print(f"{cat:<20} {info['count']:>5} {info['dominant_pos']:>12} "
              f"{info['mode_stability']:>9.1%} {info['mean_js_div']:>8.4f}")


if __name__ == "__main__":
    main()
