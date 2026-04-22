#!/usr/bin/env python3
"""Analyze average top-1 LatentLens similarity across models and layers.

Higher similarity = more interpretable visual tokens. This is a prerequisite
study replicating the basic interpretability analysis from the LatentLens paper.

Usage:
    python scripts/analyze_similarity.py
"""

import json
from collections import defaultdict
from pathlib import Path


def compute_avg_similarity(results_dir: Path, layers: list) -> dict:
    """Compute average top-1 similarity per layer.

    Returns {layer: {"avg_sim": float, "n_patches": int}}.
    """
    stats = {}
    for layer in layers:
        path = results_dir / f"latentlens_layer{layer}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        sims = []
        for img in data["results"]:
            for patch in img["patches"]:
                nbs = patch.get("nearest_contextual_neighbors", [])
                if nbs and nbs[0].get("similarity") is not None:
                    sims.append(nbs[0]["similarity"])

        if sims:
            stats[layer] = {
                "avg_sim": sum(sims) / len(sims),
                "n_patches": len(sims),
            }
    return stats


def main():
    configs = [
        # (label, results_dir, layers)
        ("Molmo (image) — PixMo", "results/pixmo100_molmo-7b-d", [1,2,4,8,16,24,26,27]),
        ("Molmo2 (video) — PixMo", "results/pixmo100_molmo2", [1,2,4,8,16,24,34,35]),
        ("Qwen2.5-VL (video) — PixMo", "results/pixmo100_qwen25vl", [1,2,4,8,16,24,26,27]),
        ("Molmo (image) — VideoFrames", "results/molmo2cap100_molmo-7b-d", [1,2,4,8,16,24,26,27]),
        ("Molmo2 (video) — VideoFrames", "results/molmo2cap100_molmo2", [1,2,4,8,16,24,34,35]),
        ("Qwen2.5-VL (video) — VideoFrames", "results/molmo2cap100_qwen25vl", [1,2,4,8,16,24,26,27]),
        # Idefics3 will be added once results are ready
        ("Idefics3 (image) — PixMo", "results/pixmo100_idefics3", [1,2,4,8,16,24,30,31]),
        ("Idefics3 (image) — VideoFrames", "results/molmo2cap100_idefics3", [1,2,4,8,16,24,30,31]),
    ]

    print(f"{'Model + Data':<40} {'Layer':>6} {'Avg Sim':>8} {'Patches':>8}")
    print("-" * 70)

    for label, results_dir, layers in configs:
        rdir = Path(results_dir)
        if not rdir.exists():
            continue
        stats = compute_avg_similarity(rdir, layers)
        if not stats:
            continue

        for i, layer in enumerate(sorted(stats.keys())):
            s = stats[layer]
            prefix = label if i == 0 else ""
            print(f"{prefix:<40} {layer:>6} {s['avg_sim']:>8.3f} {s['n_patches']:>8}")
        print()


if __name__ == "__main__":
    main()
