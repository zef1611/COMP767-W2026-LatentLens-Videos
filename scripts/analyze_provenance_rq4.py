#!/usr/bin/env python3
"""RQ4 Option D: NN layer provenance analysis across layers.

For each query layer, analyzes the `contextual_layer` field of top-k NNs
to understand where nearest neighbors come from. Compares single-frame
vs multi-frame conditions using KL divergence and paired tests.

Key insight: LatentLens searches across ALL indexed layers globally.
If multi-frame input shifts which layers the best NNs come from, that
reveals cross-frame information flow through the model.

Usage:
    python scripts/analyze_provenance_rq4.py \
        --single-molmo2 results/molmo2cap_frames_500_molmo2/ \
        --multi-molmo2 results/molmo2cap_videos_500_molmo2/ \
        --single-qwen results/molmo2cap_frames_500_qwen25vl/ \
        --multi-qwen results/molmo2cap_videos_500_qwen25vl/ \
        --output results/provenance_rq4.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import rel_entr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def compute_provenance(results_dir: Path, top_k: int = 5):
    """Compute NN provenance distribution per query layer.

    Returns {layer: {"provenance_dist": Counter, "mean_provenance": float,
                     "per_image": {img_idx: mean_contextual_layer}}}.
    """
    layers = discover_layers(results_dir)
    if not layers:
        return {}

    result = {}
    for layer in layers:
        path = results_dir / f"latentlens_layer{layer}.json"
        with open(path) as f:
            data = json.load(f)

        all_ctx_layers = []
        per_image = {}

        for img in data["results"]:
            img_idx = img["image_idx"]
            img_ctx_layers = []

            for patch in img["patches"]:
                nbs = patch.get("nearest_contextual_neighbors", [])
                for nb in nbs[:top_k]:
                    ctx_layer = nb.get("contextual_layer")
                    if ctx_layer is not None:
                        all_ctx_layers.append(ctx_layer)
                        img_ctx_layers.append(ctx_layer)

            if img_ctx_layers:
                per_image[img_idx] = float(np.mean(img_ctx_layers))

        if not all_ctx_layers:
            continue

        dist = Counter(all_ctx_layers)
        result[layer] = {
            "provenance_dist": dist,
            "mean_provenance": float(np.mean(all_ctx_layers)),
            "std_provenance": float(np.std(all_ctx_layers)),
            "n_neighbors": len(all_ctx_layers),
            "same_layer_pct": round(dist.get(layer, 0) / len(all_ctx_layers) * 100, 2),
            "per_image": per_image,
        }

    return result


def kl_divergence(dist_a: Counter, dist_b: Counter) -> float:
    """Compute KL(A || B) with smoothing."""
    all_keys = sorted(set(dist_a.keys()) | set(dist_b.keys()))
    if not all_keys:
        return 0.0

    total_a = sum(dist_a.values())
    total_b = sum(dist_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0

    # Laplace smoothing
    eps = 1e-8
    p = np.array([dist_a.get(k, 0) / total_a + eps for k in all_keys])
    q = np.array([dist_b.get(k, 0) / total_b + eps for k in all_keys])
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(rel_entr(p, q)))


def main():
    parser = argparse.ArgumentParser(description="RQ4: NN layer provenance analysis")
    parser.add_argument("--single-molmo2", type=Path, default=Path("results/molmo2cap_frames_500_molmo2"))
    parser.add_argument("--multi-molmo2", type=Path, default=Path("results/molmo2cap_videos_500_molmo2"))
    parser.add_argument("--single-qwen", type=Path, default=Path("results/molmo2cap_frames_500_qwen25vl"))
    parser.add_argument("--multi-qwen", type=Path, default=Path("results/molmo2cap_videos_500_qwen25vl"))
    parser.add_argument("--output", type=Path, default=Path("results/provenance_rq4.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

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
            prov = compute_provenance(results_dir, top_k=args.top_k)

            # Serialize (Counter → dict for JSON)
            layer_data = {}
            for layer in sorted(prov.keys()):
                p = prov[layer]
                layer_data[str(layer)] = {
                    "provenance_dist": {str(k): v for k, v in sorted(p["provenance_dist"].items())},
                    "mean_provenance": round(p["mean_provenance"], 2),
                    "std_provenance": round(p["std_provenance"], 2),
                    "n_neighbors": p["n_neighbors"],
                    "same_layer_pct": p["same_layer_pct"],
                }
                print(f"  Layer {layer:>3}: mean_src={p['mean_provenance']:.1f}  "
                      f"same_layer={p['same_layer_pct']:.1f}%  "
                      f"n={p['n_neighbors']}")

            model_output[condition] = layer_data

        # KL divergence and paired tests between conditions
        single_prov = compute_provenance(dirs["single"], top_k=args.top_k) if dirs["single"].exists() else {}
        multi_prov = compute_provenance(dirs["multi"], top_k=args.top_k) if dirs["multi"].exists() else {}

        common_layers = sorted(set(single_prov.keys()) & set(multi_prov.keys()))

        if common_layers:
            print(f"\n  --- Cross-condition comparison ---")
            kl_by_layer = {}
            paired_tests = {}

            for layer in common_layers:
                sp = single_prov[layer]
                mp = multi_prov[layer]

                # KL divergence between provenance distributions
                kl = kl_divergence(sp["provenance_dist"], mp["provenance_dist"])
                kl_by_layer[str(layer)] = round(kl, 4)

                # Paired test on per-image mean provenance
                common_imgs = sorted(set(sp["per_image"].keys()) & set(mp["per_image"].keys()))
                if len(common_imgs) >= 10:
                    s_vals = [sp["per_image"][i] for i in common_imgs]
                    m_vals = [mp["per_image"][i] for i in common_imgs]
                    diffs = [m - s for s, m in zip(s_vals, m_vals)]

                    if any(d != 0 for d in diffs):
                        stat, p = stats.wilcoxon(diffs)
                    else:
                        stat, p = 0.0, 1.0

                    paired_tests[str(layer)] = {
                        "n_paired": len(common_imgs),
                        "mean_single": round(float(np.mean(s_vals)), 2),
                        "mean_multi": round(float(np.mean(m_vals)), 2),
                        "delta": round(float(np.mean(diffs)), 2),
                        "wilcoxon_stat": float(stat),
                        "wilcoxon_p": float(p),
                    }

                print(f"  Layer {layer:>3}: KL={kl:.4f}  "
                      f"same_layer: {sp['same_layer_pct']:.1f}% → {mp['same_layer_pct']:.1f}%  "
                      f"mean_src: {sp['mean_provenance']:.1f} → {mp['mean_provenance']:.1f}")

            model_output["kl_divergence"] = kl_by_layer
            model_output["paired_tests"] = paired_tests

        output[model_name] = model_output

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
