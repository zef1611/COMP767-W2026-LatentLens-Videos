#!/usr/bin/env python3
"""RQ5: Cross-condition summary and aggregate figures.

Aggregates per-object POS consistency results across n_frames conditions
and both models. Produces summary JSON and paper figures.

Usage:
    python scripts/rq5/summarize_and_plot.py \\
        --results-dir results/ \\
        --output results/rq5_summary.json
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS = {
    "molmo2": "Molmo2-8B",
    "qwen25vl": "Qwen2.5-VL-7B",
}
N_FRAMES_LIST = [1, 2, 4, 8, 16]
POS_CATEGORIES = ["noun", "adj", "dynamic_verb", "stative_verb",
                   "temporal_adv", "other_adv", "other"]

FIGURES_DIR = Path("paper/Interpreting-VideoLLMs/figures")


def load_all_results(results_dir: Path) -> dict:
    """Load all rq5_object_pos_{model}_{n}f.json files."""
    data = {}
    for model_key, model_name in MODELS.items():
        data[model_name] = {}
        for nf in N_FRAMES_LIST:
            path = results_dir / f"rq5_object_pos_{model_key}_{nf}f.json"
            if path.exists():
                with open(path) as f:
                    data[model_name][nf] = json.load(f)
                log.info(f"Loaded {path}: {data[model_name][nf]['summary']['n_objects_tracked']} objects")
            else:
                log.warning(f"Not found: {path}")
    return data


def compute_summary(data: dict) -> dict:
    """Compute cross-condition summary with statistical tests."""
    summary = {}

    for model_name, nf_data in data.items():
        model_summary = {
            "consistency_vs_nframes": {},
            "category_breakdown": {},
        }

        for nf, results in sorted(nf_data.items()):
            s = results["summary"]
            model_summary["consistency_vs_nframes"][nf] = {
                "n_objects": s["n_objects_tracked"],
                "mode_stability": s["overall_mode_stability"],
                "mean_js_div": s["overall_mean_js_div"],
            }

            # Merge category data
            for cat, info in results["by_category"].items():
                if cat not in model_summary["category_breakdown"]:
                    model_summary["category_breakdown"][cat] = {}
                model_summary["category_breakdown"][cat][nf] = info

        # Statistical tests: compare JS divergence between frame counts
        # Paired by object (same video_name + object_id across conditions)
        tests = {}
        nf_keys = sorted(nf_data.keys())
        for i in range(len(nf_keys)):
            for j in range(i + 1, len(nf_keys)):
                nf_a, nf_b = nf_keys[i], nf_keys[j]
                if nf_a == 1 or nf_b == 1:
                    continue  # Skip 1f (no JS divergence)

                # Build paired data
                objs_a = {(r["video_name"], r["object_id"]): r["js_divergence_mean"]
                          for r in nf_data[nf_a].get("per_object", [])}
                objs_b = {(r["video_name"], r["object_id"]): r["js_divergence_mean"]
                          for r in nf_data[nf_b].get("per_object", [])}

                paired_keys = set(objs_a.keys()) & set(objs_b.keys())
                if len(paired_keys) < 5:
                    continue

                vals_a = [objs_a[k] for k in paired_keys]
                vals_b = [objs_b[k] for k in paired_keys]

                try:
                    stat, p_val = stats.wilcoxon(vals_a, vals_b)
                    tests[f"{nf_a}f_vs_{nf_b}f"] = {
                        "n_paired": len(paired_keys),
                        "mean_a": round(float(np.mean(vals_a)), 4),
                        "mean_b": round(float(np.mean(vals_b)), 4),
                        "wilcoxon_stat": round(float(stat), 2),
                        "wilcoxon_p": round(float(p_val), 6),
                    }
                except Exception:
                    pass

        model_summary["statistical_tests"] = tests
        summary[model_name] = model_summary

    return summary


def plot_consistency_vs_nframes(data: dict, output_dir: Path):
    """Line plot: POS mode stability vs n_frames, one line per model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, nf_data in data.items():
        nfs = sorted(nf_data.keys())
        # Mode stability (skip 1f if present — no temporal consistency)
        nfs_multi = [nf for nf in nfs if nf > 1]
        stabilities = [nf_data[nf]["summary"]["overall_mode_stability"] for nf in nfs_multi]
        js_divs = [nf_data[nf]["summary"]["overall_mean_js_div"] for nf in nfs_multi]

        ax1.plot(nfs_multi, stabilities, "o-", label=model_name, markersize=8)
        ax2.plot(nfs_multi, js_divs, "o-", label=model_name, markersize=8)

    ax1.set_xlabel("Number of Frames")
    ax1.set_ylabel("POS Mode Stability")
    ax1.set_title("(a) Mode Stability vs. Frame Count")
    ax1.legend()
    ax1.set_xticks([2, 4, 8, 16])
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of Frames")
    ax2.set_ylabel("Mean JS Divergence")
    ax2.set_title("(b) POS Distribution Divergence vs. Frame Count")
    ax2.legend()
    ax2.set_xticks([2, 4, 8, 16])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "rq5_consistency_vs_nframes"
    plt.savefig(f"{out_path}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out_path}.pdf + .png")


def plot_category_pos_heatmap(data: dict, output_dir: Path):
    """Heatmap: rows=PVSG categories, cols=POS tags, cells=proportion."""
    # Use the largest n_frames condition for the richest data
    for model_name, nf_data in data.items():
        if not nf_data:
            continue
        max_nf = max(nf_data.keys())
        results = nf_data[max_nf]

        categories = []
        pos_matrix = []
        for cat, info in sorted(results["by_category"].items(),
                                key=lambda x: -x[1]["count"]):
            if info["count"] < 3:
                continue
            categories.append(f"{cat} (n={info['count']})")

            # Get POS distribution from per_object data
            cat_objs = [r for r in results["per_object"]
                        if r["category"] == cat]
            total_counts = Counter()
            for obj in cat_objs:
                total_counts.update(obj["aggregate_pos"])

            total = sum(total_counts.values())
            row = [total_counts.get(pos, 0) / total * 100 if total > 0 else 0
                   for pos in POS_CATEGORIES]
            pos_matrix.append(row)

        if not categories:
            continue

        matrix = np.array(pos_matrix)
        fig, ax = plt.subplots(figsize=(10, max(6, len(categories) * 0.4)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(POS_CATEGORIES)))
        ax.set_xticklabels(POS_CATEGORIES, rotation=45, ha="right")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)

        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(POS_CATEGORIES)):
                val = matrix[i, j]
                if val > 1:
                    color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="% of NN tokens")
        ax.set_title(f"POS Distribution by PVSG Category — {model_name} ({max_nf}f)")
        plt.tight_layout()

        model_key = "molmo2" if "Molmo2" in model_name else "qwen25vl"
        out_path = output_dir / f"rq5_category_heatmap_{model_key}"
        plt.savefig(f"{out_path}.pdf", dpi=200, bbox_inches="tight")
        plt.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved {out_path}.pdf + .png")


def plot_thing_vs_stuff(data: dict, output_dir: Path):
    """Grouped bar chart: thing vs stuff POS distributions."""
    models_with_data = {k: v for k, v in data.items() if v}
    if not models_with_data:
        return

    fig, axes = plt.subplots(1, len(models_with_data), figsize=(7 * len(models_with_data), 5))
    if len(models_with_data) == 1:
        axes = [axes]

    for ax, (model_name, nf_data) in zip(axes, models_with_data.items()):
        max_nf = max(nf_data.keys())
        results = nf_data[max_nf]
        thing_pcts = results["by_is_thing"]["thing"]["pos_pct"]
        stuff_pcts = results["by_is_thing"]["stuff"]["pos_pct"]

        x = np.arange(len(POS_CATEGORIES))
        width = 0.35

        thing_vals = [thing_pcts.get(pos, 0) for pos in POS_CATEGORIES]
        stuff_vals = [stuff_pcts.get(pos, 0) for pos in POS_CATEGORIES]

        ax.bar(x - width/2, thing_vals, width, label="Thing", color="#4CAF50", alpha=0.8)
        ax.bar(x + width/2, stuff_vals, width, label="Stuff", color="#2196F3", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(POS_CATEGORIES, rotation=45, ha="right")
        ax.set_ylabel("% of NN tokens")
        ax.set_title(f"{model_name} ({max_nf}f)")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("POS Distribution: Thing vs. Stuff Objects", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / "rq5_thing_vs_stuff"
    plt.savefig(f"{out_path}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out_path}.pdf + .png")


def plot_dynamic_verbs_vs_nframes(data: dict, output_dir: Path):
    """Line plot: dynamic verb % vs n_frames, split by thing/stuff, per model.

    Shows how increasing temporal context affects the proportion of
    dynamic-verb NN tokens for tracked objects.
    """
    models_with_multi = {k: v for k, v in data.items()
                         if len(v) > 1}  # need >1 frame condition
    if not models_with_multi:
        log.warning("Not enough data for dynamic verb plot")
        return

    fig, axes = plt.subplots(1, len(models_with_multi),
                             figsize=(7 * len(models_with_multi), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, (model_name, nf_data) in zip(axes, models_with_multi.items()):
        nfs = sorted(nf_data.keys())
        nfs_multi = [nf for nf in nfs if nf > 1]

        # Compute dynamic verb % for thing vs stuff at each nf
        for label, is_thing, color, marker in [
            ("Thing", True, "#E53935", "o"),
            ("Stuff", False, "#1E88E5", "s"),
        ]:
            dyn_pcts = []
            all_pcts = []  # all verb (dynamic + stative) for reference
            for nf in nfs_multi:
                results = nf_data[nf]
                objs = [r for r in results["per_object"]
                        if r["is_thing"] == is_thing]
                total_counts = Counter()
                for obj in objs:
                    total_counts.update(obj["aggregate_pos"])
                total = sum(total_counts.values())
                if total > 0:
                    dyn_pcts.append(total_counts.get("dynamic_verb", 0) / total * 100)
                    all_pcts.append(
                        (total_counts.get("dynamic_verb", 0) +
                         total_counts.get("stative_verb", 0)) / total * 100
                    )
                else:
                    dyn_pcts.append(0)
                    all_pcts.append(0)

            ax.plot(nfs_multi, dyn_pcts, f"{marker}-", label=f"{label} — dynamic",
                    color=color, markersize=8, linewidth=2)
            ax.plot(nfs_multi, all_pcts, f"{marker}--", label=f"{label} — all verbs",
                    color=color, markersize=6, linewidth=1, alpha=0.5)

        ax.set_xlabel("Number of Frames")
        ax.set_ylabel("% of NN Tokens")
        ax.set_title(model_name)
        ax.set_xticks([2, 4, 8, 16])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Dynamic Verb Proportion vs. Frame Count (Thing vs. Stuff)", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / "rq5_dynamic_verbs_vs_nframes"
    plt.savefig(f"{out_path}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out_path}.pdf + .png")


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Cross-condition summary and figures"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/rq5_summary.json"))
    args = parser.parse_args()

    data = load_all_results(args.results_dir)

    # Check we have any data
    has_data = any(nf_data for nf_data in data.values())
    if not has_data:
        log.error("No RQ5 results found!")
        return

    # Compute summary
    summary = compute_summary(data)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {args.output}")

    # Generate figures
    fig_dir = FIGURES_DIR / "rq5"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_consistency_vs_nframes(data, fig_dir)
    plot_category_pos_heatmap(data, fig_dir)
    plot_thing_vs_stuff(data, fig_dir)
    plot_dynamic_verbs_vs_nframes(data, fig_dir)

    log.info("Done!")


if __name__ == "__main__":
    main()
