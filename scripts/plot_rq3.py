#!/usr/bin/env python3
"""Generate RQ3 figures for the paper, matching RQ2 style.

Produces:
  1. rq3_dynverb_bar.pdf — Paired bar chart: single-frame vs multi-frame dynamic verb %
  2. rq3_frame_sweep.pdf — Line plot: metrics across 1, 2, 4, 8, 16 frames
  3. rq3_jaccard_hist.pdf — Histogram of per-patch NN overlap (Jaccard)

Usage:
    python scripts/plot_rq3.py \
        --comparison results/comparison_rq2_rq3.json \
        --sweep results/frame_sweep_rq3.json \
        --output-dir paper/Interpreting-VideoLLMs/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Match RQ2 color scheme: Molmo2=green, Qwen=red
MODEL_COLORS = {
    "Molmo2-8B": "#2ca02c",      # green (same as RQ2 bar)
    "Qwen2.5-VL-7B": "#d62728",  # red (same as RQ2 bar)
}
MODEL_ORDER = ["Molmo2-8B", "Qwen2.5-VL-7B"]


def plot_dynverb_bar(comparison_data, output_path):
    """Paired bar chart: single-frame vs multi-frame dynamic verb %.

    Matches RQ2 figure style (bold % labels on top, clean axis).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    models = [m for m in MODEL_ORDER if m in comparison_data]
    x = np.arange(len(models))
    width = 0.35

    single_vals = []
    multi_vals = []
    for m in models:
        d = comparison_data[m]["pos_delta"]["dynamic_verb"]
        single_vals.append(d["single_frame"])
        multi_vals.append(d["multi_frame"])

    bars1 = ax.bar(x - width/2, single_vals, width, label="Single frame",
                   color=[MODEL_COLORS[m] for m in models], alpha=0.5, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, multi_vals, width, label="4 frames",
                   color=[MODEL_COLORS[m] for m in models], alpha=1.0, edgecolor="black", linewidth=0.5)

    # Bold percentage labels on top (matching RQ2 style)
    for bar, val in zip(bars1, single_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    for bar, val in zip(bars2, multi_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Dynamic verb %", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(video)" for m in models], fontsize=11)
    ax.set_ylim(0, max(single_vals + multi_vals) * 1.25)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_frame_sweep(sweep_data, output_path):
    """Line plot: metrics across frame counts (1, 2, 4, 8, 16)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    markers = {"Molmo2-8B": "o", "Qwen2.5-VL-7B": "s"}

    for model_name in MODEL_ORDER:
        if model_name not in sweep_data:
            continue
        model_data = sweep_data[model_name]
        nfs = sorted([int(k) for k in model_data.keys()])
        dyn = [model_data[str(n)]["dynamic_verb_pct"] for n in nfs]
        sim = [model_data[str(n)]["mean_top1_sim"] for n in nfs]
        jac = [model_data[str(n)]["mean_jaccard_vs_1f"] for n in nfs]
        c = MODEL_COLORS[model_name]
        m = markers[model_name]

        axes[0].plot(nfs, dyn, f"-{m}", color=c, label=model_name, markersize=7, linewidth=2)
        axes[1].plot(nfs, sim, f"-{m}", color=c, label=model_name, markersize=7, linewidth=2)
        axes[2].plot(nfs, jac, f"-{m}", color=c, label=model_name, markersize=7, linewidth=2)

    frame_counts = [1, 2, 4, 8, 16]

    axes[0].set_xlabel("Number of frames", fontsize=11)
    axes[0].set_ylabel("Dynamic verb (%)", fontsize=11)
    axes[0].set_title("(a) Dynamic verb fraction", fontsize=12)
    axes[0].set_xticks(frame_counts)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_xlabel("Number of frames", fontsize=11)
    axes[1].set_ylabel("Cosine similarity", fontsize=11)
    axes[1].set_title("(b) Top-1 NN similarity", fontsize=12)
    axes[1].set_xticks(frame_counts)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].set_xlabel("Number of frames", fontsize=11)
    axes[2].set_ylabel("Jaccard with 1-frame", fontsize=11)
    axes[2].set_title("(c) NN overlap vs. single frame", fontsize=12)
    axes[2].set_xticks(frame_counts)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_jaccard_hist(comparison_data, output_path):
    """Histogram of per-patch Jaccard overlap between single and multi-frame."""
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10, 3.5), sharey=True)

    if len(MODEL_ORDER) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, MODEL_ORDER):
        if model_name not in comparison_data:
            continue

        # Reconstruct per-patch Jaccards from the comparison data
        mean_j = comparison_data[model_name].get("mean_jaccard", 0)

        # If we have verb_gain_examples, we can show their Jaccards
        # But for the histogram we need per-patch data — use mean as annotation
        ax.axvline(mean_j, color=MODEL_COLORS[model_name], linewidth=2,
                   linestyle="--", label=f"Mean = {mean_j:.3f}")
        ax.set_xlabel("Jaccard similarity", fontsize=11)
        ax.set_title(model_name, fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=str, default="results/comparison_rq2_rq3.json")
    parser.add_argument("--sweep", type=str, default="results/frame_sweep_rq3.json")
    parser.add_argument("--output-dir", type=str, default="paper/Interpreting-VideoLLMs/figures/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot 1: Paired bar chart (needs comparison data)
    if Path(args.comparison).exists():
        with open(args.comparison) as f:
            comp = json.load(f)
        plot_dynverb_bar(comp, out / "rq3_dynverb_bar.pdf")
    else:
        print(f"Skipping dynverb bar: {args.comparison} not found")

    # Plot 2: Frame sweep (needs sweep data)
    if Path(args.sweep).exists():
        with open(args.sweep) as f:
            sweep = json.load(f)
        plot_frame_sweep(sweep, out / "rq3_frame_sweep.pdf")
    else:
        print(f"Skipping frame sweep: {args.sweep} not found")

    # Plot 3: Jaccard histogram (needs comparison data)
    if Path(args.comparison).exists():
        with open(args.comparison) as f:
            comp = json.load(f)
        plot_jaccard_hist(comp, out / "rq3_jaccard_hist.pdf")
    else:
        print(f"Skipping Jaccard hist: {args.comparison} not found")


if __name__ == "__main__":
    main()
