#!/usr/bin/env python3
"""RQ4: Generate figures for POS emergence curves and provenance heatmaps.

Reads results from analyze_pos_rq4.py and analyze_provenance_rq4.py,
produces publication-quality PDF figures for the paper.

Usage:
    python scripts/plot_rq4.py \
        --pos results/pos_emergence_rq4.json \
        --provenance results/provenance_rq4.json \
        --output-dir paper/Interpreting-VideoLLMs/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paper-quality settings
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_DISPLAY = {"molmo2": "Molmo2-8B", "qwen25vl": "Qwen2.5-VL-7B"}
CONDITION_STYLES = {
    "single": {"color": "#2196F3", "ls": "-", "marker": "o", "label": "Single-frame"},
    "multi": {"color": "#F44336", "ls": "--", "marker": "s", "label": "Multi-frame"},
}
POS_COLORS = {
    "dynamic_verb": "#E53935",
    "noun": "#1E88E5",
    "adj": "#43A047",
    "stative_verb": "#FB8C00",
    "temporal_adv": "#8E24AA",
}
POS_LABELS = {
    "dynamic_verb": "Dynamic verb",
    "noun": "Noun",
    "adj": "Adjective",
    "stative_verb": "Stative verb",
    "temporal_adv": "Temporal adverb",
}


def plot_pos_emergence(pos_data: dict, output_path: Path):
    """Plot POS emergence curves: one panel per model, lines for single/multi."""
    models = sorted(pos_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 2.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        ax.set_title(MODEL_DISPLAY.get(model, model))

        for condition, style in CONDITION_STYLES.items():
            if condition not in pos_data[model]:
                continue
            layer_data = pos_data[model][condition]
            layers = sorted(layer_data.keys(), key=int)
            x = [int(l) for l in layers]
            y = [layer_data[l].get("dynamic_verb", 0) for l in layers]

            ax.plot(x, y, color=style["color"], ls=style["ls"],
                    marker=style["marker"], markersize=4, linewidth=1.5,
                    label=style["label"])

        ax.set_xlabel("Layer")
        ax.set_xticks([int(l) for l in layers])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Dynamic verb (%)")
    axes[0].legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved POS emergence plot → {output_path}")


def plot_pos_all_categories(pos_data: dict, output_path: Path):
    """Plot all POS categories: 2x2 grid (model x condition)."""
    models = sorted(pos_data.keys())
    conditions = ["single", "multi"]

    fig, axes = plt.subplots(len(models), len(conditions),
                             figsize=(7, 2.8 * len(models)), sharey=True, sharex=True)
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    categories = ["dynamic_verb", "noun", "adj", "stative_verb"]

    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            if condition not in pos_data[model]:
                ax.set_visible(False)
                continue

            layer_data = pos_data[model][condition]
            layers = sorted(layer_data.keys(), key=int)
            x = [int(l) for l in layers]

            for cat in categories:
                y = [layer_data[l].get(cat, 0) for l in layers]
                ax.plot(x, y, color=POS_COLORS[cat], marker=".", markersize=3,
                        linewidth=1.2, label=POS_LABELS[cat])

            ax.set_title(f"{MODEL_DISPLAY.get(model, model)} ({CONDITION_STYLES[condition]['label']})")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel("Percentage (%)")
            if i == len(models) - 1:
                ax.set_xlabel("Layer")

    axes[0][0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved POS all-categories plot → {output_path}")


def plot_provenance_heatmaps(prov_data: dict, output_path: Path):
    """Plot provenance heatmaps: x=query layer, y=source layer, 2x2 grid."""
    models = sorted(prov_data.keys())
    conditions = ["single", "multi"]

    fig, axes = plt.subplots(len(models), len(conditions),
                             figsize=(7, 3 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for i, model in enumerate(models):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            if condition not in prov_data[model]:
                ax.set_visible(False)
                continue

            layer_data = prov_data[model][condition]
            query_layers = sorted(layer_data.keys(), key=int)

            # Collect all source layers
            all_src_layers = set()
            for ql in query_layers:
                for sl in layer_data[ql].get("provenance_dist", {}).keys():
                    all_src_layers.add(int(sl))
            src_layers = sorted(all_src_layers)

            if not src_layers or not query_layers:
                ax.set_visible(False)
                continue

            # Build heatmap matrix [src_layer x query_layer]
            matrix = np.zeros((len(src_layers), len(query_layers)))
            for qi, ql in enumerate(query_layers):
                dist = layer_data[ql].get("provenance_dist", {})
                total = sum(int(v) for v in dist.values())
                if total == 0:
                    continue
                for si, sl in enumerate(src_layers):
                    matrix[si, qi] = int(dist.get(str(sl), 0)) / total * 100

            im = ax.imshow(matrix, aspect="auto", origin="lower",
                          cmap="YlOrRd", vmin=0)
            ax.set_xticks(range(len(query_layers)))
            ax.set_xticklabels([str(int(l)) for l in query_layers], fontsize=7)
            ax.set_yticks(range(len(src_layers)))
            ax.set_yticklabels([str(l) for l in src_layers], fontsize=7)

            ax.set_title(f"{MODEL_DISPLAY.get(model, model)} ({CONDITION_STYLES[condition]['label']})")
            if j == 0:
                ax.set_ylabel("NN source layer")
            if i == len(models) - 1:
                ax.set_xlabel("Query layer")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved provenance heatmap → {output_path}")


def plot_provenance_summary(prov_data: dict, output_path: Path):
    """Plot same-layer % and mean provenance across query layers."""
    models = sorted(prov_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 2.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        ax.set_title(MODEL_DISPLAY.get(model, model))

        for condition, style in CONDITION_STYLES.items():
            if condition not in prov_data[model]:
                continue
            layer_data = prov_data[model][condition]
            layers = sorted(layer_data.keys(), key=int)
            x = [int(l) for l in layers]
            y = [layer_data[l].get("same_layer_pct", 0) for l in layers]

            ax.plot(x, y, color=style["color"], ls=style["ls"],
                    marker=style["marker"], markersize=4, linewidth=1.5,
                    label=style["label"])

        ax.set_xlabel("Query layer")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Same-layer NN (%)")
    axes[0].legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved provenance summary → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ4: Generate figures")
    parser.add_argument("--pos", type=Path, default=Path("results/pos_emergence_rq4.json"))
    parser.add_argument("--provenance", type=Path, default=Path("results/provenance_rq4.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/Interpreting-VideoLLMs/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pos.exists():
        with open(args.pos) as f:
            pos_data = json.load(f)
        plot_pos_emergence(pos_data, args.output_dir / "rq4_pos_emergence.pdf")
        plot_pos_all_categories(pos_data, args.output_dir / "rq4_pos_all_categories.pdf")
    else:
        print(f"POS data not found at {args.pos}, skipping POS plots")

    if args.provenance.exists():
        with open(args.provenance) as f:
            prov_data = json.load(f)
        plot_provenance_heatmaps(prov_data, args.output_dir / "rq4_provenance_heatmap.pdf")
        plot_provenance_summary(prov_data, args.output_dir / "rq4_provenance_summary.pdf")
    else:
        print(f"Provenance data not found at {args.provenance}, skipping provenance plots")


if __name__ == "__main__":
    main()
