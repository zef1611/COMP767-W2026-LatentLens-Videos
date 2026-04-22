#!/usr/bin/env python3
"""Plot interpretability rate across frame counts for RQ3.

Combines judge results from:
  - 1f: results/judge_evaluation_rq2.json (vidframes, first 100)
  - 4f: results/judge_evaluation_rq3.json (first 100)
  - 2f, 8f, 16f: results/judge_frame_sweep.json

Usage:
    python scripts/plot_rq3_interpretability.py \
        --output paper/Interpreting-VideoLLMs/figures/rq3_interpretability.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLORS = {
    "Molmo2-8B": "#2ca02c",
    "Qwen2.5-VL-7B": "#d62728",
}
MODEL_ORDER = ["Molmo2-8B", "Qwen2.5-VL-7B"]
MARKERS = {"Molmo2-8B": "o", "Qwen2.5-VL-7B": "s"}
MAX_IMAGES = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq2-judge", default="results/judge_evaluation_rq2.json")
    parser.add_argument("--rq3-judge", default="results/judge_evaluation_rq3.json")
    parser.add_argument("--sweep-judge", default="results/judge_frame_sweep.json")
    parser.add_argument("--output", default="paper/Interpreting-VideoLLMs/figures/rq3_interpretability.pdf")
    args = parser.parse_args()

    rq2 = json.load(open(args.rq2_judge))
    rq3 = json.load(open(args.rq3_judge))
    sweep = json.load(open(args.sweep_judge))

    # Build data: {model: {nf: pct_interpretable}}
    data = {}
    for model in MODEL_ORDER:
        data[model] = {}

        # 1f from RQ2 vidframes (first 100)
        patches_1f = [p for p in rq2[model]["patches"]
                      if p.get("ds_key") == "vidframes" and p["image_idx"] < MAX_IMAGES]
        n = len(patches_1f)
        interp = sum(1 for p in patches_1f if p.get("interpretable"))
        data[model][1] = 100 * interp / n if n else 0

        # 4f from RQ3 (first 100)
        patches_4f = [p for p in rq3[model]["patches"]
                      if p["image_idx"] < MAX_IMAGES]
        n = len(patches_4f)
        interp = sum(1 for p in patches_4f if p.get("interpretable"))
        data[model][4] = 100 * interp / n if n else 0

        # 2f, 8f, 16f from sweep
        for key, v in sweep.items():
            if v["model"] == model:
                nf = v["n_frames"]
                data[model][nf] = v["pct_interpretable"]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    frame_counts = [1, 2, 4, 8, 16]

    for model in MODEL_ORDER:
        nfs = sorted(data[model].keys())
        pcts = [data[model][nf] for nf in nfs]
        ax.plot(nfs, pcts, f"-{MARKERS[model]}", color=MODEL_COLORS[model],
                label=model, markersize=8, linewidth=2)

        # Add percentage labels
        for nf, pct in zip(nfs, pcts):
            ax.annotate(f"{pct:.0f}%", (nf, pct),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold",
                        color=MODEL_COLORS[model])

    ax.set_xlabel("Number of input frames", fontsize=12)
    ax.set_ylabel("Interpretable patches (%)", fontsize=12)
    ax.set_xticks(frame_counts)
    ax.set_ylim(50, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
