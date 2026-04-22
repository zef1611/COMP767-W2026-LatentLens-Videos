#!/usr/bin/env python3
"""RQ4: Combine POS emergence + provenance results into a summary.

Reads outputs from analyze_pos_rq4.py and analyze_provenance_rq4.py,
produces a combined JSON with per-layer statistics and a paper-ready
summary table.

Usage:
    python scripts/compare_rq4.py \
        --pos results/pos_emergence_rq4.json \
        --provenance results/provenance_rq4.json \
        --output results/comparison_rq4.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RQ4: Combined summary")
    parser.add_argument("--pos", type=Path, default=Path("results/pos_emergence_rq4.json"))
    parser.add_argument("--provenance", type=Path, default=Path("results/provenance_rq4.json"))
    parser.add_argument("--output", type=Path, default=Path("results/comparison_rq4.json"))
    args = parser.parse_args()

    with open(args.pos) as f:
        pos_data = json.load(f)
    with open(args.provenance) as f:
        prov_data = json.load(f)

    output = {}

    for model in sorted(set(pos_data.keys()) & set(prov_data.keys())):
        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"{'='*60}")

        pos_single = pos_data[model].get("single", {})
        pos_multi = pos_data[model].get("multi", {})
        pos_paired = pos_data[model].get("paired_tests", {})

        prov_single = prov_data[model].get("single", {})
        prov_multi = prov_data[model].get("multi", {})
        prov_kl = prov_data[model].get("kl_divergence", {})
        prov_paired = prov_data[model].get("paired_tests", {})

        common_layers = sorted(
            set(pos_single.keys()) & set(pos_multi.keys()),
            key=int,
        )

        # Build per-layer summary table
        table = []
        print(f"\n  {'Layer':>5} | {'DynVerb S':>9} {'DynVerb M':>9} {'Δ':>6} {'p':>10} | "
              f"{'SameLayer S':>11} {'SameLayer M':>11} {'KL':>6}")
        print(f"  {'-'*80}")

        for layer_str in common_layers:
            row = {"layer": int(layer_str)}

            # POS data
            if layer_str in pos_single and layer_str in pos_multi:
                dv_s = pos_single[layer_str].get("dynamic_verb", 0)
                dv_m = pos_multi[layer_str].get("dynamic_verb", 0)
                row["dynamic_verb_single"] = dv_s
                row["dynamic_verb_multi"] = dv_m
                row["dynamic_verb_delta"] = round(dv_m - dv_s, 2)
            else:
                dv_s = dv_m = 0

            # POS paired test
            if layer_str in pos_paired:
                row["pos_wilcoxon_p"] = pos_paired[layer_str]["wilcoxon_p"]
                row["pos_wilcoxon_stat"] = pos_paired[layer_str]["wilcoxon_stat"]
                p_val = pos_paired[layer_str]["wilcoxon_p"]
            else:
                p_val = None

            # Provenance data
            sl_s = prov_single.get(layer_str, {}).get("same_layer_pct", 0)
            sl_m = prov_multi.get(layer_str, {}).get("same_layer_pct", 0)
            kl = prov_kl.get(layer_str, 0)
            row["same_layer_pct_single"] = sl_s
            row["same_layer_pct_multi"] = sl_m
            row["kl_divergence"] = kl

            # Provenance paired test
            if layer_str in prov_paired:
                row["prov_wilcoxon_p"] = prov_paired[layer_str]["wilcoxon_p"]
                row["prov_delta"] = prov_paired[layer_str]["delta"]

            table.append(row)

            p_str = f"{p_val:.2e}" if p_val is not None else "N/A"
            print(f"  {int(layer_str):>5} | {dv_s:>8.1f}% {dv_m:>8.1f}% {dv_m-dv_s:>+5.1f} {p_str:>10} | "
                  f"{sl_s:>10.1f}% {sl_m:>10.1f}% {kl:>6.4f}")

        # Emergence layer: first layer where dynamic verbs > 5%
        emergence_single = None
        emergence_multi = None
        for layer_str in common_layers:
            if pos_single.get(layer_str, {}).get("dynamic_verb", 0) > 5 and emergence_single is None:
                emergence_single = int(layer_str)
            if pos_multi.get(layer_str, {}).get("dynamic_verb", 0) > 5 and emergence_multi is None:
                emergence_multi = int(layer_str)

        print(f"\n  Emergence layer (>5% dynamic verbs): "
              f"single={emergence_single}, multi={emergence_multi}")

        output[model] = {
            "per_layer": table,
            "emergence_layer_single": emergence_single,
            "emergence_layer_multi": emergence_multi,
        }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
