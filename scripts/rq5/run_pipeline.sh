#!/bin/bash
# RQ5 Pipeline: Object-level POS consistency across frames
#
# Tracks objects in PVSG videos across multiple frames, maps masks to patches,
# and analyzes POS consistency of NN tokens for tracked objects.
#
# Prerequisites:
#   - conda env: molmo2
#   - PVSG data at $SCRATCH/latentlens/pvsg/
#
# Usage:
#   bash scripts/rq5/run_pipeline.sh [stage]
#   Stages: prepare, latentlens, map, analyze, summarize, visualize, all

set -euo pipefail

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh"
conda activate molmo2

STAGE="${1:-all}"

# Only run new all-frames for 2+ frames; 1f uses existing spatial results
N_FRAMES_MULTI=(2 4 8 16)
N_FRAMES_ALL=(1 2 4 8 16)

LAYER=24

declare -A MODEL_IDS=(
    ["molmo2"]="allenai/Molmo2-8B"
    ["qwen25vl"]="Qwen/Qwen2.5-VL-7B-Instruct"
)

declare -A DTYPES=(
    ["molmo2"]="bfloat16"
    ["qwen25vl"]="float16"
)

declare -A INDEX_IDS=(
    ["molmo2"]="McGill-NLP/contextual_embeddings-molmo2-8b"
    ["qwen25vl"]="McGill-NLP/contextual_embeddings-qwen2.5-vl-7b"
)

SCRATCH_DIR="${SCRATCH:-/network/scratch/l/leh}/latentlens"
VIDEOS_DIR="$SCRATCH_DIR/pvsg_videos_100"
FRAMES_DIR="$SCRATCH_DIR/pvsg_frames_100"
MODELS=("molmo2" "qwen25vl")

# ── Stage 0: Prepare qualified PVSG videos ─────────────────────────────────
run_prepare() {
    echo "=== Stage 0: Prepare qualified PVSG video subset ==="
    python3 scripts/rq5/prepare_pvsg_qualified.py --seed 42 --n-videos 100
    echo "=== Preparation complete ==="
}

# ── Stage 1: All-frames LatentLens (2+ frames, both models) ───────────────
run_latentlens() {
    echo "=== Stage 1: LatentLens all-frames (frames: ${N_FRAMES_MULTI[*]}) ==="

    for model in "${MODELS[@]}"; do
        for nf in "${N_FRAMES_MULTI[@]}"; do
            output_dir="results/pvsg_100_${model}_${nf}f_allframes"
            if [ -f "$output_dir/latentlens_layer${LAYER}.json" ]; then
                echo "  Skipping $model ${nf}f (already exists)"
                continue
            fi

            echo "  Running $model ${nf}f..."
            python3 scripts/rq5/run_latentlens_allframes.py \
                --model "${MODEL_IDS[$model]}" \
                --index "${INDEX_IDS[$model]}" \
                --videos-dir "$VIDEOS_DIR" \
                --frames-dir "$FRAMES_DIR" \
                --output-dir "$output_dir" \
                --n-frames "$nf" \
                --layers "$LAYER" \
                --dtype "${DTYPES[$model]}" \
                --device cuda:0 \
                --num-videos 100 \
                2>&1 | tee "logs/rq5_latentlens_${model}_${nf}f.log"
        done
    done

    echo "=== LatentLens complete ==="
}

# ── Stage 2: Map masks to patches ─────────────────────────────────────────
run_map() {
    echo "=== Stage 2: Map PVSG masks to patches ==="

    for model in "${MODELS[@]}"; do
        for nf in "${N_FRAMES_ALL[@]}"; do
            output="results/rq5_patch_map_${model}_${nf}f.json"
            if [ -f "$output" ]; then
                echo "  Skipping $model ${nf}f (already exists)"
                continue
            fi

            if [ "$nf" -eq 1 ]; then
                # Use existing 1f spatial results
                results_dir="results/pvsg_100_${model}_1f_spatial"
            else
                results_dir="results/pvsg_100_${model}_${nf}f_allframes"
            fi

            if [ ! -d "$results_dir" ]; then
                echo "  Skipping $model ${nf}f (no results at $results_dir)"
                continue
            fi

            echo "  Mapping $model ${nf}f..."
            python3 scripts/rq5/map_masks_to_patches.py \
                --results-dir "$results_dir" \
                --n-frames "$nf" \
                --layer "$LAYER" \
                --output "$output"
        done
    done

    echo "=== Mask mapping complete ==="
}

# ── Stage 3: POS consistency analysis ─────────────────────────────────────
run_analyze() {
    echo "=== Stage 3: POS consistency analysis ==="

    for model in "${MODELS[@]}"; do
        for nf in "${N_FRAMES_ALL[@]}"; do
            patch_map="results/rq5_patch_map_${model}_${nf}f.json"
            output="results/rq5_object_pos_${model}_${nf}f.json"

            if [ -f "$output" ]; then
                echo "  Skipping $model ${nf}f (already exists)"
                continue
            fi

            if [ ! -f "$patch_map" ]; then
                echo "  Skipping $model ${nf}f (no patch map)"
                continue
            fi

            if [ "$nf" -eq 1 ]; then
                results_dir="results/pvsg_100_${model}_1f_spatial"
            else
                results_dir="results/pvsg_100_${model}_${nf}f_allframes"
            fi

            echo "  Analyzing $model ${nf}f..."
            python3 scripts/rq5/analyze_object_pos.py \
                --patch-map "$patch_map" \
                --results-dir "$results_dir" \
                --n-frames "$nf" \
                --layer "$LAYER" \
                --output "$output"
        done
    done

    echo "=== POS analysis complete ==="
}

# ── Stage 4: Summary and plots ────────────────────────────────────────────
run_summarize() {
    echo "=== Stage 4: Cross-condition summary and plots ==="
    python3 scripts/rq5/summarize_and_plot.py \
        --results-dir results/ \
        --output results/rq5_summary.json
    echo "=== Summary complete ==="
}

# ── Stage 5: Example visualizations ──────────────────────────────────────
run_visualize() {
    echo "=== Stage 5: Example visualizations ==="

    for model in "${MODELS[@]}"; do
        # Use largest available n_frames for best examples
        for nf in 16 8 4; do
            patch_map="results/rq5_patch_map_${model}_${nf}f.json"
            pos_results="results/rq5_object_pos_${model}_${nf}f.json"
            results_dir="results/pvsg_100_${model}_${nf}f_allframes"

            if [ -f "$patch_map" ] && [ -f "$pos_results" ] && [ -d "$results_dir" ]; then
                echo "  Visualizing $model ${nf}f..."
                python3 scripts/rq5/visualize_examples.py \
                    --patch-map "$patch_map" \
                    --pos-results "$pos_results" \
                    --results-dir "$results_dir" \
                    --n-frames "$nf" \
                    --model-key "$model" \
                    --n-examples 4
                break  # Only visualize the largest available
            fi
        done
    done

    echo "=== Visualization complete ==="
}

# ── Main ──────────────────────────────────────────────────────────────────
mkdir -p logs

case "$STAGE" in
    prepare)    run_prepare ;;
    latentlens) run_latentlens ;;
    map)        run_map ;;
    analyze)    run_analyze ;;
    summarize)  run_summarize ;;
    visualize)  run_visualize ;;
    all)
        run_prepare
        run_latentlens
        run_map
        run_analyze
        run_summarize
        run_visualize
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 {prepare|latentlens|map|analyze|summarize|visualize|all}"
        exit 1
        ;;
esac

echo "=== RQ5 Pipeline done ==="
