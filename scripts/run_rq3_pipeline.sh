#!/bin/bash
# RQ3/RQ4 pipeline: LatentLens video → spatial patches → judge → POS → comparison
# Runs on 500 Molmo2-Cap videos, 2 VideoLLMs (Molmo2-8B, Qwen2.5-VL-7B).
# Supports multi-layer runs for RQ4 cross-layer analysis.
#
# Prerequisites:
#   - $SCRATCH/latentlens/molmo2cap_videos_500/ has MP4 videos
#   - $SCRATCH/latentlens/molmo2cap_frames_500/ has corresponding middle frames (from RQ2)
#   - RQ2 results exist at results/judge_evaluation_rq2.json (for comparison)
#   - Gemini API key in gemini_key.txt
#
# Usage:
#   bash scripts/run_rq3_pipeline.sh [stage]
#   Stages: latentlens, spatial, judge, pos, compare, all (default: all)

set -euo pipefail

STAGE="${1:-all}"
N_FRAMES=4

# Per-model indexed layers (from HuggingFace index repos)
declare -A MODEL_LAYERS=(
    ["molmo2"]="1,2,4,8,16,24,34,35"
    ["qwen25vl"]="1,2,4,8,16,24,26,27"
)
# For judge/POS stages that operate on a single layer
JUDGE_LAYER=24

# ── Model configs (video-capable only) ──────────────────────────────────────
declare -A MODEL_IDS=(
    ["molmo2"]="allenai/Molmo2-8B"
    ["qwen25vl"]="Qwen/Qwen2.5-VL-7B-Instruct"
)

declare -A MODEL_KEYS=(
    ["molmo2"]="molmo2"
    ["qwen25vl"]="qwen25vl"
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
VIDEOS_DIR="$SCRATCH_DIR/molmo2cap_videos_500"
FRAMES_DIR="$SCRATCH_DIR/molmo2cap_frames_500"
MODELS=("molmo2" "qwen25vl")

# GPUs to use
GPUS=(0 1)

# ── Stage 1: LatentLens on videos ──────────────────────────────────────────
run_latentlens() {
    echo "=== Stage 1: LatentLens video ($N_FRAMES frames, all indexed layers) ==="

    n_videos=$(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "Videos: $n_videos in $VIDEOS_DIR"

    if [ "$n_videos" -lt 10 ]; then
        echo "  WARNING: Only $n_videos videos, aborting"
        return 1
    fi

    pids=()
    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        gpu="${GPUS[$i]}"
        layers="${MODEL_LAYERS[$model]}"
        outdir="results/molmo2cap_videos_500_${MODEL_KEYS[$model]}"

        # Count expected layer files vs existing to decide if we need to run
        n_expected=$(echo "$layers" | tr ',' '\n' | wc -l)
        n_existing=$(ls "$outdir"/latentlens_layer*.json 2>/dev/null | wc -l)
        if [ "$n_existing" -ge "$n_expected" ]; then
            echo "  $model: all $n_expected layers done, skipping"
            continue
        fi

        echo "  Starting $model on GPU $gpu (layers: $layers)..."
        CUDA_VISIBLE_DEVICES=$gpu python3 scripts/run_latentlens_video.py \
            --model "${MODEL_IDS[$model]}" \
            --index "${INDEX_IDS[$model]}" \
            --videos-dir "$VIDEOS_DIR" \
            --frames-dir "$FRAMES_DIR" \
            --output-dir "$outdir" \
            --num-videos "$n_videos" \
            --n-frames "$N_FRAMES" \
            --device cuda:0 \
            --dtype "${DTYPES[$model]}" \
            --layers "$layers" \
            --index-layers "$layers" \
            > "logs/latentlens_video_${model}.log" 2>&1 &
        pids+=($!)
        echo "    PID=$! → $outdir"
    done

    echo "  Waiting for ${#pids[@]} jobs..."
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  WARNING: PID $pid failed (exit $?)"
    done
    echo "=== LatentLens video complete ==="
}

# ── Stage 2: Spatial patches ──────────────────────────────────────────────
run_spatial() {
    echo "=== Stage 2: Extract spatial patches ==="

    n_frames=$(ls "$FRAMES_DIR"/*.{jpg,jpeg,png} 2>/dev/null | wc -l)

    for model in "${MODELS[@]}"; do
        indir="results/molmo2cap_videos_500_${MODEL_KEYS[$model]}"
        outdir="results/molmo2cap_videos_500_${MODEL_KEYS[$model]}_spatial"

        n_in=$(ls "$indir"/latentlens_layer*.json 2>/dev/null | wc -l)
        n_out=$(ls "$outdir"/latentlens_layer*.json 2>/dev/null | wc -l)
        if [ "$n_out" -ge "$n_in" ] && [ "$n_in" -gt 0 ]; then
            echo "  $model: spatial already done ($n_out layers), skipping"
            continue
        fi

        if [ "$n_in" -eq 0 ]; then
            echo "  $model: no LatentLens results, skipping"
            continue
        fi

        echo "  Extracting spatial for $model..."
        python3 scripts/extract_spatial_patches.py \
            --model "${MODEL_IDS[$model]}" \
            --model-key "${MODEL_KEYS[$model]}" \
            --results-dir "$indir" \
            --images-dir "$FRAMES_DIR" \
            --output-dir "$outdir" \
            --num-images "$n_frames" \
            > "logs/spatial_video_${model}.log" 2>&1
        echo "    Done → $outdir"
    done
    echo "=== Spatial extraction complete ==="
}

# ── Stage 3: Judge evaluation ──────────────────────────────────────────────
run_judge() {
    echo "=== Stage 3: Judge evaluation (layer $JUDGE_LAYER) ==="
    python3 scripts/run_judge_rq3.py \
        --api-key-file gemini_key.txt \
        --layer "$JUDGE_LAYER" \
        --output results/judge_evaluation_rq3.json \
        2>&1 | tee logs/judge_rq3.log
    echo "=== Judge evaluation complete ==="
}

# ── Stage 4: POS analysis ─────────────────────────────────────────────────
run_pos() {
    echo "=== Stage 4: POS analysis ==="
    python3 scripts/analyze_pos_rq3.py \
        --judge-results results/judge_evaluation_rq3.json \
        --output results/pos_analysis_rq3.json \
        2>&1 | tee logs/pos_rq3.log
    echo "=== POS analysis complete ==="
}

# ── Stage 5: Comparison with RQ2 ──────────────────────────────────────────
run_compare() {
    echo "=== Stage 5: Compare RQ2 vs RQ3 ==="
    python3 scripts/compare_rq2_rq3.py \
        --rq2-judge results/judge_evaluation_rq2.json \
        --rq3-judge results/judge_evaluation_rq3.json \
        --rq2-pos results/pos_analysis_rq2.json \
        --rq3-pos results/pos_analysis_rq3.json \
        --output results/comparison_rq2_rq3.json \
        2>&1 | tee logs/compare_rq2_rq3.log
    echo "=== Comparison complete ==="
}


# ── Main ─────────────────────────────────────────────────────────────────────
mkdir -p logs

case "$STAGE" in
    latentlens) run_latentlens ;;
    spatial)    run_spatial ;;
    judge)      run_judge ;;
    pos)        run_pos ;;
    compare)    run_compare ;;
    all)
        run_latentlens
        run_spatial
        run_judge
        run_pos
        run_compare
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [latentlens|spatial|judge|pos|compare|all]"
        exit 1
        ;;
esac

echo "=== All stages complete ==="
