#!/bin/bash
# PVSG pipeline: LatentLens video → spatial patches → judge → POS
# Runs on 100 PVSG videos (50 Ego4D + 50 VidOR), 2 VideoLLMs (Molmo2-8B, Qwen2.5-VL-7B).
# Tests N_FRAMES in {1, 2, 4, 8, 16}.
#
# Prerequisites:
#   - Run scripts/prepare_pvsg_data.py first to populate pvsg_videos_100/ and pvsg_frames_100/
#   - Gemini API key in gemini_key.txt
#
# Usage:
#   bash scripts/run_pvsg_pipeline.sh [stage]
#   Stages: prepare, latentlens, spatial, judge, pos, all (default: all)

set -euo pipefail

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh"
conda activate molmo2

STAGE="${1:-all}"
N_FRAMES_LIST=(1 2 4 8 16)

# Per-model indexed layers (same as RQ3)
declare -A MODEL_LAYERS=(
    ["molmo2"]="24"
    ["qwen25vl"]="24"
)
JUDGE_LAYER=24

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
VIDEOS_DIR="$SCRATCH_DIR/pvsg_videos_100"
FRAMES_DIR="$SCRATCH_DIR/pvsg_frames_100"
MODELS=("molmo2" "qwen25vl")

# ── Stage 0: Prepare PVSG data ──────────────────────────────────────────────
run_prepare() {
    echo "=== Stage 0: Prepare PVSG video subset ==="
    n_vids=$(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l || echo 0)
    n_frames=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)

    if [ "$n_vids" -ge 100 ] && [ "$n_frames" -ge 100 ]; then
        echo "  Already prepared ($n_vids videos, $n_frames frames), skipping"
        return 0
    fi

    python3 scripts/prepare_pvsg_data.py --seed 42 --n-per-dataset 50
    echo "=== Data preparation complete ==="
}

# ── Stage 1: LatentLens on videos (all frame counts, both models in parallel) ──
run_latentlens() {
    echo "=== Stage 1: LatentLens video (frames: ${N_FRAMES_LIST[*]}) ==="

    n_videos=$(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l || echo 0)
    echo "Videos: $n_videos in $VIDEOS_DIR"

    if [ "$n_videos" -lt 10 ]; then
        echo "  ERROR: Only $n_videos videos. Run 'prepare' stage first."
        return 1
    fi

    for model in "${MODELS[@]}"; do
        for n_frames in "${N_FRAMES_LIST[@]}"; do
            echo ""
            echo "  --- $model ${n_frames}f ---"
            layers="${MODEL_LAYERS[$model]}"
            outdir="results/pvsg_100_${MODEL_KEYS[$model]}_${n_frames}f"

            n_expected=$(echo "$layers" | tr ',' '\n' | wc -l)
            n_existing=0
            [ -d "$outdir" ] && n_existing=$(find "$outdir" -maxdepth 1 -name "latentlens_layer*.json" | wc -l) || true
            if [ "$n_existing" -ge "$n_expected" ]; then
                echo "    Already done ($n_existing/$n_expected layers), skipping"
                continue
            fi

            echo "    Running $model ${n_frames}f (layers: $layers)..."
            CUDA_VISIBLE_DEVICES=0 python3 scripts/run_latentlens_video.py \
                --model "${MODEL_IDS[$model]}" \
                --index "${INDEX_IDS[$model]}" \
                --videos-dir "$VIDEOS_DIR" \
                --frames-dir "$FRAMES_DIR" \
                --output-dir "$outdir" \
                --num-videos "$n_videos" \
                --n-frames "$n_frames" \
                --device cuda:0 \
                --index-device cpu \
                --dtype "${DTYPES[$model]}" \
                --layers "$layers" \
                --index-layers "$layers" \
                2>&1 | tee "logs/latentlens_pvsg_${model}_${n_frames}f.log"
            echo "    Done → $outdir"
        done
    done
    echo "=== LatentLens video complete ==="
}

# ── Stage 2: Spatial patches ──────────────────────────────────────────────
run_spatial() {
    echo "=== Stage 2: Extract spatial patches ==="

    n_frames_dir=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)

    for n_frames in "${N_FRAMES_LIST[@]}"; do
        for model in "${MODELS[@]}"; do
            indir="results/pvsg_100_${MODEL_KEYS[$model]}_${n_frames}f"
            outdir="results/pvsg_100_${MODEL_KEYS[$model]}_${n_frames}f_spatial"

            n_in=0
            [ -d "$indir" ] && n_in=$(find "$indir" -maxdepth 1 -name "latentlens_layer*.json" | wc -l) || true
            n_out=0
            [ -d "$outdir" ] && n_out=$(find "$outdir" -maxdepth 1 -name "latentlens_layer*.json" | wc -l) || true
            if [ "$n_out" -ge "$n_in" ] && [ "$n_in" -gt 0 ]; then
                echo "  $model ${n_frames}f: spatial already done ($n_out layers), skipping"
                continue
            fi

            if [ "$n_in" -eq 0 ]; then
                echo "  $model ${n_frames}f: no LatentLens results, skipping"
                continue
            fi

            echo "  Extracting spatial for $model ${n_frames}f..."
            python3 scripts/extract_spatial_patches.py \
                --model "${MODEL_IDS[$model]}" \
                --model-key "${MODEL_KEYS[$model]}" \
                --results-dir "$indir" \
                --images-dir "$FRAMES_DIR" \
                --output-dir "$outdir" \
                --num-images "$n_frames_dir" \
                > "logs/spatial_pvsg_${model}_${n_frames}f.log" 2>&1
            echo "    Done → $outdir"
        done
    done
    echo "=== Spatial extraction complete ==="
}

# ── Stage 3: Judge evaluation ──────────────────────────────────────────────
run_judge() {
    echo "=== Stage 3: Judge evaluation (layer $JUDGE_LAYER) ==="
    for n_frames in "${N_FRAMES_LIST[@]}"; do
        echo "  --- ${n_frames}f ---"
        python3 scripts/run_judge_pvsg.py \
            --api-key-file gemini_key.txt \
            --layer "$JUDGE_LAYER" \
            --n-frames "$n_frames" \
            --output "results/judge_evaluation_pvsg_${n_frames}f.json" \
            2>&1 | tee "logs/judge_pvsg_${n_frames}f.log"
    done
    echo "=== Judge evaluation complete ==="
}

# ── Stage 4: POS analysis ─────────────────────────────────────────────────
run_pos() {
    echo "=== Stage 4: POS analysis ==="
    for n_frames in "${N_FRAMES_LIST[@]}"; do
        echo "  --- ${n_frames}f ---"
        python3 scripts/analyze_pos_rq3.py \
            --judge-results "results/judge_evaluation_pvsg_${n_frames}f.json" \
            --output "results/pos_analysis_pvsg_${n_frames}f.json" \
            2>&1 | tee "logs/pos_pvsg_${n_frames}f.log"
    done
    echo "=== POS analysis complete ==="
}


# ── Main ─────────────────────────────────────────────────────────────────────
mkdir -p logs

case "$STAGE" in
    prepare)    run_prepare ;;
    latentlens) run_latentlens ;;
    spatial)    run_spatial ;;
    judge)      run_judge ;;
    pos)        run_pos ;;
    all)
        run_prepare
        run_latentlens
        run_spatial
        run_judge
        run_pos
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [prepare|latentlens|spatial|judge|pos|all]"
        exit 1
        ;;
esac

echo "=== All stages complete ==="
