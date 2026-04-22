#!/bin/bash
# RQ2 full pipeline: LatentLens → spatial patches → judge → POS analysis
# Runs on 500 PixMo + 500 Molmo2-Cap frames, layer 24, all 4 models.
#
# Prerequisites:
#   - data/pixmo_cap_500/ has 500 images
#   - data/molmo2cap_frames_500/ has 500 images (or as many as downloaded)
#   - Gemini API key in gemini_key.txt
#
# Usage:
#   bash scripts/run_rq2_pipeline.sh [stage]
#   Stages: latentlens, spatial, judge, pos, all (default: all)

set -euo pipefail
# source /home/nlp/users/bkroje/vl_embedding_spaces/env/bin/activate

STAGE="${1:-all}"
LAYER=24

# ── Model configs ────────────────────────────────────────────────────────────
declare -A MODEL_IDS=(
    ["molmo"]="allenai/Molmo-7B-D-0924"
    ["idefics3"]="HuggingFaceM4/Idefics3-8B-Llama3"
    ["molmo2"]="allenai/Molmo2-8B"
    ["qwen25vl"]="Qwen/Qwen2.5-VL-7B-Instruct"
)

declare -A MODEL_KEYS=(
    ["molmo"]="molmo-7b-d"
    ["idefics3"]="idefics3"
    ["molmo2"]="molmo2"
    ["qwen25vl"]="qwen25vl"
)

declare -A DTYPES=(
    ["molmo"]="float16"
    ["idefics3"]="float16"
    ["molmo2"]="bfloat16"
    ["qwen25vl"]="float16"
)

declare -A INDEX_IDS=(
    ["molmo"]="indices/molmo-7b-d"
    ["idefics3"]="indices/idefics3-8b"
    ["molmo2"]="McGill-NLP/contextual_embeddings-molmo2-8b"
    ["qwen25vl"]="McGill-NLP/contextual_embeddings-qwen2.5-vl-7b"
)

# Datasets
DATASETS=("pixmo_cap_500" "molmo2cap_frames_500")
SCRATCH_DIR="${SCRATCH:-/network/scratch/l/leh}/latentlens"
DATASET_DIRS=("$SCRATCH_DIR/pixmo_cap_500" "$SCRATCH_DIR/molmo2cap_frames_500")
MODELS=("molmo" "idefics3" "molmo2" "qwen25vl")

# GPUs to use (skip 2, 5, 6 which are occupied)
GPUS=(0 1 3 4)

# ── Stage 1: LatentLens ─────────────────────────────────────────────────────
run_latentlens() {
    echo "=== Stage 1: LatentLens (layer $LAYER) ==="

    for ds_idx in 0 1; do
        ds="${DATASETS[$ds_idx]}"
        ds_dir="${DATASET_DIRS[$ds_idx]}"

        n_images=$(ls "$ds_dir"/*.{jpg,jpeg,png} 2>/dev/null | wc -l)
        echo "Dataset: $ds ($n_images images)"

        if [ "$n_images" -lt 10 ]; then
            echo "  WARNING: Only $n_images images in $ds_dir, skipping"
            continue
        fi

        # Run models in parallel (4 GPUs)
        pids=()
        for i in "${!MODELS[@]}"; do
            model="${MODELS[$i]}"
            gpu="${GPUS[$i]}"
            outdir="results/${ds}_${MODEL_KEYS[$model]}"
            outfile="$outdir/latentlens_layer${LAYER}.json"

            if [ -f "$outfile" ]; then
                echo "  $model on $ds: already done, skipping"
                continue
            fi

            echo "  Starting $model on GPU $gpu..."
            CUDA_VISIBLE_DEVICES=$gpu python3 scripts/run_latentlens.py \
                --model "${MODEL_IDS[$model]}" \
                --index "${INDEX_IDS[$model]}" \
                --images-dir "$ds_dir" \
                --output-dir "$outdir" \
                --num-images "$n_images" \
                --device cuda:0 \
                --dtype "${DTYPES[$model]}" \
                --layers "$LAYER" \
                --index-layers "$LAYER" \
                > "logs/latentlens_${ds}_${model}.log" 2>&1 &
            pids+=($!)
            echo "    PID=$! → $outdir"
        done

        # Wait for all models on this dataset
        echo "  Waiting for ${#pids[@]} jobs..."
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "  WARNING: PID $pid failed (exit $?)"
        done
        echo "  Dataset $ds done."
    done
    echo "=== LatentLens complete ==="
}

# ── Stage 2: Spatial patches ────────────────────────────────────────────────
run_spatial() {
    echo "=== Stage 2: Extract spatial patches ==="

    for ds_idx in 0 1; do
        ds="${DATASETS[$ds_idx]}"
        ds_dir="${DATASET_DIRS[$ds_idx]}"

        n_images=$(ls "$ds_dir"/*.{jpg,jpeg,png} 2>/dev/null | wc -l)

        for model in "${MODELS[@]}"; do
            indir="results/${ds}_${MODEL_KEYS[$model]}"
            outdir="results/${ds}_${MODEL_KEYS[$model]}_spatial"

            if [ -d "$outdir" ] && [ -f "$outdir/latentlens_layer${LAYER}.json" ]; then
                echo "  $model on $ds: spatial already done, skipping"
                continue
            fi

            if [ ! -f "$indir/latentlens_layer${LAYER}.json" ]; then
                echo "  $model on $ds: no LatentLens results, skipping"
                continue
            fi

            echo "  Extracting spatial for $model on $ds..."
            python3 scripts/extract_spatial_patches.py \
                --model "${MODEL_IDS[$model]}" \
                --model-key "${MODEL_KEYS[$model]}" \
                --results-dir "$indir" \
                --images-dir "$ds_dir" \
                --output-dir "$outdir" \
                --num-images "$n_images" \
                > "logs/spatial_${ds}_${model}.log" 2>&1
            echo "    Done → $outdir"
        done
    done
    echo "=== Spatial extraction complete ==="
}

# ── Stage 3: Judge evaluation ────────────────────────────────────────────────
run_judge() {
    echo "=== Stage 3: Judge evaluation (layer $LAYER) ==="
    python3 scripts/run_judge_rq2.py \
        --api-key-file gemini_key.txt \
        --layer "$LAYER" \
        --output results/judge_evaluation_rq2.json \
        2>&1 | tee logs/judge_rq2.log
    echo "=== Judge evaluation complete ==="
}

# ── Stage 4: POS analysis ───────────────────────────────────────────────────
run_pos() {
    echo "=== Stage 4: POS analysis ==="
    python3 scripts/analyze_pos_rq2.py \
        --judge-results results/judge_evaluation_rq2.json \
        --output results/pos_analysis_rq2.json \
        2>&1 | tee logs/pos_rq2.log
    echo "=== POS analysis complete ==="
}


# ── Main ─────────────────────────────────────────────────────────────────────
mkdir -p logs

case "$STAGE" in
    latentlens) run_latentlens ;;
    spatial)    run_spatial ;;
    judge)      run_judge ;;
    pos)        run_pos ;;
    all)
        run_latentlens
        run_spatial
        run_judge
        run_pos
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [latentlens|spatial|judge|pos|all]"
        exit 1
        ;;
esac

echo "=== All stages complete ==="
