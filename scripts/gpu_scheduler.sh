#!/bin/bash
# GPU job scheduler for PVSG LatentLens pipeline.
# Manages a queue of (model, n_frames) jobs across 4 GPUs.
# Picks up already-running jobs and fills free GPUs immediately.
#
# Usage: bash scripts/gpu_scheduler.sh

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh"
conda activate molmo2

# ── Config ────────────────────────────────────────────────────────────────────
declare -A MODEL_IDS=(
    ["molmo2"]="allenai/Molmo2-8B"
    ["qwen25vl"]="Qwen/Qwen2.5-VL-7B-Instruct"
)
declare -A MODEL_LAYERS=(
    ["molmo2"]="1,2,4,8,16,24,34,35"
    ["qwen25vl"]="1,2,4,8,16,24,26,27"
)
declare -A DTYPES=(
    ["molmo2"]="bfloat16"
    ["qwen25vl"]="float16"
)
declare -A INDEX_IDS=(
    ["molmo2"]="McGill-NLP/contextual_embeddings-molmo2-8b"
    ["qwen25vl"]="McGill-NLP/contextual_embeddings-qwen2.5-vl-7b"
)
declare -A MODEL_KEYS=(
    ["molmo2"]="molmo2"
    ["qwen25vl"]="qwen25vl"
)

SCRATCH_DIR="${SCRATCH:-/network/scratch/l/leh}/latentlens"
VIDEOS_DIR="$SCRATCH_DIR/pvsg_videos_100"
FRAMES_DIR="$SCRATCH_DIR/pvsg_frames_100"
GPUS=(0 1 2 3)

N_VIDEOS=$(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Videos: $N_VIDEOS"

mkdir -p logs

# ── Build job queue (skip completed) ─────────────────────────────────────────
QUEUE=()
for model in molmo2 qwen25vl; do
    for nf in 1 2 4 8 16; do
        outdir="results/pvsg_100_${model}_${nf}f"
        layers="${MODEL_LAYERS[$model]}"
        n_expected=$(echo "$layers" | tr ',' '\n' | wc -l)
        n_existing=0
        [ -d "$outdir" ] && n_existing=$(find "$outdir" -maxdepth 1 -name "latentlens_layer*.json" 2>/dev/null | wc -l) || true
        if [ "$n_existing" -ge "$n_expected" ]; then
            echo "SKIP $model ${nf}f (already $n_existing/$n_expected layers)"
        else
            QUEUE+=("${model}:${nf}")
        fi
    done
done

echo "Queue: ${#QUEUE[@]} jobs → ${QUEUE[*]}"
echo ""

# ── Track GPU → PID mapping ──────────────────────────────────────────────────
declare -A GPU_PID   # gpu_id → pid
declare -A GPU_JOB   # gpu_id → "model:nf"

launch_job() {
    local gpu=$1
    local job=$2
    local model="${job%%:*}"
    local nf="${job##*:}"
    local key="${MODEL_KEYS[$model]}"
    local outdir="results/pvsg_100_${key}_${nf}f"
    local layers="${MODEL_LAYERS[$model]}"
    local logfile="logs/latentlens_pvsg_${model}_${nf}f.log"

    echo "[$(date +%H:%M:%S)] LAUNCH gpu=$gpu  $model ${nf}f → $outdir"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/run_latentlens_video.py \
        --model "${MODEL_IDS[$model]}" \
        --index "${INDEX_IDS[$model]}" \
        --videos-dir "$VIDEOS_DIR" \
        --frames-dir "$FRAMES_DIR" \
        --output-dir "$outdir" \
        --num-videos "$N_VIDEOS" \
        --n-frames "$nf" \
        --device cuda:0 \
        --index-device cpu \
        --dtype "${DTYPES[$model]}" \
        --layers "$layers" \
        --index-layers "$layers" \
        > "$logfile" 2>&1 &

    GPU_PID[$gpu]=$!
    GPU_JOB[$gpu]="$model ${nf}f"
    echo "         PID=${GPU_PID[$gpu]}"
}

# ── Detect already-running jobs and claim their GPUs ─────────────────────────
# Check nvidia-smi for PIDs already using GPUs
echo "Checking for already-running jobs..."
while IFS=, read -r pid gpu_idx; do
    pid=$(echo "$pid" | xargs)
    gpu_idx=$(echo "$gpu_idx" | xargs)
    # Check if this PID is one of our latentlens jobs
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "run_latentlens_video"; then
        # Figure out which job this is from its command line
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null)
        job_model=""
        job_nf=""
        for m in molmo2 qwen25vl; do
            if echo "$cmdline" | grep -q "${MODEL_IDS[$m]}"; then
                job_model="$m"
            fi
        done
        for nf in 1 2 4 8 16; do
            if echo "$cmdline" | grep -q -- "--n-frames $nf"; then
                job_nf="$nf"
            fi
        done
        if [ -n "$job_model" ] && [ -n "$job_nf" ]; then
            GPU_PID[$gpu_idx]=$pid
            GPU_JOB[$gpu_idx]="$job_model ${job_nf}f"
            # Remove from queue
            NEW_QUEUE=()
            for q in "${QUEUE[@]}"; do
                if [ "$q" != "${job_model}:${job_nf}" ]; then
                    NEW_QUEUE+=("$q")
                fi
            done
            QUEUE=("${NEW_QUEUE[@]}")
            echo "  GPU $gpu_idx: already running $job_model ${job_nf}f (PID $pid)"
        fi
    fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=, read -r pid bus_id; do
    pid=$(echo "$pid" | xargs)
    bus_id=$(echo "$bus_id" | xargs)
    # Map bus_id to GPU index
    gpu_idx=$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | grep "$bus_id" | cut -d',' -f1 | xargs)
    echo "$pid, $gpu_idx"
done)

echo ""
echo "After accounting for running jobs: ${#QUEUE[@]} jobs left → ${QUEUE[*]:-none}"

# ── Fill free GPUs ───────────────────────────────────────────────────────────
qi=0  # queue index
for gpu in "${GPUS[@]}"; do
    if [ -z "${GPU_PID[$gpu]:-}" ] && [ $qi -lt ${#QUEUE[@]} ]; then
        launch_job "$gpu" "${QUEUE[$qi]}"
        ((qi++))
    fi
done

# ── Main loop: wait for any job to finish, then launch next ──────────────────
while true; do
    # Check if any tracked jobs are still running
    any_running=false
    for gpu in "${GPUS[@]}"; do
        pid="${GPU_PID[$gpu]:-}"
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            any_running=true
        fi
    done

    if ! $any_running && [ $qi -ge ${#QUEUE[@]} ]; then
        echo ""
        echo "[$(date +%H:%M:%S)] All jobs complete!"
        break
    fi

    # Wait for any child to finish
    wait -n 2>/dev/null || true

    # Check which GPU freed up
    for gpu in "${GPUS[@]}"; do
        pid="${GPU_PID[$gpu]:-}"
        [ -z "$pid" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            # Job finished — check exit status
            wait "$pid" 2>/dev/null
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "[$(date +%H:%M:%S)] DONE  gpu=$gpu  ${GPU_JOB[$gpu]}  ✓"
            else
                echo "[$(date +%H:%M:%S)] FAIL  gpu=$gpu  ${GPU_JOB[$gpu]}  exit=$rc"
            fi
            unset GPU_PID[$gpu]
            unset GPU_JOB[$gpu]

            # Launch next job from queue if available
            if [ $qi -lt ${#QUEUE[@]} ]; then
                launch_job "$gpu" "${QUEUE[$qi]}"
                ((qi++))
            fi
        fi
    done

    sleep 5
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
for model in molmo2 qwen25vl; do
    for nf in 1 2 4 8 16; do
        d="results/pvsg_100_${model}_${nf}f"
        n=0
        [ -d "$d" ] && n=$(find "$d" -maxdepth 1 -name "latentlens_layer*.json" 2>/dev/null | wc -l) || true
        echo "  $model ${nf}f: $n layers"
    done
done
