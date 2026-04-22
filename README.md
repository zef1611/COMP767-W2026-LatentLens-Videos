# LatentLens-Videos

Extending [LatentLens](https://github.com/McGill-NLP/latentlens) interpretability methods to Video LLMs. We investigate whether video-trained models represent visual tokens differently from image-only models.

## Setup

```bash
source /home/nlp/users/bkroje/vl_embedding_spaces/env/bin/activate
pip install latentlens  # if not already installed
```

**Dependencies:** `latentlens`, `transformers`, `torch`, `spacy` (with `en_core_web_sm`), `google-generativeai`, `matplotlib`, `PIL`, `yt-dlp`, `ffmpeg`

## Current Models

| Model | HuggingFace ID | Type | dtype | Index |
|-------|---------------|------|-------|-------|
| Molmo-7B-D | `allenai/Molmo-7B-D-0924` | image | float16 | `indices/molmo-7b-d` (local) |
| Idefics3-8B | `HuggingFaceM4/Idefics3-8B-Llama3` | image | float16 | `indices/idefics3-8b` (local) |
| Molmo2-8B | `allenai/Molmo2-8B` | video | **bfloat16** | `McGill-NLP/contextual_embeddings-molmo2-8b` |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | video | float16 | `McGill-NLP/contextual_embeddings-qwen2.5-vl-7b` |

**Adding a new model:** You need (1) a contextual embedding index, (2) support in `run_latentlens.py`, (3) support in `extract_spatial_patches.py`, and (4) config entries in the judge scripts. See "Adding a new model" section below.

## Experiment Pipeline

The pipeline has 6 stages. Each produces outputs that feed into the next.

### Stage 1: Build contextual index

Build an index from the VLM's LLM backbone on the concepts corpus (117K sentences). This only needs to be done once per model.

```bash
python scripts/build_index.py \
    --model allenai/Molmo-7B-D-0924 \
    --output indices/molmo-7b-d \
    --device cuda:0 --dtype float16 \
    --layers 1,2,4,8,16,24,26,27
```

Pre-built indices on HuggingFace: `McGill-NLP/contextual_embeddings-molmo2-8b`, `McGill-NLP/contextual_embeddings-qwen2.5-vl-7b`. If your model finetunes its LLM backbone (most VLMs do), you **must** build the index from the VLM checkpoint, not the base LLM.

### Stage 2: Run LatentLens

Feed images through the VLM, extract hidden states at visual token positions, search against the contextual index.

```bash
python scripts/run_latentlens.py \
    --model allenai/Molmo-7B-D-0924 \
    --index indices/molmo-7b-d \
    --images-dir data/pixmo_cap_500 \
    --output-dir results/pixmo_cap_500_molmo-7b-d \
    --num-images 500 \
    --device cuda:0 --dtype float16 \
    --layers 24 --index-layers 24
```

**Output:** `results/<dataset>_<model>/latentlens_layer<L>.json` per layer, containing per-image, per-patch top-5 nearest neighbors with similarity scores, source sentences, and contextual layer info.

**Tip:** Use `--layers 24 --index-layers 24` to only extract layer 24 (the one we use for RQ2). Loading fewer index layers saves memory. For the full layer sweep (RQ1), omit `--layers` to use all indexed layers.

### Stage 3: Extract spatial patches

Convert raw LatentLens results (which have meaningless `patch_row`/`patch_col` for multi-crop models) into spatially-correct `_spatial/` directories.

```bash
python scripts/extract_spatial_patches.py \
    --model allenai/Molmo-7B-D-0924 \
    --model-key molmo-7b-d \
    --results-dir results/pixmo_cap_500_molmo-7b-d \
    --images-dir data/pixmo_cap_500 \
    --output-dir results/pixmo_cap_500_molmo-7b-d_spatial \
    --num-images 500
```

**Model keys:** `molmo-7b-d`, `idefics3`, `molmo2`, `qwen25vl`. The `--model-key` determines how spatial patches are extracted (first N patches for Molmo/Molmo2, last N for Idefics3, all with correct grid for Qwen).

**CRITICAL:** Always use `_spatial/` results for anything involving bounding boxes, judge evaluation, or demo visualization. Raw results have incorrect spatial coordinates for multi-crop models.

### Stage 4: Judge evaluation (RQ1 + RQ2)

**RQ1 (interpretability curve, Section 4.2):** 100 patches per layer across all indexed layers.
```bash
python scripts/run_judge_evaluation.py --api-key-file gemini_key.txt
```

**RQ2 (verb analysis, Section 4.3):** 909 patches at a single layer (24), all 4 models.
```bash
python scripts/run_judge_rq2.py \
    --api-key-file gemini_key.txt \
    --layer 24 \
    --output results/judge_evaluation_rq2.json
```

Both scripts use Gemini-2.5-Flash as the LLM judge (API key in `gemini_key.txt`). They support `--resume` (default: on) and save after each model, so interrupted runs can be continued.

**Config for new models:** Edit the `MODELS` dict and `SPATIAL_DIRS` / `IMAGES_DIRS` at the top of each judge script to add your model's result directories.

### Stage 5: POS analysis

**Full POS analysis on existing 100-image data (all layers):**
```bash
python scripts/analyze_pos_all.py
```

**RQ2-specific POS analysis on judge results (interpretable patches only):**
```bash
python scripts/analyze_pos_rq2.py \
    --judge-results results/judge_evaluation_rq2.json \
    --output results/pos_analysis_rq2.json
```

Outputs tables showing dynamic verb %, stative verb %, temporal adverb %, noun %, adj % per model, with breakdowns by dataset (PixMo-Cap vs Molmo2-Cap) and interpretability.

### Stage 6: Generate paper figures

```bash
# RQ2 appendix examples (image + bbox + per-model NNs)
python scripts/visualize_rq2_examples.py

# RQ1 interpretability curve
python scripts/visualize_judge_results.py \
    --model "Molmo-7B-D" \
    --output-dir visualizations/molmo-7b-d

# RQ4 POS emergence + provenance heatmaps
python scripts/plot_rq4.py --output-dir paper/Interpreting-VideoLLMs/figures/

# RQ5 summary plots + example figures
python scripts/rq5/summarize_and_plot.py --results-dir results/
python scripts/rq5/visualize_examples.py --patch-map results/rq5_patch_map_molmo2_16f.json \
    --pos-results results/rq5_object_pos_molmo2_16f.json \
    --results-dir results/pvsg_100_molmo2_16f_allframes \
    --n-frames 16 --model-key molmo2
```

## Full RQ2 pipeline (one command)

To run the entire RQ2 pipeline from scratch (LatentLens → spatial → judge → POS):

```bash
bash scripts/run_rq2_pipeline.sh all
```

Or run individual stages: `latentlens`, `spatial`, `judge`, `pos`.

**Note:** This script is configured for our 4 models and 2 datasets. Edit the model/index/GPU arrays at the top to add new models.

## Full RQ3 pipeline (one command)

RQ3 compares single-frame vs multi-frame LatentLens results. Only video-capable models (Molmo2-8B, Qwen2.5-VL-7B):

```bash
bash scripts/run_rq3_pipeline.sh all
```

Or run individual stages: `latentlens`, `spatial`, `judge`, `pos`, `compare`.

This runs `run_latentlens_video.py` (multi-frame input, 4 frames per video), then reuses `extract_spatial_patches.py` and the judge/POS pipeline adapted for RQ3. The final `compare` stage produces paired single-frame vs multi-frame analysis.

**Note:** `run_rq3_pipeline.sh` now supports multi-layer runs (all indexed layers per model) for RQ4. It uses per-model layer configurations from the contextual embedding indices.

## RQ4: Cross-layer analysis

RQ4 extends RQ2/RQ3 across all indexed layers (8 per model) to study how representations evolve with depth.

### Step 1: Multi-layer LatentLens (GPU)

Run `run_rq3_pipeline.sh latentlens` (which now runs all layers), or manually:
```bash
# Single-frame baseline (all layers)
python scripts/run_latentlens.py \
    --model allenai/Molmo2-8B \
    --index McGill-NLP/contextual_embeddings-molmo2-8b \
    --images-dir data/molmo2cap_frames_500/ \
    --output-dir results/molmo2cap_frames_500_molmo2/ \
    --layers 1,2,4,8,16,24,34,35 --index-layers 1,2,4,8,16,24,34,35 \
    --dtype bfloat16 --device cuda:0

# Multi-frame video (all layers)
python scripts/run_latentlens_video.py \
    --model allenai/Molmo2-8B \
    --index McGill-NLP/contextual_embeddings-molmo2-8b \
    --videos-dir data/molmo2cap_videos_500/ \
    --frames-dir data/molmo2cap_frames_500/ \
    --output-dir results/molmo2cap_videos_500_molmo2/ \
    --layers 1,2,4,8,16,24,34,35 --index-layers 1,2,4,8,16,24,34,35 \
    --n-frames 4 --dtype bfloat16 --device cuda:0
```

### Step 2: Analysis (CPU)

```bash
# POS emergence curves (Option B)
python scripts/analyze_pos_rq4.py --output results/pos_emergence_rq4.json

# NN provenance analysis (Option D)
python scripts/analyze_provenance_rq4.py --output results/provenance_rq4.json

# Combined summary with statistical tests
python scripts/compare_rq4.py --output results/comparison_rq4.json

# Generate figures
python scripts/plot_rq4.py --output-dir paper/Interpreting-VideoLLMs/figures/
```

## RQ5: Object-level POS consistency across frames

RQ5 tracks individual objects across video frames using PVSG segmentation masks and measures whether LatentLens POS interpretations are consistent over time.

### Prerequisites

Stage 0 pre-filters PVSG videos for trackable objects (large enough to dominate ≥1 patch across all frames):

```bash
python scripts/rq5/prepare_pvsg_qualified.py --seed 42 --n-videos 100
```

### Full pipeline (one command)

```bash
bash scripts/rq5/run_pipeline.sh all
```

Or run individual stages: `prepare`, `latentlens`, `map`, `analyze`, `summarize`, `visualize`.

### Manual steps

```bash
# Stage 1: All-frames LatentLens (2+ frames, both models)
python scripts/rq5/run_latentlens_allframes.py \
    --model allenai/Molmo2-8B \
    --index McGill-NLP/contextual_embeddings-molmo2-8b \
    --videos-dir $SCRATCH/latentlens/pvsg_videos_100 \
    --frames-dir $SCRATCH/latentlens/pvsg_frames_100 \
    --output-dir results/pvsg_100_molmo2_4f_allframes \
    --n-frames 4 --layers 24 --dtype bfloat16 --device cuda:0

# Stage 2: Map masks to patches
python scripts/rq5/map_masks_to_patches.py \
    --results-dir results/pvsg_100_molmo2_4f_allframes \
    --n-frames 4 --layer 24 \
    --output results/rq5_patch_map_molmo2_4f.json

# Stage 3: POS consistency analysis
python scripts/rq5/analyze_object_pos.py \
    --patch-map results/rq5_patch_map_molmo2_4f.json \
    --results-dir results/pvsg_100_molmo2_4f_allframes \
    --n-frames 4 --layer 24 \
    --output results/rq5_object_pos_molmo2_4f.json

# Stage 4: Summary and plots
python scripts/rq5/summarize_and_plot.py \
    --results-dir results/ --output results/rq5_summary.json

# Stage 5: Example visualizations
python scripts/rq5/visualize_examples.py \
    --patch-map results/rq5_patch_map_molmo2_4f.json \
    --pos-results results/rq5_object_pos_molmo2_4f.json \
    --results-dir results/pvsg_100_molmo2_4f_allframes \
    --n-frames 4 --model-key molmo2 --n-examples 4
```

**n_frames=1 special case:** Reuses existing `pvsg_100_{model}_1f_spatial/` results (no re-run needed). Serves as a POS baseline — no temporal consistency metrics since there's only one observation per object.

### RQ5 Interactive Demo

Sliding-window LatentLens extracts NNs for every frame across the full video duration, using up to 3 preceding frames as context (4f total). The interactive HTML demo lets you scrub through frames, select objects, and compare Molmo2 vs Qwen2.5-VL side-by-side.

```bash
# Step 1: Run sliding-window LatentLens (one GPU per job, index on CPU)
python scripts/rq5/run_latentlens_sliding.py \
    --model allenai/Molmo2-8B \
    --index McGill-NLP/contextual_embeddings-molmo2-8b \
    --videos $SCRATCH/latentlens/pvsg_videos_100/pvsg_0058.mp4 \
    --output-dir results/pvsg_demo_molmo2_sliding/ \
    --context-frames 3 --fps 1 --layer 24 \
    --dtype bfloat16 --device cuda:0

python scripts/rq5/run_latentlens_sliding.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --index McGill-NLP/contextual_embeddings-qwen2.5-vl-7b \
    --videos $SCRATCH/latentlens/pvsg_videos_100/pvsg_0058.mp4 \
    --output-dir results/pvsg_demo_qwen25vl_sliding/ \
    --context-frames 3 --fps 1 --layer 24 \
    --dtype float16 --device cuda:1

# Step 2: Generate the interactive HTML demo
python scripts/rq5/create_interactive_demo.py \
    --sliding-molmo2 results/pvsg_demo_molmo2_sliding/pvsg_0058_layer24.json \
    --sliding-qwen results/pvsg_demo_qwen25vl_sliding/pvsg_0058_layer24.json \
    --patch-map results/rq5_patch_map_molmo2_1f.json \
    --output interactive_demo_rq5.html
```

## Adding a New Model

To add a new VLM to the experiments:

1. **Build index** (Stage 1). If the model finetunes its LLM, build from the VLM checkpoint.

2. **Add to `run_latentlens.py`**: Add cases to `get_visual_token_mask()` (how to find visual tokens in the input) and `prepare_inputs()` (how to process images). Look at the existing model-specific blocks for examples.

3. **Add to `extract_spatial_patches.py`**: Add a case to `get_spatial_info()` that returns `mode` ("first_n", "last_n", or "all"), `n_patches`, `grid_h`, `grid_w`. This depends on how the model handles multi-crop/tiling:
   - **Multi-crop (global first):** Molmo, Molmo2 → `mode="first_n"`
   - **Multi-tile (global last):** Idefics3 → `mode="last_n"`
   - **Single grid:** Qwen2.5-VL → `mode="all"` with correct grid dims

4. **Add to judge scripts**: Add entry to the `MODELS` dict in `run_judge_evaluation.py` and/or `run_judge_rq2.py` with the model's result directories and layer config.

5. **Run the pipeline** (Stages 2-6) and verify with `python tests/test_spatial_patches.py`.

## Datasets

| Dataset | Location | N | Description |
|---------|----------|---|-------------|
| PixMo-Cap 500 | `$SCRATCH/latentlens/pixmo_cap_500/` | 500 | Static images from `allenai/pixmo-cap` |
| Molmo2-Cap 500 | `$SCRATCH/latentlens/molmo2cap_frames_500/` | 500 | Middle frames from videos (via yt-dlp) |
| Molmo2-Cap Videos 500 | `$SCRATCH/latentlens/molmo2cap_videos_500/` | 500 | Full videos (via yt-dlp) |
| PVSG 100 | `$SCRATCH/latentlens/pvsg_videos_100/` | 100 | Qualified PVSG videos with trackable objects (RQ5) |

**Download** (pre-packaged from HuggingFace, for consistency across evaluations):
```bash
huggingface-cli download huyle1611/latentlens-video-data \
    --repo-type dataset \
    --local-dir $SCRATCH/latentlens
```
This downloads `pixmo_cap_500/`, `molmo2cap_frames_500/`, and `molmo2cap_videos_500/`. PVSG data is prepared separately via `scripts/rq5/prepare_pvsg_qualified.py`.

## Tests

```bash
python tests/test_spatial_patches.py   # Verify spatial patch correctness
python tests/test_judge_pipeline.py    # Offline: data loading, structure, sampling
python tests/test_gemini_api.py        # API connectivity
python tests/test_judge_single.py      # End-to-end: 1 real patch per model
```

Run `test_spatial_patches.py` before any experiment that uses spatial data.

## Key Technical Notes

- **Molmo2-8B requires `--dtype bfloat16`** (float16 overflows in early layers)
- **All 4 models finetune their LLM backbones** → index must be built from VLM checkpoint
- **LatentLens search is cross-layer**: `index.search()` searches across ALL indexed layers, not same-layer
- **Judge uses natural images** (no padding, no black bars). Bbox from grid proportions: `col/grid_w * img_width`
- **Never use raw `results/` dirs for spatial operations** — always use `_spatial/` directories
- **PVSG masks are palette PNGs** — use `PIL.Image.open()`, not `cv2.IMREAD_GRAYSCALE` (cv2 returns wrong object IDs)

## Project Structure

```
scripts/
├── build_index.py              # Build contextual embedding index
├── run_latentlens.py           # Run LatentLens on a VLM with images
├── extract_spatial_patches.py  # Extract spatially-correct patches (CRITICAL)
├── run_judge_evaluation.py     # RQ1: Judge eval across all layers
├── run_judge_rq2.py            # RQ2: Judge eval at single layer, 1K images
├── analyze_pos_all.py          # POS analysis on raw LatentLens results
├── analyze_pos_rq2.py          # POS analysis on judge results (interpretable only)
├── analyze_similarity.py       # Average cosine similarity across models
├── calibrate_gemini_judge.py   # Calibrate Gemini vs GPT-5 + humans
├── visualize_judge_results.py  # Per-patch PNG visualizations
├── visualize_rq2_examples.py   # Appendix figures (image + bold NNs)
├── run_rq2_pipeline.sh         # Full RQ2 pipeline orchestrator
├── run_latentlens_video.py     # RQ3: Multi-frame video LatentLens
├── run_judge_rq3.py            # RQ3: Judge evaluation on video results
├── analyze_pos_rq3.py          # RQ3: POS analysis on video judge results
├── compare_rq2_rq3.py          # RQ3: Paired single vs multi-frame comparison
├── run_rq3_pipeline.sh         # Full RQ3/RQ4 pipeline orchestrator
├── analyze_pos_rq4.py          # RQ4: POS emergence curves across layers
├── analyze_provenance_rq4.py   # RQ4: NN layer provenance analysis
├── compare_rq4.py              # RQ4: Combined summary + stats
├── plot_rq4.py                 # RQ4: Generate paper figures
├── download_molmo2cap_frames.py # Download video frames
└── rq5/                        # RQ5: Object-level POS consistency
    ├── prepare_pvsg_qualified.py    # Pre-filter PVSG videos for trackable objects
    ├── run_latentlens_allframes.py  # All-frames LatentLens extraction
    ├── map_masks_to_patches.py      # Map PVSG masks to spatial patches
    ├── analyze_object_pos.py        # Per-object POS consistency analysis
    ├── summarize_and_plot.py        # Cross-condition summary + figures
    ├── visualize_examples.py        # Per-object tracking example figures
    ├── run_pipeline.sh              # Pipeline orchestrator
    ├── run_latentlens_sliding.py    # Sliding-window LatentLens (4f context per frame)
    └── create_interactive_demo.py   # Interactive HTML demo generator
tests/
├── test_spatial_patches.py     # Spatial correctness verification
├── test_judge_pipeline.py      # Offline data/structure tests
├── test_gemini_api.py          # API connectivity test
└── test_judge_single.py        # End-to-end single patch test
demo/
├── create_viewer.py            # Interactive HTML viewer
└── viewer_lib.py               # Shared utilities
paper/                          # Overleaf paper submodule
indices/                        # Local contextual embedding indices
data/                           # Image datasets (not in git)
results/                        # LatentLens outputs (not in git)
```
