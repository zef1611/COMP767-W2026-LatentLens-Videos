#!/usr/bin/env python3
"""RQ5: Generate interactive HTML demo for object tracking with LatentLens NNs.

Reads sliding-window LatentLens results (per-frame NNs) for two models,
maps PVSG object masks to the patch grid for each frame, and generates a
self-contained HTML demo with model toggle.

Usage:
    python scripts/rq5/create_interactive_demo.py \
        --sliding-molmo2 results/pvsg_demo_molmo2_sliding/pvsg_0058_layer24.json \
                         results/pvsg_demo_molmo2_sliding/pvsg_0052_layer24.json \
        --sliding-qwen   results/pvsg_demo_qwen25vl_sliding/pvsg_0058_layer24.json \
                         results/pvsg_demo_qwen25vl_sliding/pvsg_0052_layer24.json \
        --patch-map results/rq5_patch_map_molmo2_1f.json \
        --output interactive_demo_rq5.html
"""

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/network/scratch/l/leh"))
VIDEOS_DIR = SCRATCH / "latentlens" / "pvsg_videos_100"
PVSG_ROOT = SCRATCH / "latentlens" / "pvsg"

DISPLAY_SIZE = 512
MAX_NEIGHBORS = 5
MIN_COVERAGE = 0.5


def resolve_symlink(pvsg_name: str):
    link = VIDEOS_DIR / f"{pvsg_name}.mp4"
    if not link.is_symlink():
        return None, None
    target = link.resolve()
    parts = target.parts
    for i, part in enumerate(parts):
        if part == "pvsg" and i + 2 < len(parts):
            return target.stem, parts[i + 1]
    return target.stem, "unknown"


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=10,
    )
    return float(result.stdout.strip())


def extract_frame_at_timestamp(video_path: Path, timestamp: float) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "frame.jpg"
        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            "-y", str(frame_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if frame_path.exists():
            return Image.open(frame_path).convert("RGB").copy()
    return None


def image_to_base64(img: Image.Image, max_size: int = 512) -> str:
    w, h = img.size
    scale = min(max_size / w, max_size / h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def mask_to_patch_assignments(mask: np.ndarray, object_ids: list,
                              grid_h: int, grid_w: int) -> dict:
    h, w = mask.shape[:2]
    patch_h = h / grid_h
    patch_w = w / grid_w
    assignments = {}
    for r in range(grid_h):
        for c in range(grid_w):
            y0 = int(r * patch_h)
            y1 = int((r + 1) * patch_h)
            x0 = int(c * patch_w)
            x1 = int((c + 1) * patch_w)
            patch_region = mask[y0:y1, x0:x1]
            total_pixels = patch_region.size
            for obj_id in object_ids:
                coverage = float(np.sum(patch_region == obj_id)) / total_pixels
                if coverage >= MIN_COVERAGE:
                    key = f"{r},{c}"
                    if key not in assignments:
                        assignments[key] = []
                    assignments[key].append({
                        "object_id": int(obj_id),
                        "coverage": round(coverage, 2),
                    })
    return assignments


def load_sliding_results(json_paths: list) -> dict:
    """Load sliding-window results, keyed by video_name."""
    by_video = {}
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        by_video[data["video_name"]] = data
    return by_video


def build_nn_lookup_from_sliding(sliding_data: dict) -> dict:
    """Build {(timestamp_str, row, col) -> [nn...]} from sliding results."""
    lookup = {}
    for frame in sliding_data.get("frames", []):
        ts_key = str(frame["timestamp"])
        grid_h = frame["grid_h"]
        grid_w = frame["grid_w"]
        for patch in frame.get("patches", []):
            row = patch["patch_row"]
            col = patch["patch_col"]
            nns = patch.get("nearest_contextual_neighbors", [])
            lookup[(ts_key, row, col)] = [
                {
                    "token": nn.get("token_str", "").strip(),
                    "caption": nn.get("caption", ""),
                    "similarity": round(nn.get("similarity", 0), 4),
                    "layer": nn.get("contextual_layer", -1),
                }
                for nn in nns[:MAX_NEIGHBORS]
            ]
    return lookup


def build_demo_data(video_names, patch_map, molmo2_sliding, qwen_sliding):
    pm_by_name = {v["video_name"]: v for v in patch_map["videos"]}

    videos = []
    for video_name in video_names:
        pm_entry = pm_by_name.get(video_name)
        if pm_entry is None:
            log.warning(f"{video_name} not in patch map, skipping")
            continue

        objects = pm_entry["objects_in_all_frames"]
        object_ids = [o["object_id"] for o in objects]

        video_path = VIDEOS_DIR / f"{video_name}.mp4"
        if not video_path.exists():
            log.warning(f"Video not found: {video_path}")
            continue

        video_id, dataset = resolve_symlink(video_name)
        if video_id is None:
            continue

        duration = get_video_duration(video_path)
        mask_dir = PVSG_ROOT / dataset / "masks" / video_id
        n_mask_files = len(list(mask_dir.glob("*.png")))

        # Determine timestamps: union of both models' sliding timestamps
        all_timestamps = set()
        molmo2_data = molmo2_sliding.get(video_name)
        qwen_data = qwen_sliding.get(video_name)

        if molmo2_data:
            for f in molmo2_data["frames"]:
                all_timestamps.add(f["timestamp"])
        if qwen_data:
            for f in qwen_data["frames"]:
                all_timestamps.add(f["timestamp"])

        if not all_timestamps:
            log.warning(f"{video_name}: no sliding data, skipping")
            continue

        timestamps = sorted(all_timestamps)
        log.info(f"  {video_name}: {duration:.1f}s, {len(objects)} objects, "
                 f"{len(timestamps)} frames")

        # Build NN lookups per model
        molmo2_nn = build_nn_lookup_from_sliding(molmo2_data) if molmo2_data else {}
        qwen_nn = build_nn_lookup_from_sliding(qwen_data) if qwen_data else {}

        # Get grid dims per model from first frame
        molmo2_grid = (14, 14)
        qwen_grid = (14, 14)
        if molmo2_data and molmo2_data["frames"]:
            f0 = molmo2_data["frames"][0]
            molmo2_grid = (f0["grid_h"], f0["grid_w"])
        if qwen_data and qwen_data["frames"]:
            f0 = qwen_data["frames"][0]
            qwen_grid = (f0["grid_h"], f0["grid_w"])

        frames_data = []
        for ts in timestamps:
            ts_str = str(ts)

            # Extract video frame
            img = extract_frame_at_timestamp(video_path, ts)
            if img is None:
                continue

            img_b64 = image_to_base64(img, max_size=DISPLAY_SIZE)
            img_w, img_h = img.size

            # Load PVSG mask and compute object assignments for both grids
            mask_idx = min(max(0, round(ts * 5)), n_mask_files - 1)
            mask_path = mask_dir / f"{mask_idx:04d}.png"
            mask = None
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                if mask.shape[:2] != (img_h, img_w):
                    mask_pil = Image.fromarray(mask)
                    mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                    mask = np.array(mask_pil)

            obj_by_id = {o["object_id"]: o for o in objects}

            def compute_objects_and_nns(grid_h, grid_w, nn_lookup):
                patch_objects = {}
                if mask is not None:
                    raw = mask_to_patch_assignments(mask, object_ids, grid_h, grid_w)
                    for key, assigns in raw.items():
                        patch_objects[key] = [
                            {
                                "object_id": a["object_id"],
                                "category": obj_by_id.get(a["object_id"], {}).get("category", "?"),
                                "coverage": a["coverage"],
                            }
                            for a in assigns
                        ]

                patch_nns = {}
                for r in range(grid_h):
                    for c in range(grid_w):
                        nns = nn_lookup.get((ts_str, r, c), [])
                        if nns:
                            patch_nns[f"{r},{c}"] = nns

                return patch_objects, patch_nns

            m2_objs, m2_nns = compute_objects_and_nns(
                molmo2_grid[0], molmo2_grid[1], molmo2_nn)
            qw_objs, qw_nns = compute_objects_and_nns(
                qwen_grid[0], qwen_grid[1], qwen_nn)

            frames_data.append({
                "timestamp": round(ts, 2),
                "image": img_b64,
                "molmo2": {
                    "patch_objects": m2_objs,
                    "patch_nns": m2_nns,
                    "has_nns": len(m2_nns) > 0,
                },
                "qwen": {
                    "patch_objects": qw_objs,
                    "patch_nns": qw_nns,
                    "has_nns": len(qw_nns) > 0,
                },
            })

        videos.append({
            "name": video_name,
            "duration": round(duration, 2),
            "objects": [
                {"id": o["object_id"], "category": o["category"],
                 "is_thing": o.get("is_thing", True)}
                for o in objects
            ],
            "molmo2_grid": list(molmo2_grid),
            "qwen_grid": list(qwen_grid),
            "frames": frames_data,
        })
        log.info(f"  Done: {len(frames_data)} frames")

    return videos


def generate_html(videos: list) -> str:
    data_json = json.dumps(videos, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LatentLens RQ5: Object Tracking in Video</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
.container {{ max-width: 1500px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
.header {{ text-align: center; margin-bottom: 20px; border-bottom: 3px solid #007bff; padding-bottom: 15px; }}
h1 {{ color: #2c3e50; margin: 0 0 5px 0; font-size: 26px; }}
.controls {{ display: flex; gap: 15px; margin-bottom: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; align-items: center; flex-wrap: wrap; }}
.control-group {{ display: flex; flex-direction: column; gap: 5px; }}
.control-label {{ font-weight: 600; color: #495057; font-size: 13px; }}
select, .btn {{ padding: 8px 14px; border: 2px solid #dee2e6; border-radius: 6px; font-size: 14px; background: white; cursor: pointer; }}
select:focus {{ outline: none; border-color: #007bff; }}
.btn {{ background: #007bff; color: white; border: none; font-weight: 600; min-width: 80px; text-align: center; }}
.btn:hover {{ background: #0056b3; }}
.btn:disabled {{ background: #adb5bd; cursor: not-allowed; }}

.frame-slider-container {{ display: flex; align-items: center; gap: 10px; flex: 1; min-width: 300px; }}
.frame-slider {{ flex: 1; height: 6px; -webkit-appearance: none; appearance: none; background: #dee2e6; border-radius: 3px; outline: none; }}
.frame-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: #007bff; cursor: pointer; }}
.frame-slider::-moz-range-thumb {{ width: 18px; height: 18px; border-radius: 50%; background: #007bff; cursor: pointer; border: none; }}
.frame-time {{ font-weight: 600; color: #495057; min-width: 120px; text-align: center; font-size: 14px; }}
.frame-time .nn-badge {{ display: inline-block; background: #43a047; color: white; padding: 1px 6px; border-radius: 10px; font-size: 10px; margin-left: 4px; }}

.model-tab {{ padding: 8px 16px; border: 2px solid #dee2e6; border-radius: 6px; font-size: 14px; cursor: pointer; background: white; font-weight: 600; transition: all 0.15s; }}
.model-tab.active-molmo2 {{ background: #e3f2fd; border-color: #1565c0; color: #1565c0; }}
.model-tab.active-qwen {{ background: #fce4ec; border-color: #c62828; color: #c62828; }}

.object-chips {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
.object-chip {{ padding: 6px 12px; border: 2px solid #dee2e6; border-radius: 20px; font-size: 13px; cursor: pointer; background: white; font-weight: 500; transition: all 0.15s; user-select: none; }}
.object-chip:hover {{ border-color: #007bff; background: #e7f3ff; }}
.object-chip.active {{ border-color: var(--obj-color, #007bff); background: var(--obj-bg, #e7f3ff); color: var(--obj-color, #007bff); font-weight: 700; box-shadow: 0 0 0 2px var(--obj-color, #007bff)33; }}
.object-chip .chip-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
.object-chip.all-chip {{ background: #f8f9fa; border-color: #adb5bd; }}
.object-chip.all-chip.active {{ background: #e2e6ea; border-color: #6c757d; color: #333; box-shadow: 0 0 0 2px #6c757d33; }}

.main-layout {{ display: flex; gap: 20px; }}
.image-section {{ flex: 0 0 {DISPLAY_SIZE + 6}px; }}
.info-section {{ flex: 1; display: flex; flex-direction: column; gap: 10px; }}
.results-panel {{ background: #f8f9fa; padding: 12px; border-radius: 8px; border: 2px solid #dee2e6; flex: 1; display: flex; flex-direction: column; }}
.results-panel h3 {{ margin: 0 0 10px 0; font-size: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6; text-align: center; }}
.results-panel.model-molmo2 h3 {{ color: #1565c0; border-color: #1565c0; }}
.results-panel.model-qwen h3 {{ color: #c62828; border-color: #c62828; }}
.analysis-results {{ flex: 1; overflow-y: auto; max-height: 650px; }}

.image-container {{ position: relative; display: inline-block; width: {DISPLAY_SIZE}px; height: {DISPLAY_SIZE}px; }}
.base-image {{ width: {DISPLAY_SIZE}px; height: {DISPLAY_SIZE}px; object-fit: fill; border: 3px solid #34495e; border-radius: 8px; }}
.patch-overlay {{ position: absolute; top: 3px; left: 3px; width: {DISPLAY_SIZE}px; height: {DISPLAY_SIZE}px; pointer-events: none; }}
.patch {{ position: absolute; border: 1px solid rgba(255,255,255,0.15); cursor: pointer; pointer-events: all; transition: all 0.15s; background: transparent; box-sizing: border-box; }}
.patch:hover {{ border: 2px solid #fff; box-shadow: 0 0 8px rgba(255,255,255,0.8); z-index: 10; }}
.patch.active {{ border: 3px solid #ffc107; box-shadow: 0 0 12px rgba(255,193,7,0.9); z-index: 10; }}
.patch.obj-highlight {{ background: var(--obj-overlay, rgba(0,120,255,0.35)); border-color: rgba(255,255,255,0.6); }}
.patch.obj-highlight:hover {{ background: var(--obj-overlay-hover, rgba(0,120,255,0.5)); }}
.patch.dimmed {{ opacity: 0.3; }}

.no-data {{ text-align: center; color: #adb5bd; padding: 20px; font-style: italic; font-size: 13px; }}
.empty-state {{ text-align: center; color: #6c757d; padding: 30px 20px; }}
.result-item {{ margin-bottom: 8px; padding: 8px; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); font-size: 13px; }}
.model-molmo2 .result-item {{ border-left: 3px solid #1565c0; }}
.model-qwen .result-item {{ border-left: 3px solid #c62828; }}
.result-header {{ font-weight: 600; margin-bottom: 4px; font-size: 12px; }}
.model-molmo2 .result-header {{ color: #1565c0; }}
.model-qwen .result-header {{ color: #c62828; }}
.result-content {{ font-size: 13px; color: #495057; }}
.highlight {{ background: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px; }}

.instructions {{ background: #e7f3ff; padding: 12px; border-radius: 6px; border-left: 4px solid #0066cc; margin-top: 15px; font-size: 13px; }}
.instructions p {{ margin: 4px 0; color: #004080; }}
.toggle-button {{ background: #4CAF50; border: none; color: white; padding: 8px 16px; font-size: 13px; cursor: pointer; border-radius: 5px; }}
.toggle-button:hover {{ background: #45a049; }}
.toggle-button.off {{ background: #6c757d; }}
.kbd {{ display: inline-block; padding: 2px 6px; font-size: 11px; background: #eee; border: 1px solid #ccc; border-radius: 3px; font-family: monospace; }}

.nn-summary {{ padding: 10px; background: #e8f5e9; border-radius: 6px; margin-bottom: 10px; font-size: 13px; }}
.nn-summary strong {{ color: #2e7d32; }}
.no-nn-banner {{ padding: 8px 12px; background: #fff3e0; border-radius: 6px; border-left: 4px solid #ff9800; font-size: 12px; color: #e65100; margin-bottom: 8px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>LatentLens RQ5: Object Tracking &mdash; Molmo2 vs Qwen2.5-VL</h1>
        <p style="color:#666; margin:0;">Scrub through video frames, select objects, switch models to compare patch interpretations.</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label class="control-label">Video:</label>
            <select id="videoSelect"></select>
        </div>
        <div class="control-group" style="flex:1; min-width:300px;">
            <label class="control-label">Frame:</label>
            <div class="frame-slider-container">
                <button class="btn" id="prevFrame" style="min-width:40px;">&#9664;</button>
                <input type="range" class="frame-slider" id="frameSlider" min="0" max="0" value="0">
                <button class="btn" id="nextFrame" style="min-width:40px;">&#9654;</button>
                <span class="frame-time" id="frameTime">0.0s</span>
            </div>
        </div>
        <div class="control-group">
            <label class="control-label">Model:</label>
            <div style="display:flex; gap:8px;">
                <button class="model-tab active-molmo2" id="tabMolmo2">Molmo2-8B</button>
                <button class="model-tab" id="tabQwen">Qwen2.5-VL-7B</button>
            </div>
        </div>
        <div class="control-group" style="align-self:flex-end;">
            <button id="toggleGrid" class="toggle-button">Hide Grid</button>
        </div>
    </div>

    <div class="controls" style="margin-top:-10px;">
        <div class="control-group" style="flex:1;">
            <label class="control-label">Object:</label>
            <div class="object-chips" id="objectChips"></div>
        </div>
    </div>

    <div class="main-layout">
        <div class="image-section">
            <div class="image-container">
                <img id="baseImage" class="base-image" src="" alt="video frame">
                <div class="patch-overlay" id="patchOverlay"></div>
            </div>
            <div class="instructions">
                <p><strong>Scrub the slider</strong> or <span class="kbd">&#8592;</span>/<span class="kbd">&#8594;</span> to navigate. <strong>Switch models</strong> to compare grids and NNs.</p>
                <p><strong>Click an object</strong> to track it. <strong>Click a patch</strong> for its NN tokens.</p>
                <p id="gridInfo"></p>
            </div>
        </div>

        <div class="info-section">
            <div class="results-panel model-molmo2" id="resultsPanel">
                <h3 id="resultsTitle">Molmo2-8B &mdash; Nearest Neighbors</h3>
                <div id="noNnBanner" class="no-nn-banner" style="display:none;">NN data not available for this model/frame combination.</div>
                <div id="nnSummary" class="nn-summary" style="display:none;"></div>
                <div id="patchInfo" style="padding:4px 0; font-size:13px; font-weight:600; color:#333;"></div>
                <div class="analysis-results" id="results">
                    <div class="empty-state">Select an object or click a patch to see nearest neighbors.</div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
const DATA = {data_json};
const DISPLAY = {DISPLAY_SIZE};
const OBJ_COLORS = [
    '#e53935','#1e88e5','#43a047','#fb8c00','#8e24aa',
    '#00acc1','#d81b60','#6d4c41','#546e7a','#f4511e',
    '#7cb342','#039be5','#c0ca33','#5e35b1','#00897b',
    '#ff6f00','#3949ab','#c62828','#2e7d32','#ad1457'
];
const MODEL_NAMES = {{ molmo2: 'Molmo2-8B', qwen: 'Qwen2.5-VL-7B' }};

let currentVideoIdx = 0;
let currentFrameIdx = 0;
let activeModel = 'molmo2';
let selectedObjectId = null;
let activePatchKey = null;
let activePatchDiv = null;
let gridVisible = true;
let patchDivs = {{}};

function objColor(idx) {{ return OBJ_COLORS[idx % OBJ_COLORS.length]; }}
function hexToRgba(hex, a) {{
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

function getModelData(frame) {{
    return activeModel === 'molmo2' ? frame.molmo2 : frame.qwen;
}}
function getGrid(v) {{
    return activeModel === 'molmo2' ? v.molmo2_grid : v.qwen_grid;
}}

function init() {{
    const sel = document.getElementById('videoSelect');
    DATA.forEach((v, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = v.name + ' (' + v.duration + 's, ' + v.objects.length + ' obj)';
        sel.appendChild(opt);
    }});
    sel.addEventListener('change', () => goToVideo(parseInt(sel.value)));

    document.getElementById('frameSlider').addEventListener('input', (e) => {{
        goToFrame(parseInt(e.target.value));
    }});
    document.getElementById('prevFrame').addEventListener('click', () => goToFrame(currentFrameIdx - 1));
    document.getElementById('nextFrame').addEventListener('click', () => goToFrame(currentFrameIdx + 1));

    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
        if (e.key === 'ArrowLeft') {{ e.preventDefault(); goToFrame(currentFrameIdx - 1); }}
        if (e.key === 'ArrowRight') {{ e.preventDefault(); goToFrame(currentFrameIdx + 1); }}
    }});

    document.getElementById('tabMolmo2').addEventListener('click', () => setModel('molmo2'));
    document.getElementById('tabQwen').addEventListener('click', () => setModel('qwen'));

    document.getElementById('toggleGrid').addEventListener('click', () => {{
        gridVisible = !gridVisible;
        document.getElementById('patchOverlay').style.display = gridVisible ? '' : 'none';
        const btn = document.getElementById('toggleGrid');
        btn.textContent = gridVisible ? 'Hide Grid' : 'Show Grid';
        btn.classList.toggle('off', !gridVisible);
    }});

    goToVideo(0);
}}

function setModel(model) {{
    const oldModel = activeModel;
    const oldKey = activePatchKey;
    activeModel = model;

    // Update tabs
    document.getElementById('tabMolmo2').className = 'model-tab' + (model === 'molmo2' ? ' active-molmo2' : '');
    document.getElementById('tabQwen').className = 'model-tab' + (model === 'qwen' ? ' active-qwen' : '');

    // Update panel styling
    const panel = document.getElementById('resultsPanel');
    panel.className = 'results-panel model-' + model;

    // Map patch selection to new grid
    const v = DATA[currentVideoIdx];
    if (oldKey) {{
        const parts = oldKey.split(',');
        const oldRow = parseInt(parts[0]), oldCol = parseInt(parts[1]);
        const oldGrid = oldModel === 'molmo2' ? v.molmo2_grid : v.qwen_grid;
        const newGrid = getGrid(v);
        const normR = (oldRow + 0.5) / oldGrid[0];
        const normC = (oldCol + 0.5) / oldGrid[1];
        const newRow = Math.max(0, Math.min(newGrid[0] - 1, Math.round(normR * newGrid[0] - 0.5)));
        const newCol = Math.max(0, Math.min(newGrid[1] - 1, Math.round(normC * newGrid[1] - 0.5)));
        activePatchKey = newRow + ',' + newCol;
    }}
    activePatchDiv = null;

    const frame = v.frames[currentFrameIdx];
    createPatches(v, frame);

    if (selectedObjectId !== null) {{
        showObjectSummary(v, frame, selectedObjectId);
    }} else if (activePatchKey) {{
        showPatchResults(v, frame, activePatchKey);
    }} else {{
        updateResultsTitle();
        updateNnBanner(frame);
    }}
}}

function goToVideo(idx) {{
    if (idx < 0 || idx >= DATA.length) return;
    currentVideoIdx = idx;
    selectedObjectId = null;
    activePatchKey = null;
    activePatchDiv = null;

    const v = DATA[idx];
    document.getElementById('videoSelect').value = idx;

    const slider = document.getElementById('frameSlider');
    slider.max = v.frames.length - 1;
    slider.value = 0;

    buildObjectChips(v);
    goToFrame(0);
}}

function goToFrame(idx) {{
    const v = DATA[currentVideoIdx];
    if (idx < 0 || idx >= v.frames.length) return;
    currentFrameIdx = idx;

    const frame = v.frames[idx];
    document.getElementById('baseImage').src = frame.image;
    document.getElementById('frameSlider').value = idx;

    const md = getModelData(frame);
    const timeLabel = document.getElementById('frameTime');
    let html = frame.timestamp.toFixed(1) + 's / ' + v.duration + 's';
    if (md.has_nns) html += '<span class="nn-badge">NN</span>';
    timeLabel.innerHTML = html;

    document.getElementById('prevFrame').disabled = (idx === 0);
    document.getElementById('nextFrame').disabled = (idx === v.frames.length - 1);

    createPatches(v, frame);

    if (selectedObjectId !== null) {{
        showObjectSummary(v, frame, selectedObjectId);
    }} else if (activePatchKey) {{
        showPatchResults(v, frame, activePatchKey);
    }} else {{
        updateResultsTitle();
        updateNnBanner(frame);
    }}
}}

function buildObjectChips(v) {{
    const container = document.getElementById('objectChips');
    container.innerHTML = '';

    const allChip = document.createElement('span');
    allChip.className = 'object-chip all-chip active';
    allChip.textContent = 'All';
    allChip.addEventListener('click', () => selectObject(null));
    container.appendChild(allChip);

    v.objects.forEach((obj, i) => {{
        const chip = document.createElement('span');
        const color = objColor(i);
        chip.className = 'object-chip';
        chip.dataset.objectId = obj.id;
        chip.style.setProperty('--obj-color', color);
        chip.style.setProperty('--obj-bg', color + '15');
        chip.innerHTML = '<span class="chip-dot" style="background:' + color + '"></span>'
            + obj.category + (obj.is_thing ? '' : ' <small>(stuff)</small>');
        chip.addEventListener('click', () => selectObject(obj.id));
        container.appendChild(chip);
    }});
}}

function selectObject(objId) {{
    selectedObjectId = objId;
    activePatchKey = null;
    activePatchDiv = null;

    document.querySelectorAll('.object-chip').forEach(c => {{
        c.classList.remove('active');
        if (objId === null && c.classList.contains('all-chip')) c.classList.add('active');
        if (objId !== null && parseInt(c.dataset.objectId) === objId) c.classList.add('active');
    }});

    const v = DATA[currentVideoIdx];
    const frame = v.frames[currentFrameIdx];
    updatePatchHighlights(v, frame);

    if (objId !== null) {{
        showObjectSummary(v, frame, objId);
    }} else {{
        clearResults();
        updateNnBanner(frame);
    }}
}}

function createPatches(v, frame) {{
    const overlay = document.getElementById('patchOverlay');
    overlay.innerHTML = '';
    patchDivs = {{}};

    const [gh, gw] = getGrid(v);
    const pw = DISPLAY / gw, ph = DISPLAY / gh;

    document.getElementById('gridInfo').innerHTML =
        '<strong>Grid:</strong> ' + gh + '&times;' + gw + ' (' + MODEL_NAMES[activeModel] + ')';

    for (let r = 0; r < gh; r++) {{
        for (let c = 0; c < gw; c++) {{
            const key = r + ',' + c;
            const div = document.createElement('div');
            div.className = 'patch';
            div.style.left = (c * pw) + 'px';
            div.style.top = (r * ph) + 'px';
            div.style.width = pw + 'px';
            div.style.height = ph + 'px';
            div.addEventListener('click', (e) => {{
                e.stopPropagation();
                if (activePatchDiv) activePatchDiv.classList.remove('active');
                div.classList.add('active');
                activePatchDiv = div;
                activePatchKey = key;
                showPatchResults(v, v.frames[currentFrameIdx], key);
            }});
            overlay.appendChild(div);
            patchDivs[key] = div;
        }}
    }}

    updatePatchHighlights(v, frame);
}}

function updatePatchHighlights(v, frame) {{
    const [gh, gw] = getGrid(v);
    const md = getModelData(frame);
    const objId = selectedObjectId;

    let colorIdx = 0;
    if (objId !== null) {{
        colorIdx = v.objects.findIndex(o => o.id === objId);
        if (colorIdx < 0) colorIdx = 0;
    }}

    for (let r = 0; r < gh; r++) {{
        for (let c = 0; c < gw; c++) {{
            const key = r + ',' + c;
            const div = patchDivs[key];
            if (!div) continue;

            div.classList.remove('obj-highlight', 'dimmed');
            div.style.removeProperty('--obj-overlay');
            div.style.removeProperty('--obj-overlay-hover');

            const patchObjs = md.patch_objects[key];

            if (objId === null) {{
                if (patchObjs && patchObjs.length > 0) {{
                    const ci = v.objects.findIndex(o => o.id === patchObjs[0].object_id);
                    div.classList.add('obj-highlight');
                    div.style.setProperty('--obj-overlay', hexToRgba(objColor(ci >= 0 ? ci : 0), 0.4));
                    div.style.setProperty('--obj-overlay-hover', hexToRgba(objColor(ci >= 0 ? ci : 0), 0.55));
                }}
            }} else {{
                const belongs = patchObjs && patchObjs.some(o => o.object_id === objId);
                if (belongs) {{
                    div.classList.add('obj-highlight');
                    div.style.setProperty('--obj-overlay', hexToRgba(objColor(colorIdx), 0.4));
                    div.style.setProperty('--obj-overlay-hover', hexToRgba(objColor(colorIdx), 0.55));
                }} else {{
                    div.classList.add('dimmed');
                }}
            }}

            if (activePatchKey === key) {{
                div.classList.add('active');
                activePatchDiv = div;
            }}
        }}
    }}
}}

function updateResultsTitle() {{
    document.getElementById('resultsTitle').textContent =
        MODEL_NAMES[activeModel] + ' \u2014 Nearest Neighbors';
}}

function updateNnBanner(frame) {{
    const md = getModelData(frame);
    document.getElementById('noNnBanner').style.display = md.has_nns ? 'none' : 'block';
}}

function showObjectSummary(v, frame, objId) {{
    const obj = v.objects.find(o => o.id === objId);
    if (!obj) return;
    const md = getModelData(frame);

    updateResultsTitle();
    updateNnBanner(frame);

    const objPatches = [];
    for (const [key, assignments] of Object.entries(md.patch_objects)) {{
        if (assignments.some(a => a.object_id === objId)) objPatches.push(key);
    }}

    const allNns = [];
    if (md.has_nns) {{
        for (const key of objPatches) {{
            const nns = md.patch_nns[key];
            if (nns) nns.forEach(nn => allNns.push(nn));
        }}
    }}

    const seen = {{}};
    allNns.forEach(nn => {{
        if (!seen[nn.token] || nn.similarity > seen[nn.token].similarity) seen[nn.token] = nn;
    }});
    const topNns = Object.values(seen).sort((a, b) => b.similarity - a.similarity).slice(0, 8);

    const summaryDiv = document.getElementById('nnSummary');
    summaryDiv.style.display = 'block';
    let html = '<strong>' + obj.category + '</strong> (id=' + obj.id +
        ', ' + (obj.is_thing ? 'thing' : 'stuff') + ') \u2014 ' +
        objPatches.length + ' patches at t=' + frame.timestamp.toFixed(1) + 's';
    if (topNns.length > 0) {{
        html += ' \u2014 Top tokens: ' +
            topNns.map(nn => '<span class="highlight">' + nn.token + '</span> (' + nn.similarity.toFixed(3) + ')').join(', ');
    }}
    summaryDiv.innerHTML = html;

    document.getElementById('resultsTitle').textContent =
        MODEL_NAMES[activeModel] + ' \u2014 Object: ' + obj.category;
    document.getElementById('patchInfo').innerHTML = 'Click a highlighted patch for per-patch detail.';

    if (!md.has_nns) {{
        document.getElementById('results').innerHTML =
            '<div class="empty-state">No NN data for ' + MODEL_NAMES[activeModel] + ' on this frame.</div>';
    }} else {{
        document.getElementById('results').innerHTML =
            '<div class="empty-state">Click a highlighted patch for per-patch detail.</div>';
    }}
}}

function showPatchResults(v, frame, key) {{
    const md = getModelData(frame);
    const nns = md.patch_nns[key];
    const patchObjs = md.patch_objects[key] || [];

    updateResultsTitle();
    updateNnBanner(frame);

    const parts = key.split(',');
    const row = parseInt(parts[0]), col = parseInt(parts[1]);

    let objLabel = '';
    if (patchObjs.length > 0) {{
        objLabel = ' | Objects: ' + patchObjs.map(o => o.category + ' (' + Math.round(o.coverage * 100) + '%)').join(', ');
    }}

    document.getElementById('patchInfo').innerHTML =
        '<strong>Patch (' + row + ',' + col + ')' + objLabel + '</strong>';

    if (selectedObjectId === null) document.getElementById('nnSummary').style.display = 'none';

    if (!md.has_nns) {{
        document.getElementById('results').innerHTML =
            '<div class="no-data">No NN data for ' + MODEL_NAMES[activeModel] + ' on this frame.</div>';
        return;
    }}

    if (nns && nns.length > 0) {{
        let h = '';
        nns.forEach((nn, i) => {{
            const badge = (nn.layer >= 0)
                ? ' <span style="background:#6c757d;color:white;padding:1px 4px;border-radius:3px;font-size:10px;">L' + nn.layer + '</span>' : '';
            const tok = nn.token || '<i>[space]</i>';

            let captionHtml = nn.caption;
            if (nn.token) {{
                const idx = captionHtml.toLowerCase().indexOf(nn.token.toLowerCase());
                if (idx >= 0) {{
                    captionHtml = captionHtml.substring(0, idx) +
                        '<span class="highlight">' + captionHtml.substring(idx, idx + nn.token.length) + '</span>' +
                        captionHtml.substring(idx + nn.token.length);
                }}
            }}

            h += '<div class="result-item">'
               + '<div class="result-header">' + (i + 1) + '. "' + tok + '"  Sim: ' + nn.similarity.toFixed(4) + badge + '</div>'
               + '<div class="result-content" style="margin-top:3px;font-size:12px;">' + captionHtml + '</div>'
               + '</div>';
        }});
        document.getElementById('results').innerHTML = h;
    }} else {{
        document.getElementById('results').innerHTML = '<div class="no-data">No NN data for this patch.</div>';
    }}
}}

function clearResults() {{
    document.getElementById('nnSummary').style.display = 'none';
    updateResultsTitle();
    document.getElementById('patchInfo').innerHTML = '';
    document.getElementById('results').innerHTML =
        '<div class="empty-state">Select an object or click a patch to see nearest neighbors.</div>';
}}

window.addEventListener('load', () => {{
    const img = document.getElementById('baseImage');
    if (img.complete) init(); else img.onload = init;
}});
setTimeout(() => {{ if (!document.getElementById('videoSelect').options.length) init(); }}, 300);
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="RQ5: Generate interactive HTML demo")
    parser.add_argument("--sliding-molmo2", type=Path, nargs="+", required=True,
                        help="Molmo2 sliding JSON files (one per video)")
    parser.add_argument("--sliding-qwen", type=Path, nargs="+", required=True,
                        help="Qwen2.5-VL sliding JSON files (one per video)")
    parser.add_argument("--patch-map", type=Path, required=True,
                        help="Patch map for object info (any condition)")
    parser.add_argument("--output", type=Path, default=Path("interactive_demo_rq5.html"))
    args = parser.parse_args()

    log.info(f"Loading patch map from {args.patch_map}")
    with open(args.patch_map) as f:
        patch_map = json.load(f)

    log.info("Loading Molmo2 sliding results...")
    molmo2_sliding = load_sliding_results(args.sliding_molmo2)
    log.info(f"  Loaded {len(molmo2_sliding)} videos")

    log.info("Loading Qwen2.5-VL sliding results...")
    qwen_sliding = load_sliding_results(args.sliding_qwen)
    log.info(f"  Loaded {len(qwen_sliding)} videos")

    # Use video names from molmo2 results
    video_names = sorted(set(list(molmo2_sliding.keys()) + list(qwen_sliding.keys())))
    log.info(f"Building demo for videos: {video_names}")

    videos = build_demo_data(video_names, patch_map, molmo2_sliding, qwen_sliding)
    log.info(f"Built data for {len(videos)} videos")

    log.info("Generating HTML...")
    html_content = generate_html(videos)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html_content)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    log.info(f"Written to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
