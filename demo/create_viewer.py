#!/usr/bin/env python3
"""Create Interactive Viewer for LatentLens multi-model comparison.

Supports N models via --model flags. A model selector tab bar controls which
model's patch grid is overlaid on the image. Clicking a patch shows that
model's nearest-neighbor results. Switching models transfers the selection
to the nearest spatial position in the new grid.

Usage:
    python demo/create_viewer.py \
        --model "Molmo-7B-D (image)" results/pixmo10_molmo-7b-d_global/ \
        --model "Molmo2-8B (video)" results/pixmo10_molmo2_global/ \
        --model "Idefics3-8B (image)" results/pixmo10_idefics3/ \
        --model "Qwen2.5-VL-7B (video)" results/pixmo10_qwen25vl/ \
        --frames-dir data/pixmo_cap_100/ \
        --output demo_static_images.html \
        --title "LatentLens: Static Images"
"""

import argparse
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from viewer_lib import (
    escape_for_html,
    extract_patches_from_data,
    highlight_token_in_caption,
    pil_image_to_base64,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_COLORS = ["#1565c0", "#c62828", "#2e7d32", "#e65100"]


def discover_latentlens_files(data_dir: Path) -> Dict[int, Path]:
    results: Dict[int, Path] = {}
    if not data_dir.exists():
        return results
    for json_file in sorted(data_dir.glob("*.json")):
        m = re.search(r"(?:layer|visual)(\d+)", json_file.stem.lower())
        if m:
            results[int(m.group(1))] = json_file
    if results:
        log.info(f"  {data_dir.name}: layers {sorted(results.keys())}")
    return results


def load_latentlens_data(layer_files: Dict[int, Path], num_frames: int = 0) -> Dict[int, List[Dict]]:
    data: Dict[int, List[Dict]] = {}
    for layer, path in layer_files.items():
        with open(path, "r") as f:
            raw = json.load(f)
        images = raw.get("results", [])
        if not images:
            images = raw.get("splits", {}).get("validation", {}).get("images", [])
        if num_frames > 0:
            images = images[:num_frames]
        data[layer] = images
    return data


def load_frames(frames_dir: Optional[Path], num_frames: int = 0) -> List[str]:
    if frames_dir is None or not frames_dir.exists():
        return []
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if num_frames > 0:
        paths = paths[:num_frames]
    return [pil_image_to_base64(Image.open(fp)) for fp in paths]


def _get_frame_grid(frame_data: Dict) -> tuple:
    if "grid_h" in frame_data and "grid_w" in frame_data:
        return frame_data["grid_h"], frame_data["grid_w"]
    patches = extract_patches_from_data(frame_data)
    if not patches:
        return 1, 1
    return max(p.get("patch_row", 0) for p in patches) + 1, max(p.get("patch_col", 0) for p in patches) + 1


def build_patch_data(layer_data: Dict[int, List[Dict]], frame_idx: int) -> Dict[int, Dict]:
    result = {}
    for layer, frames in layer_data.items():
        if frame_idx >= len(frames):
            continue
        layer_patches = {}
        for patch in extract_patches_from_data(frames[frame_idx]):
            row, col = patch.get("patch_row", 0), patch.get("patch_col", 0)
            neighbors = patch.get("nearest_contextual_neighbors", [])[:3]
            items = []
            for i, nb in enumerate(neighbors):
                token = nb.get("token_str", "")
                caption = nb.get("caption", "")[:200]
                highlighted = highlight_token_in_caption(escape_for_html(caption), escape_for_html(token))
                items.append({
                    "rank": i + 1, "token": escape_for_html(token), "caption": highlighted,
                    "similarity": nb.get("similarity", 0.0),
                    "contextual_layer": nb.get("contextual_layer"),
                })
            layer_patches[f"{row},{col}"] = {"row": row, "col": col, "contextual_neighbors": items}
        result[layer] = layer_patches
    return result


def compute_per_model_grids(model_data: Dict[int, List[Dict]], num_frames: int) -> List[List[int]]:
    grids = []
    for fi in range(num_frames):
        gh, gw = 1, 1
        for layer, frames in model_data.items():
            if fi < len(frames):
                gh, gw = _get_frame_grid(frames[fi])
            break
        grids.append([gh, gw])
    return grids


def generate_viewer_html(models, frames_b64, num_frames, title) -> str:
    """Generate HTML viewer for N models.

    models: list of dicts with keys: label, data, grids, frames, layers, color
    """
    all_layers = sorted(set(l for m in models for l in m["layers"]))
    default_layer = all_layers[len(all_layers) // 2] if all_layers else 0

    while len(frames_b64) < num_frames:
        frames_b64.append("")

    # Build JS data
    models_js = []
    for m in models:
        models_js.append({
            "label": m["label"],
            "color": m["color"],
            "frames": m["frames"],
            "grids": m["grids"],
            "layers": m["layers"],
        })

    # Generate tab buttons HTML
    tab_buttons = []
    for i, m in enumerate(models):
        active = ' active' if i == 0 else ''
        tab_buttons.append(
            f'<button class="model-tab{active}" data-idx="{i}" '
            f'style="--model-color:{m["color"]}">{escape_for_html(m["label"])}</button>'
        )
    tabs_html = "\n                ".join(tab_buttons)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape_for_html(title)}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
.container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
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
.frame-nav {{ display: flex; align-items: center; gap: 10px; }}
.frame-counter {{ font-weight: 600; color: #495057; min-width: 90px; text-align: center; font-size: 14px; }}
.main-layout {{ display: flex; gap: 20px; }}
.image-section {{ flex: 0 0 512px; }}
.info-section {{ flex: 1; display: flex; flex-direction: column; gap: 10px; }}
.results-panel {{ background: #f8f9fa; padding: 12px; border-radius: 8px; border: 2px solid #dee2e6; flex: 1; display: flex; flex-direction: column; }}
.results-panel h3 {{ margin: 0 0 10px 0; font-size: 15px; padding-bottom: 8px; border-bottom: 2px solid; text-align: center; }}
.analysis-results {{ flex: 1; overflow-y: auto; max-height: 600px; }}
.image-container {{ position: relative; display: inline-block; width: 512px; height: 512px; }}
.base-image {{ width: 512px; height: 512px; object-fit: fill; border: 3px solid #34495e; border-radius: 8px; }}
.patch-overlay {{ position: absolute; top: 3px; left: 3px; width: 512px; height: 512px; pointer-events: none; }}
.patch {{ position: absolute; border: 1px solid rgba(255,255,255,0.3); cursor: pointer; pointer-events: all; transition: all 0.15s; background: transparent; box-sizing: border-box; }}
.patch:hover {{ border: 2px solid #fff; box-shadow: 0 0 8px rgba(255,255,255,0.8); z-index: 10; }}
.patch.active {{ border: 3px solid #ffc107; box-shadow: 0 0 12px rgba(255,193,7,0.9); z-index: 10; }}
.no-data {{ text-align: center; color: #adb5bd; padding: 20px; font-style: italic; font-size: 13px; }}
.empty-state {{ text-align: center; color: #6c757d; padding: 30px 20px; }}
.result-item {{ margin-bottom: 8px; padding: 8px; background: white; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); font-size: 13px; }}
.result-header {{ font-weight: 600; margin-bottom: 4px; font-size: 12px; }}
.result-content {{ font-size: 13px; color: #495057; }}
.highlight {{ background: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px; }}
.instructions {{ background: #e7f3ff; padding: 12px; border-radius: 6px; border-left: 4px solid #0066cc; margin-top: 15px; font-size: 13px; }}
.instructions p {{ margin: 4px 0; color: #004080; }}
.toggle-button {{ background: #4CAF50; border: none; color: white; padding: 8px 16px; font-size: 13px; cursor: pointer; border-radius: 5px; }}
.toggle-button:hover {{ background: #45a049; }}
.toggle-button.off {{ background: #6c757d; }}
.kbd {{ display: inline-block; padding: 2px 6px; font-size: 11px; background: #eee; border: 1px solid #ccc; border-radius: 3px; font-family: monospace; }}
.model-tab {{ padding: 8px 16px; border: 2px solid #dee2e6; border-radius: 6px; font-size: 13px; cursor: pointer; background: white; font-weight: 600; transition: all 0.15s; }}
.model-tab.active {{ background: color-mix(in srgb, var(--model-color) 12%, white); border-color: var(--model-color); color: var(--model-color); }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{escape_for_html(title)}</h1>
        <p style="color:#666; margin:0;">Switch models to compare patch interpretations. Each model has its own grid resolution.</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label class="control-label">Image:</label>
            <div class="frame-nav">
                <button class="btn" id="prevFrame">&#9664;</button>
                <select id="frameSelect"></select>
                <button class="btn" id="nextFrame">&#9654;</button>
                <span class="frame-counter" id="frameCounter">0/0</span>
            </div>
        </div>
        <div class="control-group">
            <label class="control-label">Layer:</label>
            <select id="layerSelect"></select>
        </div>
        <div class="control-group">
            <label class="control-label">Model:</label>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
                {tabs_html}
            </div>
        </div>
        <div class="control-group" style="align-self:flex-end;">
            <button id="toggleGrid" class="toggle-button">Hide Grid</button>
        </div>
    </div>

    <div class="main-layout">
        <div class="image-section">
            <div class="image-container">
                <img id="baseImage" class="base-image" src="" alt="image">
                <div class="patch-overlay" id="patchOverlay"></div>
            </div>
            <div class="instructions">
                <p><strong>Click</strong> a patch to see nearest neighbors. <strong>Switch models</strong> to compare &mdash; patch selection transfers to the nearest spatial position.</p>
                <p><span class="kbd">&#8592;</span>/<span class="kbd">&#8594;</span> navigate images.</p>
                <p id="gridInfo"></p>
            </div>
        </div>
        <div class="info-section">
            <div id="patchInfo" style="text-align:center; padding:8px; background:#e7f3ff; border-radius:6px;">
                <strong>Click on a patch to see results</strong>
            </div>
            <div class="results-panel" id="resultsPanel">
                <h3 id="resultsTitle"></h3>
                <div class="analysis-results" id="results"><div class="empty-state">Select a patch</div></div>
            </div>
        </div>
    </div>
</div>

<script>
const models = {json.dumps(models_js)};
const frameImages = {json.dumps(frames_b64)};
const allLayers = {json.dumps(all_layers)};
const numFrames = {num_frames};
const defaultLayer = {default_layer};

let currentFrame = 0;
let currentLayer = defaultLayer;
let activeIdx = 0;
let activePatchKey = null;
let activePatchDiv = null;
let gridVisible = true;
let patches = {{}};

function init() {{
    const frameSel = document.getElementById('frameSelect');
    for (let i = 0; i < numFrames; i++) {{
        const opt = document.createElement('option');
        opt.value = i; opt.textContent = 'Image ' + i;
        frameSel.appendChild(opt);
    }}
    frameSel.addEventListener('change', () => goToFrame(parseInt(frameSel.value)));

    const layerSel = document.getElementById('layerSelect');
    allLayers.forEach(l => {{
        const opt = document.createElement('option');
        opt.value = l; opt.textContent = 'Layer ' + l;
        if (l === defaultLayer) opt.selected = true;
        layerSel.appendChild(opt);
    }});
    layerSel.addEventListener('change', () => {{ currentLayer = parseInt(layerSel.value); createPatches(); }});

    document.getElementById('prevFrame').addEventListener('click', () => goToFrame(currentFrame - 1));
    document.getElementById('nextFrame').addEventListener('click', () => goToFrame(currentFrame + 1));
    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowLeft') {{ e.preventDefault(); goToFrame(currentFrame - 1); }}
        if (e.key === 'ArrowRight') {{ e.preventDefault(); goToFrame(currentFrame + 1); }}
    }});

    document.querySelectorAll('.model-tab').forEach(btn => {{
        btn.addEventListener('click', () => setActiveModel(parseInt(btn.dataset.idx)));
    }});
    document.getElementById('toggleGrid').addEventListener('click', () => {{
        gridVisible = !gridVisible;
        document.getElementById('patchOverlay').style.display = gridVisible ? '' : 'none';
        const btn = document.getElementById('toggleGrid');
        btn.textContent = gridVisible ? 'Hide Grid' : 'Show Grid';
        btn.classList.toggle('off', !gridVisible);
    }});

    updatePanelStyle();
    goToFrame(0);
}}

function updatePanelStyle() {{
    const m = models[activeIdx];
    const panel = document.getElementById('resultsPanel');
    panel.style.borderColor = m.color;
    const title = document.getElementById('resultsTitle');
    title.textContent = m.label;
    title.style.color = m.color;
    title.style.borderColor = m.color;
    document.querySelectorAll('.result-item').forEach(el => el.style.borderLeftColor = m.color);
    document.querySelectorAll('.result-header').forEach(el => el.style.color = m.color);
}}

function setActiveModel(idx) {{
    const oldIdx = activeIdx;
    const oldKey = activePatchKey;
    activeIdx = idx;

    document.querySelectorAll('.model-tab').forEach(btn => {{
        btn.classList.toggle('active', parseInt(btn.dataset.idx) === idx);
    }});

    // Map patch to nearest position in new grid
    if (oldKey) {{
        const parts = oldKey.split(',');
        const oldRow = parseInt(parts[0]), oldCol = parseInt(parts[1]);
        const [ogh, ogw] = models[oldIdx].grids[currentFrame];
        const [ngh, ngw] = models[idx].grids[currentFrame];
        const normR = (oldRow + 0.5) / ogh;
        const normC = (oldCol + 0.5) / ogw;
        const newRow = Math.max(0, Math.min(ngh - 1, Math.round(normR * ngh - 0.5)));
        const newCol = Math.max(0, Math.min(ngw - 1, Math.round(normC * ngw - 0.5)));
        activePatchKey = newRow + ',' + newCol;
    }}
    activePatchDiv = null;
    updatePanelStyle();
    createPatches();
}}

function goToFrame(idx) {{
    if (idx < 0 || idx >= numFrames) return;
    currentFrame = idx;
    document.getElementById('baseImage').src = frameImages[idx] || '';
    document.getElementById('frameSelect').value = idx;
    document.getElementById('frameCounter').textContent = (idx + 1) + '/' + numFrames;
    document.getElementById('prevFrame').disabled = (idx === 0);
    document.getElementById('nextFrame').disabled = (idx === numFrames - 1);
    createPatches();
}}

function createPatches() {{
    const overlay = document.getElementById('patchOverlay');
    overlay.innerHTML = '';
    patches = {{}};

    const m = models[activeIdx];
    const [gh, gw] = m.grids[currentFrame];
    const displaySize = 512;

    document.getElementById('gridInfo').innerHTML =
        '<strong>Grid:</strong> ' + gh + '&times;' + gw + ' (' + m.label + ')';

    const layerKey = String(currentLayer);
    const data = m.frames[currentFrame]?.[layerKey] || {{}};

    if (Object.keys(data).length > 0) {{
        Object.entries(data).forEach(([key, pd]) => {{
            const div = makePatch(key, pd.row, pd.col, displaySize, gh, gw);
            overlay.appendChild(div);
            patches[key] = div;
        }});
    }} else {{
        for (let r = 0; r < gh; r++) {{
            for (let c = 0; c < gw; c++) {{
                const key = r + ',' + c;
                const div = makePatch(key, r, c, displaySize, gh, gw);
                overlay.appendChild(div);
                patches[key] = div;
            }}
        }}
    }}

    if (activePatchKey && patches[activePatchKey]) {{
        patches[activePatchKey].classList.add('active');
        activePatchDiv = patches[activePatchKey];
        showResults(activePatchKey);
    }}
}}

function makePatch(key, row, col, size, gh, gw) {{
    const div = document.createElement('div');
    div.className = 'patch';
    const pw = size / gw, ph = size / gh;
    div.style.left = (col * pw) + 'px';
    div.style.top = (row * ph) + 'px';
    div.style.width = pw + 'px';
    div.style.height = ph + 'px';
    div.addEventListener('click', (e) => {{
        e.stopPropagation();
        if (activePatchDiv) activePatchDiv.classList.remove('active');
        div.classList.add('active');
        activePatchDiv = div;
        activePatchKey = key;
        showResults(key);
    }});
    return div;
}}

function renderNeighbors(data, noMsg) {{
    const color = models[activeIdx].color;
    if (data && data.contextual_neighbors && data.contextual_neighbors.length > 0) {{
        let h = '';
        data.contextual_neighbors.forEach(ctx => {{
            const badge = (ctx.contextual_layer != null)
                ? ' <span style="background:#6c757d;color:white;padding:1px 4px;border-radius:3px;font-size:10px;">L' + ctx.contextual_layer + '</span>' : '';
            const tok = ctx.token.trim();
            const tokenDisplay = tok ? '"' + tok + '"' : '<i>[space]</i>';
            h += '<div class="result-item" style="border-left:3px solid ' + color + '"><div class="result-header" style="color:' + color + '">' + ctx.rank + '. ' + tokenDisplay + '  Sim: ' + ctx.similarity.toFixed(3) + badge + '</div>'
               + '<div class="result-content" style="margin-top:3px;font-size:12px;">' + ctx.caption + '</div></div>';
        }});
        return h;
    }}
    return '<div class="no-data">' + noMsg + '</div>';
}}

function showResults(key) {{
    const layerKey = String(currentLayer);
    const m = models[activeIdx];
    const data = m.frames[currentFrame]?.[layerKey]?.[key];

    const parts = key.split(',');
    const row = parseInt(parts[0]), col = parseInt(parts[1]);

    document.getElementById('patchInfo').innerHTML =
        '<strong>Patch (' + row + ',' + col + ') | Layer ' + currentLayer + ' | Image ' + currentFrame + '</strong>';
    document.getElementById('results').innerHTML =
        renderNeighbors(data, 'No data for this layer');
}}

window.addEventListener('load', () => {{
    const img = document.getElementById('baseImage');
    if (img.complete) init(); else img.onload = init;
}});
setTimeout(() => {{ if (!document.getElementById('frameSelect').options.length) init(); }}, 300);
</script>
</body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate interactive LatentLens comparison viewer")
    parser.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "DIR"),
                        help="Model label and results directory (repeat for each model)")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("viewer.html"))
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--title", type=str, default="LatentLens Comparison")
    args = parser.parse_args()

    if not args.model or len(args.model) < 1:
        parser.error("At least one --model is required")

    log.info("Discovering analysis files...")
    models = []
    for i, (label, dir_str) in enumerate(args.model):
        data_dir = Path(dir_str)
        layer_files = discover_latentlens_files(data_dir)
        models.append({"label": label, "dir": data_dir, "layer_files": layer_files,
                        "color": MODEL_COLORS[i % len(MODEL_COLORS)]})

    num_frames = args.num_frames
    if num_frames == 0:
        for m in models:
            for path in m["layer_files"].values():
                with open(path) as f:
                    num_frames = max(num_frames, len(json.load(f).get("results", [])))
                break
            if num_frames > 0:
                break
    num_frames = max(num_frames, 1)
    log.info(f"Using {num_frames} images")

    for m in models:
        m["data"] = load_latentlens_data(m["layer_files"], num_frames)
        m["grids"] = compute_per_model_grids(m["data"], num_frames)
        m["frames"] = [build_patch_data(m["data"], fi) for fi in range(num_frames)]
        m["layers"] = sorted(m["data"].keys())

    frames_b64 = load_frames(args.frames_dir, num_frames)
    log.info(f"Loaded {len(frames_b64)} images")

    html = generate_viewer_html(models, frames_b64, num_frames, title=args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Viewer written to {args.output}")


if __name__ == "__main__":
    main()
