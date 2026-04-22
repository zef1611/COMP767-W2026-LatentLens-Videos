#!/usr/bin/env python3
"""Generate RQ3 example figures for the paper appendix.

Each figure shows one video frame with bbox on top, plus per-model
comparison of single-frame vs multi-frame top-3 NNs shown in their
source-sentence context with the matched token bolded.

Layout matches the RQ2 appendix figures (Figures 8-9 style).
Generates one figure per frame-count setting (4f, 8f, 16f).

Usage:
    python scripts/visualize_rq3_examples.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw


RQ2_JUDGE = Path("results/judge_evaluation_rq2.json")
RQ3_JUDGE = Path("results/judge_evaluation_rq3.json")

_scratch = Path(os.environ.get("SCRATCH", "data")) / "latentlens"
IMAGES_DIR = _scratch / "molmo2cap_frames_500"

MODELS = ["Molmo2-8B", "Qwen2.5-VL-7B"]

# 1-frame spatial dirs (shared baseline for all comparisons)
SPATIAL_DIRS_1F = {
    "Molmo2-8B": "results/molmo2cap_frames_500_molmo2_spatial",
    "Qwen2.5-VL-7B": "results/molmo2cap_frames_500_qwen25vl_spatial",
}

# Multi-frame spatial dirs keyed by n_frames suffix
SPATIAL_DIRS_MULTI = {
    "2f": {
        "Molmo2-8B": "results/molmo2cap_videos_500_molmo2_2f_spatial",
        "Qwen2.5-VL-7B": "results/molmo2cap_videos_500_qwen25vl_2f_spatial",
    },
    "4f": {
        "Molmo2-8B": "results/molmo2cap_videos_500_molmo2_spatial",
        "Qwen2.5-VL-7B": "results/molmo2cap_videos_500_qwen25vl_spatial",
    },
    "8f": {
        "Molmo2-8B": "results/molmo2cap_videos_500_molmo2_8f_spatial",
        "Qwen2.5-VL-7B": "results/molmo2cap_videos_500_qwen25vl_8f_spatial",
    },
    "16f": {
        "Molmo2-8B": "results/molmo2cap_videos_500_molmo2_16f_spatial",
        "Qwen2.5-VL-7B": "results/molmo2cap_videos_500_qwen25vl_16f_spatial",
    },
}

BBOX_SIZE = 3
MAX_CAPTION_LEN = 80
N_NEIGHBORS = 3


def get_bbox_coords(patch, grid_h, grid_w, img_w, img_h):
    row, col = patch["patch_row"], patch["patch_col"]
    left = col / grid_w * img_w
    top = row / grid_h * img_h
    right = (col + BBOX_SIZE) / grid_w * img_w
    bottom = (row + BBOX_SIZE) / grid_h * img_h
    return max(0, left), max(0, top), min(img_w, right), min(img_h, bottom)


def get_patch_for_frame(judge_data, model, frame_name, ds_key="vidframes"):
    patches = judge_data[model].get("patches", [])
    for p in patches:
        if p.get("image_path") == frame_name:
            if ds_key is None or p.get("ds_key") == ds_key:
                return p
    return None


def load_spatial_data(spatial_dir, frame_name):
    path = Path(spatial_dir) / "latentlens_layer24.json"
    with open(path) as f:
        data = json.load(f)
    for img_data in data["results"]:
        if img_data.get("image_path") == frame_name:
            return img_data
    return None


def get_spatial_patch(spatial_data, patch_row, patch_col):
    if spatial_data is None:
        return None
    for p in spatial_data.get("patches", []):
        if p["patch_row"] == patch_row and p["patch_col"] == patch_col:
            return p
    return None


def truncate_caption(caption, token_str, max_len=MAX_CAPTION_LEN):
    token_clean = token_str.strip()
    idx = caption.lower().find(token_clean.lower())
    if idx == -1:
        if len(caption) > max_len:
            return caption[:max_len] + "..."
        return caption

    token_end = idx + len(token_clean)
    half_window = (max_len - len(token_clean)) // 2
    start = max(0, idx - half_window)
    end = min(len(caption), token_end + half_window)
    if start == 0:
        end = min(len(caption), max_len)
    if end == len(caption):
        start = max(0, len(caption) - max_len)

    result = caption[start:end]
    if start > 0:
        result = "..." + result
    if end < len(caption):
        result = result + "..."
    return result


def render_example(frame_name, rq2_data, rq3_data, nf_key, output_path):
    """Render a single comparison example: 1 frame vs N frames.

    Args:
        nf_key: one of "4f", "8f", "16f"
    """
    img_path = IMAGES_DIR / frame_name
    if not img_path.exists():
        print(f"  Skipping {frame_name}: image not found at {img_path}")
        return

    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size

    n_frames_int = int(nf_key.replace("f", ""))
    multi_spatial_dirs = SPATIAL_DIRS_MULTI[nf_key]

    model_sections = []
    ref_patch = None
    ref_grid = None

    for model in MODELS:
        jp_rq2 = get_patch_for_frame(rq2_data, model, frame_name, ds_key="vidframes")
        if jp_rq2 is None:
            continue

        sp_1f_data = load_spatial_data(SPATIAL_DIRS_1F[model], frame_name)
        sp_nf_data = load_spatial_data(multi_spatial_dirs[model], frame_name)

        grid_h = sp_1f_data.get("grid_h", 12) if sp_1f_data else 12
        grid_w = sp_1f_data.get("grid_w", 12) if sp_1f_data else 12

        if ref_patch is None:
            ref_patch = jp_rq2
            ref_grid = (grid_h, grid_w)

        pr, pc = jp_rq2["patch_row"], jp_rq2["patch_col"]

        nns_1f = []
        if sp_1f_data:
            sp = get_spatial_patch(sp_1f_data, pr, pc)
            if sp:
                nns_1f = sp.get("nearest_contextual_neighbors", [])[:N_NEIGHBORS]

        nns_nf = []
        if sp_nf_data:
            sp = get_spatial_patch(sp_nf_data, pr, pc)
            if sp:
                nns_nf = sp.get("nearest_contextual_neighbors", [])[:N_NEIGHBORS]

        # Determine interpretability from judge data
        interp_1f = jp_rq2.get("interpretable", False)
        # For multi-frame, use RQ3 judge if available (4f only), otherwise infer
        if nf_key == "4f":
            jp_rq3 = get_patch_for_frame(rq3_data, model, frame_name, ds_key=None)
            interp_nf = jp_rq3.get("interpretable", False) if jp_rq3 else False
        else:
            # No separate judge for 8f/16f: check if top NNs have meaningful tokens
            meaningful = [nn for nn in nns_nf
                          if len(nn.get("token_str", "").strip()) >= 3
                          and nn.get("token_str", "").strip().isalpha()]
            interp_nf = len(meaningful) >= 1

        model_sections.append({
            "model": model,
            "nns_1f": nns_1f,
            "nns_nf": nns_nf,
            "interp_1f": interp_1f,
            "interp_nf": interp_nf,
        })

    if ref_patch is None or not model_sections:
        print(f"  Skipping {frame_name}: no patch data")
        return

    # Draw bbox
    grid_h, grid_w = ref_grid
    left, top, right, bottom = get_bbox_coords(
        ref_patch, grid_h, grid_w, img_w, img_h
    )
    img_with_bbox = pil_img.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    lw = max(2, min(img_w, img_h) // 100)
    draw.rectangle([left, top, right, bottom], outline="red", width=lw)

    # Count text lines
    n_lines = 0
    for sec in model_sections:
        n_lines += 1 + len(sec["nns_1f"])   # 1f header + NNs
        n_lines += 1 + len(sec["nns_nf"])   # Nf header + NNs
    n_lines += len(model_sections)  # spacing between models

    line_height_inches = 0.22
    text_height = n_lines * line_height_inches + 0.3
    img_display_height = 5.0
    fig_height = img_display_height + text_height

    fig = plt.figure(figsize=(10, fig_height))
    gs = GridSpec(2, 1, height_ratios=[img_display_height, text_height], hspace=0.02)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_with_bbox)
    ax_img.axis("off")

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    line_h = 1.0 / max(n_lines, 1)
    y = 1.0 - line_h * 0.5

    for i, sec in enumerate(model_sections):
        model = sec["model"]

        for cond_label, nns, interp in [
            ("1 frame", sec["nns_1f"], sec["interp_1f"]),
            (f"{n_frames_int} frames", sec["nns_nf"], sec["interp_nf"]),
        ]:
            interp_text = "interpretable" if interp else "not interpretable"
            interp_color = "#2e7d32" if interp else "#c62828"

            ax_text.text(0.0, y, f"{model}", fontsize=10.5, fontweight="bold",
                         transform=ax_text.transAxes, va="center")
            ax_text.text(0.18, y, f"({cond_label})", fontsize=9.5,
                         color="#555555", transform=ax_text.transAxes, va="center")
            ax_text.text(0.99, y, interp_text, fontsize=9.5, fontstyle="italic",
                         color=interp_color, ha="right",
                         transform=ax_text.transAxes, va="center")
            y -= line_h

            for nn in nns:
                token_str = nn.get("token_str", "").strip()
                caption = nn.get("caption", "")
                sim = nn.get("similarity", 0.0)
                truncated = truncate_caption(caption, token_str)

                idx = truncated.lower().find(token_str.lower()) if token_str else -1

                if idx >= 0:
                    before = truncated[:idx]
                    bold_part = truncated[idx:idx + len(token_str)]
                    after = truncated[idx + len(token_str):]
                    line_str = f"[{sim:.2f}]  {before}$\\bf{{{bold_part}}}${after}"
                else:
                    line_str = f"[{sim:.2f}]  {truncated}"

                ax_text.text(0.02, y, line_str, fontsize=8.5,
                             fontfamily="serif", color="#333333",
                             transform=ax_text.transAxes, va="center")
                y -= line_h

        if i < len(model_sections) - 1:
            y -= line_h * 0.5

    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"  Saved {output_path} + {png_path}")


def main():
    rq2_data = json.loads(RQ2_JUDGE.read_text())
    rq3_data = json.loads(RQ3_JUDGE.read_text())

    output_dir = Path("paper/Interpreting-VideoLLMs/figures/rq3_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # One example per frame-count setting (all different frames)
    examples = [
        # (frame, nf_key, output_stem)
        ("frame_0036.jpg", "2f",  "1f_vs_2f"),
        ("frame_0004.jpg", "4f",  "1f_vs_4f"),
        ("frame_0012.jpg", "8f",  "1f_vs_8f"),
        ("frame_0014.jpg", "16f", "1f_vs_16f"),
    ]

    for frame_name, nf_key, stem in examples:
        render_example(frame_name, rq2_data, rq3_data, nf_key,
                       output_dir / f"{stem}.pdf")

    print("Done!")


if __name__ == "__main__":
    main()
