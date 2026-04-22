#!/usr/bin/env python3
"""Generate RQ2 example figures for the paper appendix.

Each figure: large image with bbox on top, per-model NN entries below
with the retrieved token rendered in bold using renderer-based positioning.

Usage:
    python scripts/visualize_rq2_examples.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


JUDGE_FILE = Path("results/judge_evaluation_rq2.json")
IMAGES_DIR = Path("data/molmo2cap_frames_500")

MODELS = ["Molmo-7B-D", "Idefics3-8B", "Molmo2-8B", "Qwen2.5-VL-7B"]
MODEL_TYPES = {"Molmo-7B-D": "image", "Idefics3-8B": "image",
               "Molmo2-8B": "video", "Qwen2.5-VL-7B": "video"}

SPATIAL_DIRS = {
    "Molmo-7B-D": "results/molmo2cap_frames_500_molmo-7b-d_spatial",
    "Idefics3-8B": "results/molmo2cap_frames_500_idefics3_spatial",
    "Molmo2-8B": "results/molmo2cap_frames_500_molmo2_spatial",
    "Qwen2.5-VL-7B": "results/molmo2cap_frames_500_qwen25vl_spatial",
}

BBOX_SIZE = 3
MAX_SNIPPET = 75
NN_FONTSIZE = 8
HEADER_FONTSIZE = 9


def get_bbox_coords(patch, grid_h, grid_w, img_w, img_h):
    row, col = patch["patch_row"], patch["patch_col"]
    half = BBOX_SIZE // 2
    left = (col - half) / grid_w * img_w
    top = (row - half) / grid_h * img_h
    right = (col - half + BBOX_SIZE) / grid_w * img_w
    bottom = (row - half + BBOX_SIZE) / grid_h * img_h
    return max(0, left), max(0, top), min(img_w, right), min(img_h, bottom)


def load_judge_data():
    with open(JUDGE_FILE) as f:
        return json.load(f)


def get_patch_for_frame(judge_data, model, frame_name):
    for p in judge_data[model].get("patches", []):
        if p.get("image_path") == frame_name and p.get("ds_key") == "vidframes":
            return p
    return None


def get_grid_for_frame(model, frame_name):
    path = Path(SPATIAL_DIRS[model]) / "latentlens_layer24.json"
    with open(path) as f:
        data = json.load(f)
    for img_data in data["results"]:
        if img_data.get("image_path") == frame_name:
            return img_data.get("grid_h", 12), img_data.get("grid_w", 12)
    return 12, 12


def find_token_span(caption, token_str):
    if not token_str or not caption:
        return None
    low_cap = caption.lower()
    low_tok = token_str.strip().lower()
    idx = low_cap.find(low_tok)
    if idx >= 0:
        return idx, idx + len(low_tok)
    if len(low_tok) >= 3:
        for m in re.finditer(r'\S+', caption):
            if low_tok in m.group().lower():
                return m.start(), m.end()
    return None


def snippet_parts(caption, token_str, max_len=MAX_SNIPPET):
    """Return (prefix, bold_part, suffix) for rendering."""
    span = find_token_span(caption, token_str)
    if span is None:
        trunc = caption[:max_len] + ("..." if len(caption) > max_len else "")
        return (trunc, "", "")

    start, end = span
    bold_text = caption[start:end]
    context_budget = max_len - len(bold_text) - 6
    left_budget = context_budget // 2
    right_budget = context_budget - left_budget

    ctx_start = max(0, start - left_budget)
    ctx_end = min(len(caption), end + right_budget)

    prefix = ("..." if ctx_start > 0 else "") + caption[ctx_start:start]
    suffix = caption[end:ctx_end] + ("..." if ctx_end < len(caption) else "")
    return (prefix, bold_text, suffix)


def draw_rich_text(ax, fig, x, y, parts, fontsize=NN_FONTSIZE):
    """Draw text with bold segment using renderer-based positioning.

    parts: list of (text, weight) tuples, e.g. [("prefix ", "normal"), ("token", "bold"), (" suffix", "normal")]
    """
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    cur_x = x

    for text, weight in parts:
        if not text:
            continue
        t = ax.text(cur_x, y, text, fontsize=fontsize,
                    fontweight=weight, fontfamily="serif",
                    color="#000000" if weight == "bold" else "#333333",
                    transform=ax.transAxes, va="top")
        bb = t.get_window_extent(renderer=renderer)
        bb_axes = inv.transform(bb)
        width = bb_axes[1][0] - bb_axes[0][0]
        cur_x += width


def render_example(frame_name, title, judge_data, output_path):
    img_path = IMAGES_DIR / frame_name
    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size

    model_data = {}
    for model in MODELS:
        patch = get_patch_for_frame(judge_data, model, frame_name)
        if patch is None:
            continue
        grid_h, grid_w = get_grid_for_frame(model, frame_name)
        nbs = patch.get("neighbors", [])[:3]
        nn_entries = []
        for nb in nbs:
            tok = nb.get("token_str", "").strip()
            cap = nb.get("caption", "")
            sim = nb.get("similarity", 0)
            prefix, bold, suffix = snippet_parts(cap, tok)
            nn_entries.append({"sim": sim, "prefix": prefix, "bold": bold, "suffix": suffix})
        model_data[model] = {
            "patch": patch, "grid_h": grid_h, "grid_w": grid_w,
            "nns": nn_entries,
            "interpretable": patch.get("interpretable", False),
        }

    ref_model = next((m for m in MODELS if m in model_data), None)
    if ref_model is None:
        return

    # Draw bbox
    ref = model_data[ref_model]
    left, top, right, bottom = get_bbox_coords(
        ref["patch"], ref["grid_h"], ref["grid_w"], img_w, img_h
    )
    img_with_bbox = pil_img.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    lw = max(3, min(img_w, img_h) // 100)
    draw.rectangle([left, top, right, bottom], outline="red", width=lw)

    # Layout — cap image height to avoid portrait images blowing up
    n_models = sum(1 for m in MODELS if m in model_data)
    text_h = n_models * 0.85 + 0.15
    img_aspect = img_w / img_h
    img_disp_w = 5.5
    img_disp_h = img_disp_w / img_aspect
    max_img_h = 2.8  # cap image height (inches)
    min_fig_w = 5.5  # minimum figure width for text readability
    if img_disp_h > max_img_h:
        img_disp_h = max_img_h
        img_disp_w = img_disp_h * img_aspect
    # Ensure figure is wide enough for the text even if image is narrow
    fig_w = max(img_disp_w, min_fig_w)
    total_h = img_disp_h + text_h

    fig = plt.figure(figsize=(fig_w, total_h))

    # Image on top, centered if narrower than figure
    img_bottom_frac = text_h / total_h
    img_frac_w = img_disp_w / fig_w  # image width as fraction of figure
    img_left = (1.0 - img_frac_w) / 2  # center horizontally
    ax_img = fig.add_axes([img_left, img_bottom_frac + 0.01, img_frac_w, img_disp_h / total_h - 0.01])
    ax_img.imshow(img_with_bbox)
    ax_img.axis("off")

    # Text below
    ax_text = fig.add_axes([0.03, 0.0, 0.94, text_h / total_h])
    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    # We need the renderer for positioning, so draw the figure first
    fig.canvas.draw()

    y = 0.96
    line_step = 0.038  # step per NN line
    model_gap = 0.025

    for model in MODELS:
        if model not in model_data:
            continue
        md = model_data[model]
        interp = md["interpretable"]
        interp_color = "#2e7d32" if interp else "#b71c1c"
        interp_label = "interpretable" if interp else "not interpretable"

        # Model header
        ax_text.text(0.0, y, f"{model}", fontsize=HEADER_FONTSIZE, fontweight="bold",
                     fontfamily="serif", transform=ax_text.transAxes, va="top")
        ax_text.text(0.25, y, f"({MODEL_TYPES[model]})", fontsize=8, color="#666666",
                     fontfamily="serif", transform=ax_text.transAxes, va="top")
        ax_text.text(1.0, y, interp_label, fontsize=7.5, color=interp_color,
                     fontfamily="serif", style="italic",
                     transform=ax_text.transAxes, va="top", ha="right")
        y -= 0.05

        # NN lines with bold token
        for entry in md["nns"]:
            sim_str = f"[{entry['sim']:.2f}]  "
            parts = [
                (sim_str, "normal"),
                (entry["prefix"], "normal"),
                (entry["bold"], "bold"),
                (entry["suffix"], "normal"),
            ]
            draw_rich_text(ax_text, fig, 0.02, y, parts, fontsize=NN_FONTSIZE)
            y -= line_step

        y -= model_gap

    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  Saved {output_path}")


def main():
    judge_data = load_judge_data()

    examples = [
        ("frame_0488.jpg", "Sauce simmering in a pan"),
        ("frame_0484.jpg", "Ajax fans celebrating a goal"),
        ("frame_0420.jpg", "Hand holding a battery"),
        ("frame_0194.jpg", "Tilled soil in a garden"),
        ("frame_0389.jpg", "Winding dirt road"),
        ("frame_0322.jpg", 'Screen showing "Task completed" (OCR)'),
        ("frame_0349.jpg", 'Form with "Cancel" button (OCR)'),
    ]

    output_dir = Path("paper/figures/rq2_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_name, title in examples:
        stem = frame_name.replace(".jpg", "")
        render_example(frame_name, title, judge_data,
                       output_dir / f"{stem}.pdf")
    print("Done!")


if __name__ == "__main__":
    main()
