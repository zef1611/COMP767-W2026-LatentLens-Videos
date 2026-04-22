#!/usr/bin/env python3
"""Generate PVSG example figures comparing 1f vs Nf LatentLens results.

Each figure shows one video frame with bbox on top, plus per-model
comparison of single-frame vs multi-frame top-3 NNs shown in their
source-sentence context with the matched token bolded.

Layout matches the RQ3 appendix figures style.

Usage:
    python scripts/visualize_pvsg_examples.py [--n-examples 4]
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw


_scratch = Path(os.environ.get("SCRATCH", "data")) / "latentlens"
IMAGES_DIR = _scratch / "pvsg_frames_100"

MODELS = ["Molmo2-8B", "Qwen2.5-VL-7B"]
MODEL_KEYS = {"Molmo2-8B": "molmo2", "Qwen2.5-VL-7B": "qwen25vl"}
NF_LIST = [2, 4, 8, 16]

BBOX_SIZE = 3
MAX_CAPTION_LEN = 80
N_NEIGHBORS = 3
SEED = 42


def spatial_dir(model, nf):
    key = MODEL_KEYS[model]
    return Path(f"results/pvsg_100_{key}_{nf}f_spatial")


def load_spatial_data(sdir, frame_name):
    path = sdir / "latentlens_layer24.json"
    if not path.exists():
        return None
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


def get_bbox_coords(patch, grid_h, grid_w, img_w, img_h):
    row, col = patch["patch_row"], patch["patch_col"]
    left = col / grid_w * img_w
    top = row / grid_h * img_h
    right = (col + BBOX_SIZE) / grid_w * img_w
    bottom = (row + BBOX_SIZE) / grid_h * img_h
    return max(0, left), max(0, top), min(img_w, right), min(img_h, bottom)


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


def pick_interesting_patch(spatial_data, rng):
    """Pick a patch that has meaningful (non-whitespace) top neighbors."""
    if spatial_data is None:
        return None
    patches = spatial_data.get("patches", [])
    good_patches = []
    for p in patches:
        nns = p.get("nearest_contextual_neighbors", [])[:5]
        meaningful = [nn for nn in nns
                      if len(nn.get("token_str", "").strip()) >= 3
                      and nn.get("token_str", "").strip().replace(".", "").isalpha()]
        if len(meaningful) >= 2:
            good_patches.append(p)
    if good_patches:
        return rng.choice(good_patches)
    if patches:
        return rng.choice(patches)
    return None


def find_good_examples(n_examples, seed=SEED):
    """Find frames where both models have meaningful patches in both 1f and some Nf."""
    rng = random.Random(seed)
    frames = sorted([f.name for f in IMAGES_DIR.glob("*.jpg")])

    # Load all 1f spatial data for both models
    sp_1f = {}
    for model in MODELS:
        sdir = spatial_dir(model, 1)
        sp_1f[model] = {}
        layer_file = sdir / "latentlens_layer24.json"
        if not layer_file.exists():
            continue
        with open(layer_file) as f:
            data = json.load(f)
        for img_data in data["results"]:
            sp_1f[model][img_data.get("image_path", "")] = img_data

    examples = []
    used_nf = set()
    rng.shuffle(frames)

    for frame in frames:
        if len(examples) >= n_examples:
            break

        # Check both models have data for 1f
        ok = True
        for model in MODELS:
            if frame not in sp_1f[model]:
                ok = False
                break
            img_data = sp_1f[model][frame]
            patches = img_data.get("patches", [])
            meaningful = [p for p in patches
                          if any(len(nn.get("token_str", "").strip()) >= 3
                                 for nn in p.get("nearest_contextual_neighbors", [])[:3])]
            if len(meaningful) < 3:
                ok = False
                break
        if not ok:
            continue

        # Pick a nf_key that hasn't been used yet
        available_nf = [nf for nf in NF_LIST if nf not in used_nf]
        if not available_nf:
            available_nf = NF_LIST  # all used, allow repeats
        nf_key = rng.choice(available_nf)
        used_nf.add(nf_key)
        examples.append((frame, nf_key))

    return examples


def render_example(frame_name, nf, output_path):
    """Render 1f vs Nf comparison for one frame."""
    img_path = IMAGES_DIR / frame_name
    if not img_path.exists():
        print(f"  Skipping {frame_name}: image not found")
        return

    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size

    model_sections = []
    ref_patch = None
    ref_grid = None

    for model in MODELS:
        sp_1f_data = load_spatial_data(spatial_dir(model, 1), frame_name)
        sp_nf_data = load_spatial_data(spatial_dir(model, nf), frame_name)

        if sp_1f_data is None:
            continue

        grid_h = sp_1f_data.get("grid_h", 12)
        grid_w = sp_1f_data.get("grid_w", 12)

        # Pick a patch (use same patch for both conditions)
        rng = random.Random(hash(frame_name + model))
        patch = pick_interesting_patch(sp_1f_data, rng)
        if patch is None:
            continue

        pr, pc = patch["patch_row"], patch["patch_col"]

        if ref_patch is None:
            ref_patch = patch
            ref_grid = (grid_h, grid_w)

        nns_1f = patch.get("nearest_contextual_neighbors", [])[:N_NEIGHBORS]

        nns_nf = []
        if sp_nf_data:
            sp = get_spatial_patch(sp_nf_data, pr, pc)
            if sp:
                nns_nf = sp.get("nearest_contextual_neighbors", [])[:N_NEIGHBORS]

        # Heuristic interpretability based on NN quality
        def is_interpretable(nns):
            meaningful = [nn for nn in nns
                          if len(nn.get("token_str", "").strip()) >= 3
                          and nn.get("token_str", "").strip().replace(".", "").isalpha()]
            return len(meaningful) >= 1

        model_sections.append({
            "model": model,
            "nns_1f": nns_1f,
            "nns_nf": nns_nf,
            "interp_1f": is_interpretable(nns_1f),
            "interp_nf": is_interpretable(nns_nf),
        })

    if ref_patch is None or not model_sections:
        print(f"  Skipping {frame_name}: no patch data")
        return

    # Draw bbox
    grid_h, grid_w = ref_grid
    left, top, right, bottom = get_bbox_coords(ref_patch, grid_h, grid_w, img_w, img_h)
    img_with_bbox = pil_img.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    lw = max(2, min(img_w, img_h) // 100)
    draw.rectangle([left, top, right, bottom], outline="red", width=lw)

    # Count text lines
    n_lines = 0
    for sec in model_sections:
        n_lines += 1 + len(sec["nns_1f"])
        n_lines += 1 + len(sec["nns_nf"])
    n_lines += len(model_sections)

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
            (f"{nf} frames", sec["nns_nf"], sec["interp_nf"]),
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
                    # Escape LaTeX special chars in all parts
                    def esc(s):
                        for c in ['\\', '$', '%', '&', '#', '_', '{', '}', '^', '~']:
                            s = s.replace(c, '\\' + c)
                        return s
                    line_str = f"[{sim:.2f}]  {esc(before)}$\\bf{{{esc(bold_part)}}}${esc(after)}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--output-dir", type=str,
                        default="paper/Interpreting-VideoLLMs/figures/pvsg_examples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = find_good_examples(args.n_examples)
    print(f"Found {len(examples)} examples")

    for frame_name, nf in examples:
        vid_id = Path(frame_name).stem  # e.g. "pvsg_0042"
        stem = f"{vid_id}_1f_vs_{nf}f"
        print(f"Rendering {frame_name} ({stem})...")
        render_example(frame_name, nf, output_dir / f"{stem}.pdf")

    print("Done!")


if __name__ == "__main__":
    main()
