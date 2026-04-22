#!/usr/bin/env python3
"""Test the judge evaluation pipeline components without API calls.

Verifies: data loading, patch sampling, image preprocessing, prompt formatting.

Run: python tests/test_judge_pipeline.py
"""

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "vl_embedding_spaces/third_party/molmo/llm_judge"))
from prompts import IMAGE_PROMPT_WITH_CROP


RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "data" / "pixmo_cap_100"

MODELS = {
    "Molmo-7B-D": {"dir": "pixmo100_molmo-7b-d", "last_layer": 27},
    "Idefics3-8B": {"dir": "pixmo100_idefics3", "last_layer": 31},
    "Molmo2-8B": {"dir": "pixmo100_molmo2", "last_layer": 35},
    "Qwen2.5-VL-7B": {"dir": "pixmo100_qwen25vl", "last_layer": 27},
}


def test_results_exist():
    """Test: all expected result files exist."""
    for name, cfg in MODELS.items():
        d = RESULTS_DIR / cfg["dir"]
        assert d.exists(), f"Missing results dir: {d}"
        layer_file = d / f"latentlens_layer{cfg['last_layer']}.json"
        assert layer_file.exists(), f"Missing layer file: {layer_file}"
    print("  PASS results_exist: all 4 models have result files")


def test_results_structure():
    """Test: result JSON has expected structure with patches and neighbors."""
    for name, cfg in MODELS.items():
        path = RESULTS_DIR / cfg["dir"] / f"latentlens_layer{cfg['last_layer']}.json"
        with open(path) as f:
            data = json.load(f)

        assert "results" in data, f"{name}: missing 'results' key"
        images = data["results"]
        assert len(images) == 100, f"{name}: expected 100 images, got {len(images)}"

        # Check first image has patches with neighbors
        img0 = images[0]
        patches = img0.get("patches", [])
        assert len(patches) > 0, f"{name}: no patches in image 0"

        p = patches[0]
        assert "patch_row" in p, f"{name}: patch missing 'patch_row'"
        assert "patch_col" in p, f"{name}: patch missing 'patch_col'"
        assert "patch_idx" in p, f"{name}: patch missing 'patch_idx'"

        nbs = p.get("nearest_contextual_neighbors", [])
        assert len(nbs) >= 5, f"{name}: expected >= 5 neighbors, got {len(nbs)}"

        nb = nbs[0]
        assert "token_str" in nb, f"{name}: neighbor missing 'token_str'"
        assert "similarity" in nb, f"{name}: neighbor missing 'similarity'"
        assert "caption" in nb, f"{name}: neighbor missing 'caption'"
        assert "contextual_layer" in nb, f"{name}: neighbor missing 'contextual_layer'"

    print("  PASS results_structure: all models have correct JSON structure")


def test_cross_layer_search():
    """Test: neighbors have contextual_layer field and cross-layer search works at mid layers."""
    for name, cfg in MODELS.items():
        # Use a mid-layer where cross-layer diversity is expected
        mid_layer = 8
        path = RESULTS_DIR / cfg["dir"] / f"latentlens_layer{mid_layer}.json"
        with open(path) as f:
            data = json.load(f)

        ctx_layers = set()
        for img in data["results"][:10]:
            for patch in img["patches"][:10]:
                for nb in patch.get("nearest_contextual_neighbors", []):
                    cl = nb.get("contextual_layer")
                    assert cl is not None, f"{name}: neighbor missing contextual_layer"
                    ctx_layers.add(cl)

        assert len(ctx_layers) > 1, (
            f"{name} layer {mid_layer}: all neighbors from single contextual layer "
            f"{ctx_layers}. Expected cross-layer search."
        )

    print("  PASS cross_layer_search: mid-layer neighbors come from multiple contextual layers")


def test_patch_sampling():
    """Test: can sample valid patches (3x3 bbox fits within grid)."""
    rng = random.Random(42)
    bbox_size = 3

    for name, cfg in MODELS.items():
        path = RESULTS_DIR / cfg["dir"] / f"latentlens_layer{cfg['last_layer']}.json"
        with open(path) as f:
            data = json.load(f)

        img0 = data["results"][0]
        patches = img0["patches"]
        grid_h = max(p["patch_row"] for p in patches) + 1
        grid_w = max(p["patch_col"] for p in patches) + 1

        valid = [p for p in patches
                 if p["patch_row"] + bbox_size <= grid_h
                 and p["patch_col"] + bbox_size <= grid_w]

        assert len(valid) > 0, f"{name}: no valid patches for 3x3 bbox in {grid_h}x{grid_w} grid"
        sampled = rng.choice(valid)
        assert sampled["patch_row"] + bbox_size <= grid_h
        assert sampled["patch_col"] + bbox_size <= grid_w

    print(f"  PASS patch_sampling: all models have valid 3x3 bbox positions")


def test_image_preprocessing():
    """Test: images can be loaded and resized to 512x512 with bbox."""
    if not IMAGES_DIR.exists():
        print("  SKIP image_preprocessing: data/pixmo_cap_100/ not found")
        return

    paths = sorted(p for p in IMAGES_DIR.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert len(paths) >= 100, f"Expected 100 images, found {len(paths)}"

    # Test first image
    img = Image.open(paths[0]).convert("RGB")
    w, h = img.size
    scale = min(512 / w, 512 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (512, 512), (0, 0, 0))
    padded.paste(resized, ((512 - new_w) // 2, (512 - new_h) // 2))

    assert padded.size == (512, 512)

    # Draw bbox
    draw = ImageDraw.Draw(padded)
    draw.rectangle([100, 100, 164, 164], outline="red", width=3)

    # Crop
    crop = padded.crop((100, 100, 164, 164))
    assert crop.size == (64, 64)

    print("  PASS image_preprocessing: load, resize, pad, bbox, crop all work")


def test_prompt_formatting():
    """Test: judge prompt formats correctly with candidate words."""
    candidates = ["dog", "running", "park"]
    prompt = IMAGE_PROMPT_WITH_CROP.format(candidate_words=json.dumps(candidates))

    assert "dog" in prompt
    assert "running" in prompt
    assert "Concrete" in prompt
    assert "Abstract" in prompt
    assert "Global" in prompt
    assert "interpretable" in prompt

    print("  PASS prompt_formatting: IMAGE_PROMPT_WITH_CROP formats correctly")


def test_layer_coverage():
    """Test: all expected layers have result files."""
    expected = {
        "Molmo-7B-D": [0, 1, 4, 8, 16, 24, 26, 27],
        "Idefics3-8B": [0, 1, 4, 8, 16, 24, 30, 31],
        "Molmo2-8B": [0, 1, 4, 8, 16, 24, 32, 34, 35],
        "Qwen2.5-VL-7B": [0, 1, 4, 8, 16, 24, 26, 27],
    }

    for name, layers in expected.items():
        d = RESULTS_DIR / MODELS[name]["dir"]
        for layer in layers:
            path = d / f"latentlens_layer{layer}.json"
            assert path.exists(), f"{name}: missing layer {layer} at {path}"

    print("  PASS layer_coverage: all expected layer files present")


def main():
    print("Judge pipeline tests:")
    test_results_exist()
    test_results_structure()
    test_cross_layer_search()
    test_patch_sampling()
    test_image_preprocessing()
    test_prompt_formatting()
    test_layer_coverage()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
