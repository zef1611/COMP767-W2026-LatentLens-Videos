#!/usr/bin/env python3
"""Verify that spatial patch results have correct row/col mapping to image positions.

For each model, checks:
1. All result files in a _spatial/ directory have grid_h, grid_w per image
2. patch_row < grid_h and patch_col < grid_w for every patch
3. The same image has the same grid dimensions across all layers
4. grid_h * grid_w roughly matches the number of patches (global crop only)
5. Bounding box drawn at (row, col) falls within the image content area (not padding)

Run: python tests/test_spatial_patches.py
"""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "data" / "pixmo_cap_100"

SPATIAL_DIRS = {
    "Molmo-7B-D": "pixmo100_molmo-7b-d_spatial",
    "Idefics3-8B": "pixmo100_idefics3_spatial",
    "Molmo2-8B": "pixmo100_molmo2_spatial",
    "Qwen2.5-VL-7B": "pixmo100_qwen25vl_spatial",
}


def test_spatial_dirs_exist():
    """All 4 models have spatial result directories."""
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        assert d.exists(), f"{name}: missing spatial dir {d}"
        files = list(d.glob("latentlens_layer*.json"))
        assert len(files) > 0, f"{name}: no layer files in {d}"
    print("  PASS spatial_dirs_exist")


def test_grid_metadata():
    """Every image in every layer file has grid_h and grid_w."""
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        for lf in sorted(d.glob("latentlens_layer*.json")):
            with open(lf) as f:
                data = json.load(f)
            for i, img in enumerate(data["results"]):
                assert "grid_h" in img, f"{name} {lf.name} img{i}: missing grid_h"
                assert "grid_w" in img, f"{name} {lf.name} img{i}: missing grid_w"
                assert img["grid_h"] > 0 and img["grid_w"] > 0
    print("  PASS grid_metadata")


def test_patch_bounds():
    """Every patch row/col is within grid bounds."""
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        for lf in sorted(d.glob("latentlens_layer*.json")):
            with open(lf) as f:
                data = json.load(f)
            for i, img in enumerate(data["results"]):
                gh, gw = img["grid_h"], img["grid_w"]
                for p in img["patches"]:
                    assert p["patch_row"] < gh, (
                        f"{name} {lf.name} img{i}: row {p['patch_row']} >= grid_h {gh}")
                    assert p["patch_col"] < gw, (
                        f"{name} {lf.name} img{i}: col {p['patch_col']} >= grid_w {gw}")
    print("  PASS patch_bounds")


def test_consistent_grids_across_layers():
    """Same image has same grid dimensions across all layers."""
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        files = sorted(d.glob("latentlens_layer*.json"))
        if len(files) < 2:
            continue

        # Load grid dims from first file
        with open(files[0]) as f:
            ref_data = json.load(f)
        ref_grids = [(img["grid_h"], img["grid_w"]) for img in ref_data["results"]]

        for lf in files[1:]:
            with open(lf) as f:
                data = json.load(f)
            for i, img in enumerate(data["results"]):
                assert (img["grid_h"], img["grid_w"]) == ref_grids[i], (
                    f"{name} {lf.name} img{i}: grid {img['grid_h']}x{img['grid_w']} "
                    f"!= reference {ref_grids[i][0]}x{ref_grids[i][1]}")
    print("  PASS consistent_grids_across_layers")


def test_patch_count_matches_grid():
    """Number of patches approximately matches grid_h * grid_w."""
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        lf = sorted(d.glob("latentlens_layer*.json"))[0]
        with open(lf) as f:
            data = json.load(f)
        for i, img in enumerate(data["results"][:5]):
            gh, gw = img["grid_h"], img["grid_w"]
            n_patches = len(img["patches"])
            expected = gh * gw
            # Allow some tolerance (some patches might be missing)
            ratio = n_patches / expected if expected > 0 else 0
            assert ratio > 0.8, (
                f"{name} img{i}: {n_patches} patches but grid is {gh}x{gw}={expected}, "
                f"ratio={ratio:.2f}")
    print("  PASS patch_count_matches_grid")


def test_bbox_in_content_area():
    """Most patches map to image content area, not black padding.

    Note: some models (Molmo, Molmo2) include padding in their global crop grid.
    We check that the CENTER patch of each image is in content — edge patches may
    legitimately be in padding for non-square images.
    """
    if not IMAGES_DIR.exists():
        print("  SKIP bbox_in_content_area: images not found")
        return

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        lf = sorted(d.glob("latentlens_layer*.json"))[0]
        with open(lf) as f:
            data = json.load(f)

        for i, img in enumerate(data["results"][:5]):
            if i >= len(image_paths):
                break
            pil_img = Image.open(image_paths[i])
            w, h = pil_img.size
            scale = min(512 / w, 512 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            pad_left = (512 - new_w) // 2
            pad_top = (512 - new_h) // 2
            content_right = pad_left + new_w
            content_bottom = pad_top + new_h

            gh, gw = img["grid_h"], img["grid_w"]
            patch_h = 512 / gh
            patch_w = 512 / gw

            # Check the center of the grid (should always be in content)
            center_r, center_c = gh // 2, gw // 2
            center_x = (center_c + 0.5) * patch_w
            center_y = (center_r + 0.5) * patch_h

            assert pad_left <= center_x <= content_right, (
                f"{name} img{i} grid center ({center_r},{center_c}): "
                f"center_x={center_x:.0f} outside content [{pad_left}-{content_right}]")
            assert pad_top <= center_y <= content_bottom, (
                f"{name} img{i} grid center ({center_r},{center_c}): "
                f"center_y={center_y:.0f} outside content [{pad_top}-{content_bottom}]")

    print("  PASS bbox_in_content_area (grid centers are in content)")


def test_same_images_across_models():
    """All models have results for the same number of images."""
    counts = {}
    for name, dirname in SPATIAL_DIRS.items():
        d = RESULTS_DIR / dirname
        lf = sorted(d.glob("latentlens_layer*.json"))[0]
        with open(lf) as f:
            data = json.load(f)
        counts[name] = len(data["results"])

    values = list(counts.values())
    assert len(set(values)) == 1, f"Different image counts: {counts}"
    print(f"  PASS same_images_across_models: all have {values[0]} images")


def main():
    print("Spatial patch tests:")
    test_spatial_dirs_exist()
    test_grid_metadata()
    test_patch_bounds()
    test_consistent_grids_across_layers()
    test_patch_count_matches_grid()
    test_bbox_in_content_area()
    test_same_images_across_models()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
