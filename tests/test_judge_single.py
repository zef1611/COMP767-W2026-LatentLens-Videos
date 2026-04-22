#!/usr/bin/env python3
"""End-to-end test: run the judge on 1 real patch from each model.

Loads actual LatentLens results, prepares the image+bbox+candidates,
calls Gemini, and prints the judge's response. Use this to visually
verify the pipeline before a full run.

Run: python tests/test_judge_single.py
"""

import json
import random
import sys
from pathlib import Path

import google.generativeai as genai
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "vl_embedding_spaces/third_party/molmo/llm_judge"))
from prompts import IMAGE_PROMPT_WITH_CROP

RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "data" / "pixmo_cap_100"
BBOX_SIZE = 3

MODELS = {
    "Molmo-7B-D": {"dir": "pixmo100_molmo-7b-d", "layer": 27},
    "Idefics3-8B": {"dir": "pixmo100_idefics3", "layer": 31},
    "Molmo2-8B": {"dir": "pixmo100_molmo2", "layer": 35},
    "Qwen2.5-VL-7B": {"dir": "pixmo100_qwen25vl", "layer": 27},
}


def resize_and_pad_pil(pil_img, size=512):
    w, h = pil_img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return padded


def main():
    key_file = ROOT / "gemini_key.txt"
    if not key_file.exists():
        print(f"SKIP: {key_file} not found")
        sys.exit(0)

    api_key = key_file.read_text().strip()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    rng = random.Random(42)

    print("End-to-end judge test (1 patch per model):\n")

    for name, cfg in MODELS.items():
        path = RESULTS_DIR / cfg["dir"] / f"latentlens_layer{cfg['layer']}.json"
        with open(path) as f:
            data = json.load(f)

        # Pick image 0, sample a valid patch
        img_data = data["results"][0]
        patches = img_data["patches"]
        grid_h = max(p["patch_row"] for p in patches) + 1
        grid_w = max(p["patch_col"] for p in patches) + 1

        valid = [p for p in patches
                 if p["patch_row"] + BBOX_SIZE <= grid_h
                 and p["patch_col"] + BBOX_SIZE <= grid_w]
        patch = rng.choice(valid)

        row, col = patch["patch_row"], patch["patch_col"]
        nbs = patch.get("nearest_contextual_neighbors", [])[:5]
        candidates = [nb["token_str"].strip() for nb in nbs if nb.get("token_str", "").strip()]

        # Prepare image
        processed = resize_and_pad_pil(Image.open(image_paths[0]).convert("RGB"))
        patch_h = 512 / grid_h
        patch_w = 512 / grid_w
        left = max(0, col * patch_w)
        top = max(0, row * patch_h)
        right = min(512, (col + BBOX_SIZE) * patch_w)
        bottom = min(512, (row + BBOX_SIZE) * patch_h)

        img_bbox = processed.copy()
        ImageDraw.Draw(img_bbox).rectangle([left, top, right, bottom], outline="red", width=3)
        crop = processed.crop((int(left), int(top), int(right), int(bottom)))

        prompt = IMAGE_PROMPT_WITH_CROP.format(candidate_words=json.dumps(candidates))

        # Call Gemini
        resp = model.generate_content([img_bbox, crop, prompt])
        text = resp.text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
        else:
            parsed = {"error": "no JSON", "raw": text[:200]}

        print(f"  {name} (layer {cfg['layer']}, patch [{row},{col}], grid {grid_h}x{grid_w}):")
        print(f"    Candidates: {candidates}")
        print(f"    Interpretable: {parsed.get('interpretable', '?')}")
        print(f"    Concrete: {parsed.get('concrete_words', [])}")
        print(f"    Abstract: {parsed.get('abstract_words', [])}")
        print(f"    Global: {parsed.get('global_words', [])}")
        print(f"    Reasoning: {parsed.get('reasoning', '?')[:150]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
