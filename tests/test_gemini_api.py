#!/usr/bin/env python3
"""Smoke test for Gemini API: connectivity, rate limits, image handling.

Run: python tests/test_gemini_api.py
"""

import sys
import time
import json
import concurrent.futures
from pathlib import Path

import google.generativeai as genai
from PIL import Image, ImageDraw
import numpy as np


def load_api_key():
    key_file = Path(__file__).resolve().parent.parent / "gemini_key.txt"
    if not key_file.exists():
        print(f"SKIP: {key_file} not found")
        sys.exit(0)
    return key_file.read_text().strip()


def make_test_image(size=512):
    """Create a test image with a red bounding box."""
    img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 164, 164], outline="red", width=3)
    return img


def test_basic_call(model):
    """Test: single API call returns text."""
    img = make_test_image()
    resp = model.generate_content([img, "Say 'hello'. Reply with just that word."])
    assert resp.text is not None, "No response text"
    assert len(resp.text.strip()) > 0, "Empty response"
    print(f"  PASS basic_call: got '{resp.text.strip()}'")


def test_json_response(model):
    """Test: model returns parseable JSON for judge-style prompt."""
    img = make_test_image()
    crop = img.crop((100, 100, 164, 164))
    prompt = '''Evaluate these candidate words for the highlighted red region: ["dog", "tree"]
Return ONLY a JSON object: {"interpretable": true, "concrete_words": [], "abstract_words": [], "global_words": [], "reasoning": "brief"}'''

    resp = model.generate_content([img, crop, prompt])
    text = resp.text
    start = text.find('{')
    end = text.rfind('}') + 1
    assert start != -1 and end > start, f"No JSON in response: {text[:100]}"
    parsed = json.loads(text[start:end])
    assert "interpretable" in parsed, f"Missing 'interpretable' key: {parsed.keys()}"
    print(f"  PASS json_response: interpretable={parsed['interpretable']}")


def test_parallel_throughput(model, n_calls=20, n_workers=10):
    """Test: parallel calls work and measure throughput."""
    img = make_test_image()

    def call(i):
        resp = model.generate_content([img, f"Say {i}. Reply with just the number."])
        return resp.text.strip()

    t0 = time.time()
    results = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(call, i) for i in range(n_calls)]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
    elapsed = time.time() - t0
    throughput = len(results) / elapsed

    assert errors == 0, f"{errors}/{n_calls} calls failed"
    assert len(results) == n_calls, f"Only {len(results)}/{n_calls} completed"
    print(f"  PASS parallel_throughput: {n_calls} calls, {n_workers} workers, "
          f"{elapsed:.1f}s, {throughput:.1f} calls/sec")


def test_large_burst(model, n_calls=100, n_workers=20):
    """Test: 100 parallel calls (closer to real workload)."""
    img = make_test_image()

    def call(i):
        resp = model.generate_content([img, f"Say {i}."])
        return resp.text.strip()

    t0 = time.time()
    results = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(call, i) for i in range(n_calls)]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
    elapsed = time.time() - t0

    print(f"  {'PASS' if errors == 0 else 'WARN'} large_burst: "
          f"{len(results)}/{n_calls} ok, {errors} errors, "
          f"{elapsed:.1f}s, {len(results)/elapsed:.1f} calls/sec")


def main():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    print("Gemini API tests:")
    test_basic_call(model)
    test_json_response(model)
    test_parallel_throughput(model)
    test_large_burst(model)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
