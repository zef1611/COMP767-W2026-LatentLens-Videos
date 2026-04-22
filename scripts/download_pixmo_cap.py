#!/usr/bin/env python3
"""Download images from PixMo-Cap dataset.

Downloads images from the allenai/pixmo-cap HuggingFace dataset by URL
and saves them as JPEGs. Skips corrupted or unavailable images.

Usage:
    python scripts/download_pixmo_cap.py \
        --output-dir $SCRATCH/latentlens/pixmo_cap_500 \
        --num-images 500
"""

import argparse
import io
import logging
import os
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_image(url: str, output_path: Path, timeout: int = 15) -> bool:
    """Download an image from URL and save as JPEG."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(output_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    scratch = os.environ.get("SCRATCH", "data")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(f"{scratch}/latentlens/pixmo_cap_500"),
    )
    parser.add_argument("--num-images", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(args.output_dir.glob("*.jpg")))
    log.info(f"Output: {args.output_dir}")
    log.info(f"Already have {existing} images, targeting {args.num_images}")

    if existing >= args.num_images:
        log.info("Already have enough images, nothing to do.")
        return

    log.info("Loading PixMo-Cap train split (streaming)...")
    ds = load_dataset("allenai/pixmo-cap", split="train", streaming=True)

    count = existing
    skipped = 0
    for sample in ds:
        if count >= args.num_images:
            break

        out_path = args.output_dir / f"pixmo_{count:04d}.jpg"
        if out_path.exists():
            count += 1
            continue

        url = sample["image_url"]
        ok = download_image(url, out_path)

        if ok:
            count += 1
            if count % 50 == 0:
                log.info(f"  Downloaded {count}/{args.num_images} ({skipped} skipped)")
        else:
            skipped += 1
            if skipped % 20 == 0:
                log.warning(f"  {skipped} images skipped so far")

    log.info(f"Done: {count} images saved, {skipped} skipped")


if __name__ == "__main__":
    main()
