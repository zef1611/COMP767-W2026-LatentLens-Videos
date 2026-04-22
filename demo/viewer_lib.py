#!/usr/bin/env python3
"""
Shared library for viewer generation.

Common functions used by create_viewer.py. Adapted from the LatentLens (image) project
but generalized — no model-specific preprocessors, no olmo/molmo imports.

Key functions:
- Image/frame processing: pil_image_to_base64()
- Grid utilities: patch_idx_to_row_col(), get_grid_dimensions()
- Data loading: load_analysis_data(), extract_patches_from_data()
- Patch processing: process_nn_patch(), process_logit_patch(), process_contextual_patch()
"""

import base64
import re as _re
import html
import io
import json
import math
from PIL import Image
from typing import Tuple, Dict, List, Any
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def escape_for_html(text: str) -> str:
    """Properly escape text for HTML."""
    if not text:
        return ""
    return html.escape(text, quote=True)


def highlight_token_in_caption(caption: str, token: str) -> str:
    """Highlight the full word containing the BPE token in the caption.

    For subword tokens (e.g. "st"), prefers occurrences where the token
    is embedded inside a larger word (e.g. "blood**st**ained") over
    occurrences at word boundaries (e.g. "**st**ared"). This matches
    BPE tokenization behavior where continuation tokens are subwords.
    """
    if not caption or not token:
        return caption
    token_clean = token.strip()
    if not token_clean:
        return caption

    low_cap = caption.lower()
    low_tok = token_clean.lower()

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def expand_to_word(pos: int) -> tuple:
        """Expand match at pos to full word boundaries. Returns (start, end)."""
        s, e = pos, pos + len(low_tok)
        while s > 0 and is_word_char(caption[s - 1]):
            s -= 1
        while e < len(caption) and is_word_char(caption[e]):
            e += 1
        return s, e

    # Find all occurrences
    matches = []
    start = 0
    while True:
        idx = low_cap.find(low_tok, start)
        if idx == -1:
            break
        ws, we = expand_to_word(idx)
        word = caption[ws:we]
        is_subword = (ws < idx) or (we > idx + len(low_tok))  # embedded in larger word
        is_exact = word.lower() == low_tok  # token IS the full word
        matches.append((idx, ws, we, word, is_subword, is_exact))
        start = idx + 1

    if not matches:
        return caption

    # Pick the best match. BPE continuation tokens (no leading space) are
    # typically mid-word, so prefer matches where the token does NOT start
    # the word (i.e., there are characters before it in the word).
    # Score: mid-word > word-start-subword > exact-word
    def score(m):
        idx, ws, we, word, is_subword, is_exact = m
        if is_exact:
            return 0  # exact word match (lowest priority for subword tokens)
        mid_word = ws < idx  # token starts after word start
        return 2 if mid_word else 1

    best = max(matches, key=score)

    idx, ws, we, word, is_subword, is_exact = best
    return caption[:ws] + f'<span class="highlight">{caption[ws:we]}</span>' + caption[we:]


def pil_image_to_base64(img: Image.Image, max_size: int = 512, quality: int = 75) -> str:
    """Convert PIL Image to base64 string for embedding in HTML."""
    if img is None:
        return ""
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to display size to save space
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def patch_idx_to_row_col(patch_idx: int, patches_per_chunk: int) -> Tuple[int, int]:
    """Convert patch index to (row, col) assuming a square grid."""
    grid_size = int(math.sqrt(patches_per_chunk))
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return row, col


def get_grid_dimensions(image_data: Dict, default_grid_size: int = 16) -> Tuple[int, int, int]:
    """Determine grid dimensions from image data.

    Returns (grid_rows, grid_cols, patches_per_chunk).
    """
    patches = extract_patches_from_data(image_data)
    if not patches:
        return default_grid_size, default_grid_size, default_grid_size * default_grid_size

    patches_per_chunk = len(patches)
    max_row = max((p.get("patch_row", 0) for p in patches), default=0)
    max_col = max((p.get("patch_col", 0) for p in patches), default=0)

    if max_row > 0 or max_col > 0:
        return max_row + 1, max_col + 1, patches_per_chunk
    grid_size = int(math.sqrt(patches_per_chunk))
    return grid_size, grid_size, patches_per_chunk


def extract_patches_from_data(image_data: Dict) -> List[Dict]:
    """Extract patches from image data, handling both Format A and Format B.

    Format A: {chunks: [{patches: [...]}]}
    Format B: {patches: [...]}
    """
    if "chunks" in image_data:
        all_patches = []
        for chunk in image_data.get("chunks", []):
            all_patches.extend(chunk.get("patches", []))
        return all_patches
    return image_data.get("patches", [])


def load_analysis_data(json_path: Path, num_images: int = 0) -> List[Dict]:
    """Load analysis results from a JSON file.

    Handles both {results: [...]} and {splits: {validation: {images: [...]}}} formats.
    If num_images > 0, truncate to that many entries.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    images = data.get("results", [])
    if not images:
        images = data.get("splits", {}).get("validation", {}).get("images", [])

    if num_images > 0:
        images = images[:num_images]
    return images


def process_nn_patch(patch: Dict, grid_size: int) -> Dict:
    """Process a single nearest-neighbor patch into unified format."""
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)
    neighbors = patch.get("nearest_neighbors", []) or patch.get("top_neighbors", [])
    nn_list = [
        {"rank": i + 1, "token": escape_for_html(nn.get("token", "")), "similarity": nn.get("similarity", 0.0)}
        for i, nn in enumerate(neighbors[:5])
    ]
    return {"patch_idx": patch_idx, "row": row, "col": col, "neighbors": nn_list}


def process_logit_patch(patch: Dict, grid_size: int) -> Dict:
    """Process a single LogitLens patch into unified format."""
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)
    pred_list = [
        {
            "rank": i + 1,
            "token": escape_for_html(pred.get("token", "")),
            "logit": pred.get("logit", 0.0),
            "token_id": pred.get("token_id", 0),
        }
        for i, pred in enumerate(patch.get("top_predictions", [])[:5])
    ]
    return {"patch_idx": patch_idx, "row": row, "col": col, "predictions": pred_list}


def process_contextual_patch(patch: Dict, grid_size: int) -> Dict:
    """Process a single contextual (LatentLens) patch into unified format."""
    patch_idx = patch.get("patch_idx", -1)
    row = patch.get("patch_row", patch_idx // grid_size)
    col = patch.get("patch_col", patch_idx % grid_size)

    ctx_list = []
    for i, neighbor in enumerate(patch.get("nearest_contextual_neighbors", [])[:5]):
        token_str = escape_for_html(neighbor.get("token_str", ""))
        caption = escape_for_html(neighbor.get("caption", ""))
        highlighted_caption = caption.replace(token_str, f'<span class="highlight">{token_str}</span>', 1)
        ctx_list.append(
            {
                "rank": i + 1,
                "token": token_str,
                "caption": highlighted_caption,
                "similarity": neighbor.get("similarity", 0.0),
                "contextual_layer": neighbor.get("contextual_layer", None),
                "position": neighbor.get("position", 0),
            }
        )
    return {"patch_idx": patch_idx, "row": row, "col": col, "contextual_neighbors": ctx_list}
