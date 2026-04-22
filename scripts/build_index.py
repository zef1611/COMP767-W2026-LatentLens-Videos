#!/usr/bin/env python3
"""Build a LatentLens contextual embedding index for a model.

Supports both standard LLMs and VLMs (loads via multiple Auto classes).

Usage:
    python scripts/build_index.py \
        --model allenai/Molmo-7B-D-0924 \
        --output indices/molmo-7b-d \
        --device cuda:0
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoTokenizer,
)

from latentlens import ContextualIndex
from latentlens.extract import auto_layers, load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).resolve().parent.parent.parent / "vl_embedding_spaces/third_party/molmo/latentlens_release/concepts.txt"


def load_model_any(model_name, device, dtype):
    """Load model trying multiple Auto classes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)
    for auto_cls in [AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq]:
        try:
            model = auto_cls.from_pretrained(model_name, **load_kwargs)
            log.info(f"Loaded with {auto_cls.__name__}")
            break
        except (ValueError, KeyError, TypeError):
            continue
    else:
        raise RuntimeError(f"Could not load {model_name}")

    model = model.to(device).eval()
    return model, tokenizer


def get_num_layers(model):
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers
    raise AttributeError("Cannot determine num_hidden_layers")


def main():
    parser = argparse.ArgumentParser(description="Build LatentLens contextual index")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-contexts", type=int, default=50)
    parser.add_argument("--layers", type=str, default=None)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    log.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_any(args.model, device, dtype)

    n_layers = get_num_layers(model)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = auto_layers(n_layers)
    log.info(f"Layers: {layers} (model has {n_layers})")

    texts = load_corpus(str(args.corpus))
    log.info(f"Corpus: {len(texts)} sentences")

    # Build index: same logic as latentlens.extract.build_index
    layer_embeddings = defaultdict(list)
    layer_metadata = defaultdict(list)
    seen_prefixes = set()
    token_counts = defaultdict(int)

    for batch_start in tqdm(range(0, len(texts), args.batch_size), desc="Building index", unit="batch"):
        batch_texts = texts[batch_start : batch_start + args.batch_size]
        encodings = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for sent_idx in range(input_ids.shape[0]):
            sent_ids = input_ids[sent_idx]
            mask = attention_mask[sent_idx]
            valid_len = mask.sum().item()

            for pos in range(2, valid_len):
                prefix = tuple(sent_ids[:pos + 1].tolist())
                prefix_hash = hash(prefix)
                if prefix_hash in seen_prefixes:
                    continue

                token_id = sent_ids[pos].item()
                token_str = tokenizer.decode([token_id])

                if token_counts[token_str] >= args.max_contexts:
                    continue

                seen_prefixes.add(prefix_hash)
                token_counts[token_str] += 1

                caption = batch_texts[sent_idx] if sent_idx < len(batch_texts) else ""
                meta = {"token_str": token_str, "token_id": token_id, "caption": caption, "position": pos}

                for layer_idx in layers:
                    emb = hidden_states[layer_idx][sent_idx, pos, :].cpu()
                    layer_embeddings[layer_idx].append(emb)
                    layer_metadata[layer_idx].append(meta)

        del hidden_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Assemble and normalize
    layers_data = {}
    for layer_idx in layers:
        if not layer_embeddings[layer_idx]:
            continue
        emb_tensor = torch.stack(layer_embeddings[layer_idx])
        emb_tensor = F.normalize(emb_tensor.float(), dim=-1)
        layers_data[layer_idx] = {"embeddings": emb_tensor, "metadata": layer_metadata[layer_idx]}

    index = ContextualIndex(layers_data)
    args.output.mkdir(parents=True, exist_ok=True)
    index.save(str(args.output))
    log.info(f"Index saved to {args.output}")


if __name__ == "__main__":
    main()
