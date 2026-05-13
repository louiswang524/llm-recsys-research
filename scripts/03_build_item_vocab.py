#!/usr/bin/env python3
"""Fit item tokenizer (text or RQ-VAE semantic IDs) and extend the LLM tokenizer."""

import sys
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_recsys.data.item_tokenizer.text_tokenizer import TextItemTokenizer
from llm_recsys.data.item_tokenizer.semantic_id import SemanticIDTokenizer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.processed_dir) / cfg.data.name
    vocab_dir = processed_dir / "vocab"
    vocab_dir.mkdir(exist_ok=True)

    with open(processed_dir / "all_item_ids.pkl", "rb") as f:
        all_item_ids = pickle.load(f)
    with open(processed_dir / "item_meta.pkl", "rb") as f:
        item_meta = pickle.load(f)

    tok_cfg = cfg.model.item_tokenizer
    is_semantic = "SemanticID" in tok_cfg._target_
    item_tokenizer = SemanticIDTokenizer(tok_cfg) if is_semantic else TextItemTokenizer(tok_cfg)

    print(f"Building {tok_cfg._target_} for {len(all_item_ids):,} items ...")
    item_tokenizer.build(all_item_ids, item_meta)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = item_tokenizer.extend_tokenizer(tokenizer)
    print(f"Added {n_added:,} tokens → tokenizer size {len(tokenizer):,}")

    item_tokenizer.save(str(vocab_dir / "item_tokenizer.pkl"))
    tokenizer.save_pretrained(str(vocab_dir / "tokenizer"))
    print(f"Saved to {vocab_dir}")


if __name__ == "__main__":
    main()
