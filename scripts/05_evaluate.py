#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split.

Usage:
  python scripts/05_evaluate.py checkpoint=outputs/2024-01-01_12-00-00/final
"""

import sys
import pickle
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_recsys.data.item_tokenizer.text_tokenizer import TextItemTokenizer
from llm_recsys.data.item_tokenizer.semantic_id import SemanticIDTokenizer
from llm_recsys.data.verbalization.templates import get_verbalizer
from llm_recsys.evaluation.candidate_scoring import CandidateScorer
from llm_recsys.evaluation.evaluator import RecEvaluator
from llm_recsys.models.llm_rec import LLMRecModel


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Pass checkpoint=<path> on the command line.")

    processed_dir = Path(cfg.processed_dir) / cfg.data.name
    vocab_dir = processed_dir / "vocab"
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    with open(processed_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)
    with open(processed_dir / "all_item_ids.pkl", "rb") as f:
        all_item_ids = pickle.load(f)

    is_semantic = "SemanticID" in cfg.model.item_tokenizer._target_
    item_tok_cls = SemanticIDTokenizer if is_semantic else TextItemTokenizer
    item_tokenizer = item_tok_cls.load(str(vocab_dir / "item_tokenizer.pkl"))
    tokenizer = AutoTokenizer.from_pretrained(str(vocab_dir / "tokenizer"))

    model = LLMRecModel.from_pretrained(checkpoint, cfg)
    model.eval()

    verbalizer = get_verbalizer(cfg.data.get("verbalization_template", "rating_history"))
    scorer = CandidateScorer(tokenizer, item_tokenizer, device=device)
    evaluator = RecEvaluator(cfg, split.test, all_item_ids, verbalizer, scorer)

    print(f"\nEvaluating {checkpoint} on {cfg.data.name} test split ...\n")
    metrics = evaluator.evaluate(model)

    print("=== Results ===")
    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
