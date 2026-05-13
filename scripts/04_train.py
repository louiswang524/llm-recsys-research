#!/usr/bin/env python3
"""Main training entry point. Stage (cpt/sft/dpo) is set in configs/training/."""

import sys
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_recsys.data.datasets.base import UserHistory
from llm_recsys.data.item_tokenizer.text_tokenizer import TextItemTokenizer
from llm_recsys.data.item_tokenizer.semantic_id import SemanticIDTokenizer
from llm_recsys.data.verbalization.templates import get_verbalizer
from llm_recsys.data.verbalization.formatter import InstructionFormatter
from llm_recsys.data.collator import RecDataCollator
from llm_recsys.data.splits import _slice_history
from llm_recsys.models.llm_rec import LLMRecModel
from llm_recsys.training.stages import run_stage


class SequenceDataset(TorchDataset):
    """Every (context[:i], item[i]) pair in the user history is one training example."""

    def __init__(self, users: dict[str, UserHistory], formatter: InstructionFormatter):
        self.samples: list[dict] = []
        for history in users.values():
            if len(history.item_ids) < 2:
                continue
            for i in range(1, len(history.item_ids)):
                context = _slice_history(history, end=i)
                target_id = history.item_ids[i]
                target_rating = history.ratings[i] if history.ratings else 0.0
                sample = formatter.format(context, target_id)
                sample["rating"] = target_rating
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    processed_dir = Path(cfg.processed_dir) / cfg.data.name
    vocab_dir = processed_dir / "vocab"

    with open(processed_dir / "split.pkl", "rb") as f:
        split = pickle.load(f)

    is_semantic = "SemanticID" in cfg.model.item_tokenizer._target_
    item_tok_cls = SemanticIDTokenizer if is_semantic else TextItemTokenizer
    item_tokenizer = item_tok_cls.load(str(vocab_dir / "item_tokenizer.pkl"))

    tokenizer = AutoTokenizer.from_pretrained(str(vocab_dir / "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verbalizer = get_verbalizer(cfg.data.get("verbalization_template", "rating_history"))
    formatter = InstructionFormatter(verbalizer, tokenizer, stage=cfg.training.stage)

    print("Building train / val datasets ...")
    train_ds = SequenceDataset(split.train, formatter)
    val_ds = SequenceDataset(split.val, formatter)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    # Number of new tokens added to LLM vocab by the item tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    vocab_delta = len(tokenizer) - len(base_tokenizer)
    model = LLMRecModel(cfg, vocab_size_delta=max(0, vocab_delta))

    collator = RecDataCollator(tokenizer, max_length=cfg.max_length)

    run_stage(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=cfg.output_dir,
        data_collator=collator,
    )


if __name__ == "__main__":
    main()
