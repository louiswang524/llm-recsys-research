#!/usr/bin/env python3
"""Load dataset, apply train/val/test split, save processed files."""

import sys
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_recsys.data.datasets.movielens import MovieLensDataset
from llm_recsys.data.datasets.amazon import AmazonReviewsDataset
from llm_recsys.data.splits import split_users


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.processed_dir) / cfg.data.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    target = cfg.data._target_
    dataset_cls = MovieLensDataset if "movielens" in target else AmazonReviewsDataset
    dataset = dataset_cls(cfg.data, cfg.data_dir)

    print(f"Loading {cfg.data.name} ...")
    dataset.load()
    print(f"  {len(dataset.users):,} users  |  {len(dataset.all_item_ids):,} items")

    split = split_users(dataset.users, cfg.data.split)
    print(f"  split → train={len(split.train):,}  val={len(split.val):,}  test={len(split.test):,}")

    for name, obj in [("split", split), ("item_meta", dataset.item_meta),
                      ("all_item_ids", dataset.all_item_ids)]:
        with open(processed_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    print(f"Saved to {processed_dir}")


if __name__ == "__main__":
    main()
