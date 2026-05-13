import json
import gzip
from pathlib import Path
from collections import defaultdict
from .base import RecDataset, UserHistory


class AmazonReviewsDataset(RecDataset):
    """Amazon Reviews 2023: https://amazon-reviews-2023.github.io/"""

    def load(self) -> None:
        data_dir = Path(self.data_dir) / self.cfg.name
        category = self.cfg.category

        self._item_meta = self._load_meta(data_dir / f"meta_{category}.jsonl.gz")

        user_histories: dict[str, list[dict]] = defaultdict(list)
        with gzip.open(data_dir / f"{category}.jsonl.gz", "rt") as f:
            for line in f:
                rev = json.loads(line)
                uid = rev.get("user_id", "")
                iid = rev.get("asin", "")
                if not uid or not iid or iid not in self._item_meta:
                    continue
                user_histories[uid].append({
                    "item_id": iid,
                    "rating": float(rev.get("rating", 0.0)),
                    "timestamp": int(rev.get("timestamp", 0)),
                    "text": rev.get("text", ""),
                })

        max_words = getattr(self.cfg, "max_review_words", 50)
        for uid, interactions in user_histories.items():
            interactions.sort(key=lambda x: x["timestamp"])
            if len(interactions) < self.cfg.min_history_len:
                continue
            interactions = interactions[-self.cfg.max_history_len:]
            review_texts = (
                [" ".join(i["text"].split()[:max_words]) for i in interactions]
                if getattr(self.cfg, "include_review_text", False) else None
            )
            self._users[uid] = UserHistory(
                user_id=uid,
                item_ids=[i["item_id"] for i in interactions],
                ratings=[i["rating"] for i in interactions] if getattr(self.cfg, "include_ratings", False) else None,
                timestamps=[i["timestamp"] for i in interactions],
                item_metadata=self._item_meta,
                extra={"review_texts": review_texts} if review_texts else None,
            )

    def _load_meta(self, path: Path) -> dict[str, dict]:
        meta = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                item = json.loads(line)
                iid = item.get("parent_asin") or item.get("asin", "")
                if not iid:
                    continue
                meta[iid] = {
                    "title": item.get("title", ""),
                    "categories": item.get("categories", []),
                    "description": " ".join(item.get("description", [])),
                    "brand": item.get("store", ""),
                }
        return meta
