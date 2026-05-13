from .base import BaseVerbalizer
from ..datasets.base import UserHistory


class RatingHistoryVerbalizer(BaseVerbalizer):
    """Full verbalization: title, genre/category, rating, optional review excerpt."""

    def verbalize(self, history: UserHistory, target_item_id: str | None = None) -> str:
        review_texts = (history.extra or {}).get("review_texts", [])
        ratings = history.ratings or []
        lines = []
        for idx, item_id in enumerate(history.item_ids, 1):
            meta = history.item_metadata.get(item_id, {})
            title = meta.get("title", item_id)
            genre = self._genre_str(meta)
            line = f"{idx}. {title}{genre}"
            if idx - 1 < len(ratings):
                line += f" — {ratings[idx - 1]:.0f}/5"
            if idx - 1 < len(review_texts) and review_texts[idx - 1]:
                line += f'\n   "{review_texts[idx - 1]}"'
            lines.append(line)

        prompt = "User history:\n" + "\n".join(lines) + "\n\nPredict the next item.\nNext:"
        if target_item_id is not None:
            target_title = history.item_metadata.get(target_item_id, {}).get("title", target_item_id)
            prompt += f" {target_title}"
        return prompt

    def _genre_str(self, meta: dict) -> str:
        genres = meta.get("genres", "") or meta.get("categories", "")
        if isinstance(genres, list):
            genres = ", ".join(str(g) for g in genres[:2])
        return f" [{genres}]" if genres else ""


class MinimalVerbalizer(BaseVerbalizer):
    """Title-only chain — fast baseline for ablations."""

    def verbalize(self, history: UserHistory, target_item_id: str | None = None) -> str:
        titles = [
            history.item_metadata.get(iid, {}).get("title", iid)
            for iid in history.item_ids
        ]
        prompt = "History: " + " → ".join(titles) + "\nNext:"
        if target_item_id is not None:
            target_title = history.item_metadata.get(target_item_id, {}).get("title", target_item_id)
            prompt += f" {target_title}"
        return prompt


_REGISTRY: dict[str, type[BaseVerbalizer]] = {
    "rating_history": RatingHistoryVerbalizer,
    "minimal": MinimalVerbalizer,
}


def get_verbalizer(name: str) -> BaseVerbalizer:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown verbalizer '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
