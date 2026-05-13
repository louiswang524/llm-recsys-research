from transformers import PreTrainedTokenizer
from .base import BaseItemTokenizer


class TextItemTokenizer(BaseItemTokenizer):
    """Items as plain text — no vocabulary extension needed."""

    def build(self, item_ids: list[str], item_meta: dict[str, dict]) -> None:
        self._item_meta = item_meta

    def item_to_tokens(self, item_id: str) -> list[str]:
        title = self._item_meta.get(item_id, {}).get("title", item_id)
        max_words = getattr(self.cfg, "max_title_tokens", 32)
        return [" ".join(title.split()[:max_words])]

    def extend_tokenizer(self, tokenizer: PreTrainedTokenizer) -> int:
        return 0
