import pickle
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer


class BaseItemTokenizer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def build(self, item_ids: list[str], item_meta: dict[str, dict]) -> None:
        """Fit tokenizer on the full item corpus."""
        ...

    @abstractmethod
    def item_to_tokens(self, item_id: str) -> list[str]:
        """Token string(s) representing this item (text phrase or special token(s))."""
        ...

    @abstractmethod
    def extend_tokenizer(self, tokenizer: PreTrainedTokenizer) -> int:
        """Add item-specific special tokens to tokenizer. Returns count of new tokens."""
        ...

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseItemTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)
