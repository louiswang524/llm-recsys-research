from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserHistory:
    user_id: str
    item_ids: list[str]
    ratings: list[float] | None
    timestamps: list[int] | None
    item_metadata: dict[str, dict]  # item_id -> {title, genres, description, ...}
    extra: dict[str, Any] | None = field(default=None)  # review texts, etc.


class RecDataset(ABC):
    def __init__(self, cfg, data_dir: str):
        self.cfg = cfg
        self.data_dir = data_dir
        self._users: dict[str, UserHistory] = {}
        self._item_meta: dict[str, dict] = {}

    @abstractmethod
    def load(self) -> None:
        """Load raw data into self._users and self._item_meta."""
        ...

    @property
    def users(self) -> dict[str, UserHistory]:
        return self._users

    @property
    def item_meta(self) -> dict[str, dict]:
        return self._item_meta

    @property
    def all_item_ids(self) -> list[str]:
        return list(self._item_meta.keys())
