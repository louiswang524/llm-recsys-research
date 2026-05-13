from abc import ABC, abstractmethod
from ..datasets.base import UserHistory


class BaseVerbalizer(ABC):
    @abstractmethod
    def verbalize(self, history: UserHistory, target_item_id: str | None = None) -> str:
        """Convert user history into text. Appends target item title if provided."""
        ...
