from dataclasses import dataclass
from typing import Any
import torch
from transformers import PreTrainedTokenizer


@dataclass
class RecDataCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list, attn_mask_list, labels_list, ratings_list = [], [], [], []

        for feat in features:
            enc = self.tokenizer(
                feat["full_text"],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]
            labels = input_ids.clone()

            # In SFT mode, mask the context tokens so loss is only on the response.
            input_len = feat.get("input_len", 0)
            if input_len > 0:
                labels[:input_len] = self.label_pad_id

            input_ids_list.append(input_ids)
            attn_mask_list.append(enc["attention_mask"][0])
            labels_list.append(labels)
            ratings_list.append(feat.get("rating", 0.0))

        return {
            "input_ids": self._pad(input_ids_list, self.tokenizer.pad_token_id or 0),
            "attention_mask": self._pad(attn_mask_list, 0),
            "labels": self._pad(labels_list, self.label_pad_id),
            "ratings": torch.tensor(ratings_list, dtype=torch.float32),
        }

    def _pad(self, tensors: list[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), max_len), pad_val, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out
