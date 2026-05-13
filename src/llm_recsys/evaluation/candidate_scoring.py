import torch
from transformers import PreTrainedTokenizer
from ..data.item_tokenizer.base import BaseItemTokenizer


class CandidateScorer:
    """Rank candidate items by log P(item tokens | context) under the LLM."""

    def __init__(self, tokenizer: PreTrainedTokenizer, item_tokenizer: BaseItemTokenizer,
                 device: str = "cuda"):
        self.tokenizer = tokenizer
        self.item_tokenizer = item_tokenizer
        self.device = device

    @torch.no_grad()
    def score_candidates(self, model, context_text: str, candidate_ids: list[str]) -> dict[str, float]:
        context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        scores = {}

        for item_id in candidate_ids:
            item_text = " ".join(self.item_tokenizer.item_to_tokens(item_id))
            item_ids = self.tokenizer.encode(item_text, add_special_tokens=False)
            full_ids = torch.tensor([context_ids + item_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(full_ids)

            outputs = model.base_model(input_ids=full_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # (L, V)

            # Log-prob of item tokens given context
            log_probs = torch.log_softmax(logits, dim=-1)
            item_start = len(context_ids)
            item_log_prob = sum(
                log_probs[item_start + i - 1, item_ids[i]].item()
                for i in range(len(item_ids))
            )
            scores[item_id] = item_log_prob

        return scores

    def rank_candidates(self, model, context_text: str, candidate_ids: list[str]) -> list[str]:
        scores = self.score_candidates(model, context_text, candidate_ids)
        return sorted(candidate_ids, key=lambda x: scores[x], reverse=True)
