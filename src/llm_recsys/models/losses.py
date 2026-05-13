import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class RecLoss(nn.Module):
    """Modular loss registry. Enable/disable objectives via config weights."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.lm_weight = cfg.lm_weight
        self.engage_weight = cfg.engage_weight
        self.mtp_weight = cfg.mtp_weight
        self.mtp_k = cfg.get("mtp_k", 3)

    def forward(self, model_output, model, batch: dict) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        if self.lm_weight > 0 and model_output.loss is not None:
            losses["lm"] = model_output.loss * self.lm_weight

        if self.engage_weight > 0:
            ratings = batch.get("ratings")
            if ratings is not None and ratings.sum() > 0:
                pred = model.predict_engagement(batch["input_ids"], batch["attention_mask"])
                target = (ratings / 5.0).to(pred.device)
                losses["engage"] = F.mse_loss(pred, target) * self.engage_weight

        if self.mtp_weight > 0 and "mtp_labels" in batch:
            # Multi-token prediction: treat next-k item tokens as additional targets.
            logits = model_output.logits  # (B, L, V)
            mtp_labels = batch["mtp_labels"]  # (B, L)
            losses["mtp"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                mtp_labels[:, 1:].reshape(-1),
                ignore_index=-100,
            ) * self.mtp_weight

        losses["total"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses
