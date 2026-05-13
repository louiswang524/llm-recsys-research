from transformers import Trainer
from omegaconf import DictConfig
from ..models.losses import RecLoss


class RecTrainer(Trainer):
    """HF Trainer subclass that routes through the modular RecLoss."""

    def __init__(self, *args, loss_cfg: DictConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec_loss = RecLoss(loss_cfg)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Strip non-model keys before forward pass
        forward_inputs = {k: v for k, v in inputs.items() if k not in ("ratings", "mtp_labels")}
        outputs = model(**forward_inputs)
        losses = self.rec_loss(outputs, model, inputs)

        # Log sub-losses without polluting the main loss tracker
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({f"loss/{k}": v.item() for k, v in losses.items() if k != "total"})

        return (losses["total"], outputs) if return_outputs else losses["total"]
