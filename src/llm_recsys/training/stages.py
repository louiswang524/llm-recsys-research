import os
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments
from torch.utils.data import Dataset
from .trainer import RecTrainer


def _distributed_type() -> str:
    """Detect accelerate distributed backend from environment."""
    # accelerate sets this when launched via `accelerate launch`
    return os.environ.get("ACCELERATE_DISTRIBUTED_TYPE", "NO").upper()


def build_training_args(cfg: DictConfig, output_dir: str) -> TrainingArguments:
    t = cfg.training
    dist = _distributed_type()

    # FSDP requires explicit opt-in in TrainingArguments
    fsdp = "full_shard auto_wrap" if dist == "FSDP" else ""
    fsdp_transformer_layer_cls = (
        # Qwen2 / Llama transformer block class names for auto-wrap policy
        "Qwen2DecoderLayer,LlamaDecoderLayer,MistralDecoderLayer"
        if dist == "FSDP" else ""
    )

    # bf16 is handled differently on TPU (always on) vs GPU
    use_bf16 = t.bf16 and dist != "TPU"

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=getattr(t, "eval_batch_size", t.per_device_batch_size),
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio,
        max_grad_norm=t.max_grad_norm,
        bf16=use_bf16,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=t.dataloader_num_workers,
        report_to=["wandb"] if cfg.get("use_wandb") else ["none"],
        remove_unused_columns=False,
        fsdp=fsdp,
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls or None,
        ddp_find_unused_parameters=False,  # LoRA leaves some params unused in DDP
    )


def run_stage(
    cfg: DictConfig,
    model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
    data_collator=None,
    callbacks: list | None = None,
) -> RecTrainer:
    training_args = build_training_args(cfg, output_dir)
    resume_from = cfg.training.get("resume_from_checkpoint")

    trainer = RecTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks or [],
        loss_cfg=cfg.loss,
    )
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = os.path.join(output_dir, "final")
    model.save(final_dir)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
    return trainer
