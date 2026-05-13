import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from omegaconf import DictConfig


class LLMRecModel(nn.Module):
    def __init__(self, cfg: DictConfig, vocab_size_delta: int = 0):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        bnb_config = None
        if model_cfg.get("load_in_4bit"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        attn_impl = "flash_attention_2" if model_cfg.get("use_flash_attention") else "eager"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.base_model,
            torch_dtype=torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float32,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map="auto",
        )

        if vocab_size_delta > 0:
            self.base_model.resize_token_embeddings(
                self.base_model.config.vocab_size + vocab_size_delta
            )

        if model_cfg.get("use_gradient_checkpointing"):
            self.base_model.gradient_checkpointing_enable()

        if model_cfg.lora.enabled:
            lora_cfg = model_cfg.lora
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                target_modules=list(lora_cfg.target_modules),
                lora_dropout=lora_cfg.lora_dropout,
                bias=lora_cfg.bias,
            )
            self.base_model = get_peft_model(self.base_model, peft_cfg)
            self.base_model.print_trainable_parameters()

        hidden_size = self.base_model.config.hidden_size
        self.engage_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

    def get_last_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        seq_lens = attention_mask.sum(dim=1) - 1
        hidden = out.hidden_states[-1]
        return hidden[torch.arange(hidden.size(0)), seq_lens]

    def predict_engagement(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # no_grad only for inference — during training, RecLoss calls this inside the training graph
        return self.engage_head(self.get_last_hidden(input_ids, attention_mask)).squeeze(-1)

    def save(self, path: str) -> None:
        self.base_model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, cfg: DictConfig) -> "LLMRecModel":
        model = cls(cfg)
        model.base_model = PeftModel.from_pretrained(model.base_model, checkpoint_path)
        return model
