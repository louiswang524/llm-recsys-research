from transformers import PreTrainedTokenizer
from .base import BaseVerbalizer
from ..datasets.base import UserHistory


class InstructionFormatter:
    """Wraps a Verbalizer with the model's chat template (SFT) or plain text (CPT)."""

    def __init__(self, verbalizer: BaseVerbalizer, tokenizer: PreTrainedTokenizer, stage: str):
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.stage = stage  # "cpt" | "sft"

    def format(self, history: UserHistory, target_item_id: str) -> dict:
        """Returns dict with full_text and input_len (# tokens to mask from loss in SFT)."""
        target_title = history.item_metadata.get(target_item_id, {}).get("title", target_item_id)

        if self.stage == "cpt":
            full_text = self.verbalizer.verbalize(history, target_item_id)
            return {"full_text": full_text, "input_len": 0}

        # SFT: apply chat template so the model learns the right response format
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            user_content = self.verbalizer.verbalize(history, target_item_id=None)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_title},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_only = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            full_text = self.verbalizer.verbalize(history, target_item_id)
            prompt_only = self.verbalizer.verbalize(history, target_item_id=None)

        input_len = len(self.tokenizer.encode(prompt_only, add_special_tokens=False))
        return {"full_text": full_text, "input_len": input_len}
