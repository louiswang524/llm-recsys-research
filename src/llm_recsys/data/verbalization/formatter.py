from typing import Optional
from transformers import PreTrainedTokenizer
from .base import BaseVerbalizer
from ..datasets.base import UserHistory


class InstructionFormatter:
    """Wraps a Verbalizer with the model's chat template (SFT) or plain text (CPT).

    Verbalization modes (see llm_native_rec_guide.md §1.3):
      Mode 1 — text target (item_tokenizer=None): response is the item title string.
      Mode 2 — SID target (item_tokenizer provided): response is "<item_L1_x>..." tokens.
      Mode 3 — CPT cross-modal: context already includes SIDs via the verbalizer;
                target is SID tokens (same item_tokenizer needed).
    """

    def __init__(
        self,
        verbalizer: BaseVerbalizer,
        tokenizer: PreTrainedTokenizer,
        stage: str,
        item_tokenizer=None,  # SemanticIDTokenizer or None → Mode 1
    ):
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.stage = stage  # "cpt" | "sft"
        self.item_tokenizer = item_tokenizer

    def _target_text(self, history: UserHistory, target_item_id: str) -> str:
        if self.item_tokenizer is not None:
            tokens = self.item_tokenizer.item_to_tokens(target_item_id)
            return " ".join(tokens)
        return history.item_metadata.get(target_item_id, {}).get("title", target_item_id)

    def format(self, history: UserHistory, target_item_id: str) -> dict:
        """Returns dict with full_text and input_len (# tokens to mask from loss in SFT)."""
        if self.stage == "cpt":
            full_text = self.verbalizer.verbalize(history, target_item_id)
            return {"full_text": full_text, "input_len": 0}

        target_text = self._target_text(history, target_item_id)

        # SFT: apply chat template so the model learns the right response format
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            user_content = self.verbalizer.verbalize(history, target_item_id=None)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_text},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_only = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            full_text = self.verbalizer.verbalize(history, target_item_id) + "\nNext: " + target_text
            prompt_only = self.verbalizer.verbalize(history, target_item_id=None)

        input_len = len(self.tokenizer.encode(prompt_only, add_special_tokens=False))
        return {"full_text": full_text, "input_len": input_len}
