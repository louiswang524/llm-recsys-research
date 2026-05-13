import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """BERT for Sequential Recommendation (Sun et al., 2019).
    Vocab: 0=pad, 1=mask, 2..n_items+1=items (shifted by 2).
    """

    MASK_TOKEN = 1

    def __init__(self, n_items: int, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, max_seq_len: int = 50, dropout: float = 0.2,
                 mask_prob: float = 0.15):
        super().__init__()
        self.n_items = n_items
        self.mask_prob = mask_prob
        vocab_size = n_items + 2  # pad + mask + items
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, item_seqs: torch.Tensor) -> torch.Tensor:
        B, L = item_seqs.shape
        pos = torch.arange(L, device=item_seqs.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(item_seqs) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=(item_seqs == 0))
        return self.head(x)  # (B, L, vocab_size)

    def mask_for_training(self, item_seqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = item_seqs.clone()
        labels = torch.full_like(item_seqs, -100)
        mask = (torch.rand_like(item_seqs.float()) < self.mask_prob) & (item_seqs != 0)
        labels[mask] = item_seqs[mask]
        masked[mask] = self.MASK_TOKEN
        return masked, labels

    def predict_next(self, item_seqs: torch.Tensor) -> torch.Tensor:
        """Mask last item and return logits over items for that position."""
        masked = item_seqs.clone()
        masked[:, -1] = self.MASK_TOKEN
        return self.forward(masked)[:, -1]  # (B, vocab_size)
