import torch
import torch.nn as nn


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)."""

    def __init__(self, n_items: int, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, max_seq_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            _SASRecBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, item_seqs: torch.Tensor) -> torch.Tensor:
        B, L = item_seqs.shape
        pos = torch.arange(L, device=item_seqs.device).unsqueeze(0).expand(B, -1)
        x = self.drop(self.item_emb(item_seqs) + self.pos_emb(pos))
        pad_mask = item_seqs == 0
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x)  # (B, L, d_model)

    def predict(self, item_seqs: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        user_repr = self.forward(item_seqs)[:, -1]          # (B, d)
        cand_emb = self.item_emb(candidate_ids)              # (B, K, d) or (K, d)
        if cand_emb.dim() == 2:
            return user_repr @ cand_emb.T                    # (B, K)
        return (user_repr.unsqueeze(1) * cand_emb).sum(-1)  # (B, K)


class _SASRecBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad_mask)
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ff(x)))
