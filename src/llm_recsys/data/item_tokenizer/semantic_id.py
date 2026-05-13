import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from .base import BaseItemTokenizer


class ResidualQuantizer(nn.Module):
    """Iterative K-means residual quantizer (RQ-VAE lite, no decoder needed)."""

    def __init__(self, n_levels: int, codebook_size: int, dim: int):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        # codebooks stored as plain tensors after fit (no gradient needed)
        self.register_buffer("codebooks", torch.zeros(n_levels, codebook_size, dim))

    def fit(self, embeddings: np.ndarray, n_iter: int = 100) -> None:
        from sklearn.cluster import MiniBatchKMeans
        residual = embeddings.copy().astype(np.float32)
        for level in range(self.n_levels):
            km = MiniBatchKMeans(
                n_clusters=self.codebook_size, n_init=3,
                max_iter=n_iter, random_state=42,
            )
            km.fit(residual)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            self.codebooks[level] = centers
            # subtract the assigned centroid from each point
            assignments = km.predict(residual)
            residual = residual - km.cluster_centers_[assignments]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, dim) -> codes: (N, n_levels)  [each in 0..codebook_size-1]"""
        codes = []
        residual = x.clone()
        for level in range(self.n_levels):
            cb = self.codebooks[level]  # (K, dim)
            dists = torch.cdist(residual, cb)
            idx = dists.argmin(dim=-1)  # (N,)
            codes.append(idx)
            residual = residual - cb[idx]
        return torch.stack(codes, dim=-1)  # (N, n_levels)


class SemanticIDTokenizer(BaseItemTokenizer):
    """PLUM/OneRec-style: RQ-VAE item codes → special tokens added to LLM vocab."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_levels: int = cfg.n_levels
        self.codebook_size: int = cfg.codebook_size
        self.item_embed_dim: int = cfg.item_embed_dim
        self.token_prefix: str = cfg.get("token_prefix", "item")
        self._item_codes: dict[str, list[int]] = {}
        self._rq: ResidualQuantizer | None = None

    def build(self, item_ids: list[str], item_meta: dict[str, dict]) -> None:
        embeddings = self._embed_items(item_ids, item_meta)
        self._rq = ResidualQuantizer(self.n_levels, self.codebook_size, self.item_embed_dim)
        self._rq.fit(embeddings)

        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        codes = self._rq.encode(emb_tensor).numpy()  # (N, n_levels)
        for item_id, code in zip(item_ids, codes):
            self._item_codes[item_id] = code.tolist()

    def _embed_items(self, item_ids: list[str], item_meta: dict[str, dict]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        encoder_name = getattr(self.cfg, "item_encoder", "sentence-transformers/all-MiniLM-L6-v2")
        encoder = SentenceTransformer(encoder_name)
        texts = [
            (item_meta.get(iid, {}).get("title", iid) + " " +
             item_meta.get(iid, {}).get("description", "")).strip()
            for iid in item_ids
        ]
        embs = encoder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype(np.float32)

        if embs.shape[1] != self.item_embed_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.item_embed_dim, random_state=42)
            embs = pca.fit_transform(embs).astype(np.float32)
        return embs

    def item_to_tokens(self, item_id: str) -> list[str]:
        if item_id not in self._item_codes:
            return [f"<{self.token_prefix}_unk>"]
        return [
            f"<{self.token_prefix}_L{lvl + 1}_{code}>"
            for lvl, code in enumerate(self._item_codes[item_id])
        ]

    def extend_tokenizer(self, tokenizer: PreTrainedTokenizer) -> int:
        new_tokens = [
            f"<{self.token_prefix}_L{lvl + 1}_{code}>"
            for lvl in range(self.n_levels)
            for code in range(self.codebook_size)
        ] + [f"<{self.token_prefix}_unk>"]
        return tokenizer.add_tokens(new_tokens)
