"""Building blocks with a forward pass only (chapter 1)."""

from __future__ import annotations

import torch


class Linear:
    """y = x W + b. Implemented in section 1.6."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.6: create weight and bias")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.6")


class Embedding:
    """Lookup table from token id to vector."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        initial = (
            torch.randn(
                num_embeddings, embedding_dim, generator=generator, dtype=dtype
            )
            * 0.02
        )
        self.weight = initial
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            raise TypeError("embedding indices must have dtype torch.long")
        if token_ids.numel() and (
            bool((token_ids < 0).any())
            or bool((token_ids >= self.num_embeddings).any())
        ):
            raise IndexError("embedding index is outside the vocabulary")
        return self.weight[token_ids]


class LayerNorm:
    """Normalise the last dimension, then scale and shift."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.5: create gain and bias")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.5")


class GELU:
    """Smooth activation used inside the feed-forward block."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.5")
