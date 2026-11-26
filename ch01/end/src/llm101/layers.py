"""Building blocks with a forward pass only (chapter 1 scaffold)."""

from __future__ import annotations

import torch

from llm101.parameters import Parameter


class Linear:
    """y = x W + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.3: create weight and bias parameters")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.3")


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
        raise NotImplementedError("TODO 1.3: create the embedding table")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.3")


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
