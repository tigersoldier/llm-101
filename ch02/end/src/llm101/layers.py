"""Building blocks with a forward pass only (chapter 1)."""

from __future__ import annotations

import math

import torch


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
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = (
            torch.randn(in_features, out_features, generator=generator, dtype=dtype)
            * 0.02
        )
        self.bias = torch.zeros(out_features, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError("last dimension does not match in_features")
        return inputs @ self.weight + self.bias


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
            torch.randn(num_embeddings, embedding_dim, generator=generator, dtype=dtype)
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
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gain = torch.ones(normalized_shape, dtype=dtype)
        self.bias = torch.zeros(normalized_shape, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.normalized_shape:
            raise ValueError("last dimension does not match normalized_shape")
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.gain + self.bias


class GELU:
    """Smooth activation used inside the feed-forward block."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cubic = inputs.pow(3)
        inner = math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * cubic)
        return 0.5 * inputs * (1.0 + torch.tanh(inner))
