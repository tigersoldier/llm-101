"""Building blocks with a forward pass only (chapter 1)."""

from __future__ import annotations

import math

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
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        scale = math.sqrt(2.0 / (in_features + out_features))
        weight = (
            torch.randn(in_features, out_features, generator=generator, dtype=dtype)
            * scale
        )
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(out_features, dtype=dtype))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected final input dimension {self.in_features}, "
                f"got {inputs.shape[-1]}"
            )
        return inputs @ self.weight.value + self.bias.value

    def parameters(self) -> tuple[Parameter, Parameter]:
        return self.weight, self.bias


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
        self.weight = Parameter(initial)
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
        return self.weight.value[token_ids]

    def parameters(self) -> tuple[Parameter]:
        return (self.weight,)


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
