"""Two-layer decoder-only transformer (chapter 1 scaffold)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm101.attention import MultiHeadSelfAttention
from llm101.layers import Embedding, GELU, LayerNorm, Linear
from llm101.parameters import Parameter


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int


class TransformerBlock:
    """Pre-LN block: attention then feed-forward, each with a residual."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.7: build norm1/attention/norm2/mlp")

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError("TODO 1.7")


class TinyGPT:
    """Embedding, N blocks, final norm and a tied output projection."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.8")

    def parameters(self) -> tuple[Parameter, ...]:
        raise NotImplementedError("TODO 1.8")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO 1.8")
