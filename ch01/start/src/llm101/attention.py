"""Causal multi-head self-attention (chapter 1 scaffold)."""

from __future__ import annotations

import torch

from llm101.layers import Linear


class MultiHeadSelfAttention:
    """Scaled dot-product attention with a causal mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("TODO 1.6: create the four projections")

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the projected output plus Q/K/V and per-head weights."""
        raise NotImplementedError("TODO 1.6")
