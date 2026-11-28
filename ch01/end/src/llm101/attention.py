"""Causal multi-head self-attention (chapter 1 scaffold)."""

from __future__ import annotations

import math

import torch

from llm101.layers import Linear


def _split_heads(
    states: torch.Tensor,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Split the feature dimension into heads."""
    states = states.reshape(*states.shape[:-1], n_heads, head_dim)
    return states.transpose(-3, -2)


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
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, generator=generator, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, generator=generator, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, generator=generator, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, generator=generator, dtype=dtype)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the projected output plus Q/K/V and per-head weights."""
        if inputs.shape[-1] != self.d_model:
            raise ValueError("last dimension does not match d_model")

        q = self.q_proj.forward(inputs)
        k = self.k_proj.forward(inputs)
        v = self.v_proj.forward(inputs)

        q = _split_heads(q, self.n_heads, self.head_dim)
        k = _split_heads(k, self.n_heads, self.head_dim)
        v = _split_heads(v, self.n_heads, self.head_dim)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        length = inputs.shape[-2]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=inputs.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(-3, -2).reshape(*inputs.shape)
        output = self.out_proj.forward(attended)
        return output, {"q": q, "k": k, "v": v, "weights": weights}
