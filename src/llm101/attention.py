"""Causal multi-head self-attention with an explicit backward pass."""

from __future__ import annotations

import math

import torch

from llm101.layers import Linear
from llm101.parameters import Parameter


class MultiHeadSelfAttention:
    """Dense causal self-attention for inputs shaped ``[batch, time, width]``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("model width and head count must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = Linear(
            d_model, d_model, dtype=dtype, generator=generator
        )
        self.key_projection = Linear(d_model, d_model, dtype=dtype, generator=generator)
        self.value_projection = Linear(
            d_model, d_model, dtype=dtype, generator=generator
        )
        self.output_projection = Linear(
            d_model, d_model, dtype=dtype, generator=generator
        )

        self._query: torch.Tensor | None = None
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None
        self._attention_weights: torch.Tensor | None = None
        self._batch_size: int | None = None
        self._sequence_length: int | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("attention inputs must have shape [batch, time, width]")
        batch_size, sequence_length, width = inputs.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if width != self.d_model:
            raise ValueError(f"expected model width {self.d_model}, got {width}")

        query = self._split_heads(self.query_projection.forward(inputs))
        key = self._split_heads(self.key_projection.forward(inputs))
        value = self._split_heads(self.value_projection.forward(inputs))

        scores = (query @ key.transpose(-2, -1)) * self.scale
        future_positions = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=inputs.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future_positions, float("-inf"))
        shifted_scores = scores - scores.max(dim=-1, keepdim=True).values
        exponentials = shifted_scores.exp()
        attention_weights = exponentials / exponentials.sum(dim=-1, keepdim=True)

        context = attention_weights @ value
        merged_context = self._merge_heads(context)
        self._query = query
        self._key = key
        self._value = value
        self._attention_weights = attention_weights
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        return self.output_projection.forward(merged_context)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Accumulate projection gradients and return the input gradient."""
        if (
            self._query is None
            or self._key is None
            or self._value is None
            or self._attention_weights is None
            or self._batch_size is None
            or self._sequence_length is None
        ):
            raise RuntimeError("attention.backward() requires a preceding forward()")

        expected_shape = (self._batch_size, self._sequence_length, self.d_model)
        if tuple(grad_output.shape) != expected_shape:
            raise ValueError(
                f"expected grad_output shape {expected_shape}, got {tuple(grad_output.shape)}"
            )

        query = self._query
        key = self._key
        value = self._value
        attention_weights = self._attention_weights

        grad_merged_context = self.output_projection.backward(grad_output)
        grad_context = self._split_heads(grad_merged_context)

        grad_attention_weights = grad_context @ value.transpose(-2, -1)
        grad_value = attention_weights.transpose(-2, -1) @ grad_context
        softmax_projection = (
            grad_attention_weights * attention_weights
        ).sum(dim=-1, keepdim=True)
        grad_scores = attention_weights * (grad_attention_weights - softmax_projection)

        grad_query = (grad_scores @ key) * self.scale
        grad_key = (grad_scores.transpose(-2, -1) @ query) * self.scale

        grad_query = self._merge_heads(grad_query)
        grad_key = self._merge_heads(grad_key)
        grad_value = self._merge_heads(grad_value)
        grad_input = (
            self.query_projection.backward(grad_query)
            + self.key_projection.backward(grad_key)
            + self.value_projection.backward(grad_value)
        )

        self._query = None
        self._key = None
        self._value = None
        self._attention_weights = None
        self._batch_size = None
        self._sequence_length = None
        return grad_input

    def parameters(self) -> tuple[Parameter, ...]:
        return (
            *self.query_projection.parameters(),
            *self.key_projection.parameters(),
            *self.value_projection.parameters(),
            *self.output_projection.parameters(),
        )

    def _split_heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        return states.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = states.shape
        return (
            states.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, self.d_model)
        )
