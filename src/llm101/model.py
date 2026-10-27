"""A small pre-LN decoder-only Transformer trained without autograd."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm101.attention import MultiHeadSelfAttention
from llm101.layers import GELU, Embedding, LayerNorm, Linear
from llm101.parameters import Parameter


@dataclass(frozen=True)
class GPTConfig:
    """Shape settings for the educational decoder-only model."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model width, layer count, and head count must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0 or self.max_seq_len <= 0:
            raise ValueError("feed-forward width and maximum sequence length must be positive")


class TransformerBlock:
    """Pre-LN causal attention and GELU MLP, each with a residual connection."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        self.attention_norm = LayerNorm(config.d_model, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            config.d_model,
            config.n_heads,
            dtype=dtype,
            generator=generator,
        )
        self.mlp_norm = LayerNorm(config.d_model, dtype=dtype)
        self.mlp_up = Linear(
            config.d_model, config.d_ff, dtype=dtype, generator=generator
        )
        self.activation = GELU()
        self.mlp_down = Linear(
            config.d_ff, config.d_model, dtype=dtype, generator=generator
        )
        self._has_forwarded = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm.forward(inputs)
        attended = self.attention.forward(normalized)
        after_attention = inputs + attended

        normalized_mlp = self.mlp_norm.forward(after_attention)
        mlp_hidden = self.mlp_up.forward(normalized_mlp)
        activated = self.activation.forward(mlp_hidden)
        mlp_output = self.mlp_down.forward(activated)
        self._has_forwarded = True
        return after_attention + mlp_output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if not self._has_forwarded:
            raise RuntimeError("TransformerBlock.backward() requires a preceding forward()")

        grad_mlp_branch = self.mlp_down.backward(grad_output)
        grad_mlp_branch = self.activation.backward(grad_mlp_branch)
        grad_mlp_branch = self.mlp_up.backward(grad_mlp_branch)
        grad_after_attention = grad_output + self.mlp_norm.backward(grad_mlp_branch)

        grad_attention_branch = self.attention.backward(grad_after_attention)
        grad_attention_branch = self.attention_norm.backward(grad_attention_branch)
        self._has_forwarded = False
        return grad_after_attention + grad_attention_branch

    def parameters(self) -> tuple[Parameter, ...]:
        return (
            *self.attention_norm.parameters(),
            *self.attention.parameters(),
            *self.mlp_norm.parameters(),
            *self.mlp_up.parameters(),
            *self.mlp_down.parameters(),
        )


class TinyGPT:
    """Pre-LN causal Transformer with learned positions and tied token weights.

    The token embedding matrix is also the vocabulary projection matrix:
    ``logits = hidden_states @ token_embedding.weight.T``. Its gradient receives
    contributions from both input lookups and the output projection.
    """

    def __init__(
        self,
        config: GPTConfig,
        *,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.token_embedding = Embedding(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            generator=generator,
        )
        self.position_embedding = Embedding(
            config.max_seq_len,
            config.d_model,
            dtype=dtype,
            generator=generator,
        )
        self.blocks = [
            TransformerBlock(config, dtype=dtype, generator=generator)
            for _ in range(config.n_layers)
        ]
        self.final_norm = LayerNorm(config.d_model, dtype=dtype)
        self._final_hidden_states: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        batch_size, sequence_length = input_ids.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.config.max_seq_len}"
            )
        if input_ids.device != self.token_embedding.weight.value.device:
            raise ValueError("input IDs and model parameters must be on the same device")

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        hidden_states = self.token_embedding.forward(input_ids)
        hidden_states = hidden_states + self.position_embedding.forward(position_ids)
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
        hidden_states = self.final_norm.forward(hidden_states)
        self._final_hidden_states = hidden_states
        return hidden_states @ self.token_embedding.weight.value.T

    def backward(self, grad_logits: torch.Tensor) -> None:
        """Backpropagate logits gradients and accumulate every parameter gradient."""
        if self._final_hidden_states is None:
            raise RuntimeError("TinyGPT.backward() requires a preceding forward()")
        expected_shape = (
            self._final_hidden_states.shape[0],
            self._final_hidden_states.shape[1],
            self.config.vocab_size,
        )
        if tuple(grad_logits.shape) != expected_shape:
            raise ValueError(
                f"expected grad_logits shape {expected_shape}, got {tuple(grad_logits.shape)}"
            )

        flat_hidden = self._final_hidden_states.reshape(-1, self.config.d_model)
        flat_grad_logits = grad_logits.reshape(-1, self.config.vocab_size)
        self.token_embedding.weight.grad.add_(flat_grad_logits.T @ flat_hidden)
        grad_hidden = grad_logits @ self.token_embedding.weight.value
        grad_hidden = self.final_norm.backward(grad_hidden)
        for block in reversed(self.blocks):
            grad_hidden = block.backward(grad_hidden)
        self.position_embedding.backward(grad_hidden)
        self.token_embedding.backward(grad_hidden)
        self._final_hidden_states = None

    def parameters(self) -> tuple[Parameter, ...]:
        """Return each unique model parameter once, including the tied matrix once."""
        return (
            *self.token_embedding.parameters(),
            *self.position_embedding.parameters(),
            *(parameter for block in self.blocks for parameter in block.parameters()),
            *self.final_norm.parameters(),
        )

    def parameter_count(self) -> int:
        return sum(parameter.value.numel() for parameter in self.parameters())
