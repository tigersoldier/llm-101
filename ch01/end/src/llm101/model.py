"""Two-layer decoder-only transformer (chapter 1 scaffold)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm101.attention import MultiHeadSelfAttention
from llm101.layers import GELU, Embedding, LayerNorm, Linear


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int


class MLP:
    """Per-position feed-forward branch: Linear -> GELU -> Linear."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.up = Linear(d_model, d_ff, generator=generator, dtype=dtype)
        self.gelu = GELU()
        self.down = Linear(d_ff, d_model, generator=generator, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.gelu.forward(self.up.forward(inputs))
        return self.down.forward(hidden)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (
            self.up.weight,
            self.up.bias,
            self.down.weight,
            self.down.bias,
        )


class TransformerBlock:
    """Pre-LN block: attention then feed-forward, each with a residual."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.norm1 = LayerNorm(config.d_model, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            config.d_model,
            config.n_heads,
            generator=generator,
            dtype=dtype,
        )
        self.norm2 = LayerNorm(config.d_model, dtype=dtype)
        self.mlp = MLP(
            config.d_model,
            config.d_ff,
            generator=generator,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        normed1 = self.norm1.forward(inputs)
        attended, attention_cache = self.attention.forward(normed1)
        hidden1 = inputs + attended

        normed2 = self.norm2.forward(hidden1)
        mlp_out = self.mlp.forward(normed2)
        output = hidden1 + mlp_out

        return output, {
            "norm1": normed1,
            "attention": attention_cache,
            "hidden1": hidden1,
            "norm2": normed2,
            "mlp": mlp_out,
        }

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (
            self.norm1.gain,
            self.norm1.bias,
            self.attention.q_proj.weight,
            self.attention.q_proj.bias,
            self.attention.k_proj.weight,
            self.attention.k_proj.bias,
            self.attention.v_proj.weight,
            self.attention.v_proj.bias,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            self.norm2.gain,
            self.norm2.bias,
            *self.mlp.parameters(),
        )


class TinyGPT:
    """Embedding, N blocks, final norm and a tied output projection."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.token_embedding = Embedding(
            config.vocab_size,
            config.d_model,
            generator=generator,
            dtype=dtype,
        )
        self.position_embedding = Embedding(
            config.max_seq_len,
            config.d_model,
            generator=generator,
            dtype=dtype,
        )
        self.blocks: list[TransformerBlock] = [
            TransformerBlock(config, generator=generator, dtype=dtype)
            for _ in range(config.n_layers)
        ]
        self.final_norm = LayerNorm(config.d_model, dtype=dtype)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        parameters = [
            self.token_embedding.weight,
            self.position_embedding.weight,
        ]
        for block in self.blocks:
            parameters.extend(block.parameters())
        parameters.extend((self.final_norm.gain, self.final_norm.bias))
        return tuple(parameters)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits with shape ``[B, T, vocab]``."""
        if token_ids.shape[-1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {token_ids.shape[-1]} exceeds the model context "
                f"of {self.config.max_seq_len}"
            )
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        hidden = self.token_embedding.forward(
            token_ids
        ) + self.position_embedding.forward(positions)
        for block in self.blocks:
            hidden, _ = block.forward(hidden)
        hidden = self.final_norm.forward(hidden)
        return hidden @ self.token_embedding.weight.transpose(-1, -2)

    def forward_with_trace(self, token_ids: torch.Tensor) -> dict[str, object]:
        """Same result as ``forward`` plus every intermediate tensor."""
        raise NotImplementedError("TODO 1.8")
