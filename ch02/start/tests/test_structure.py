"""Tests for the chapter 1 forward pass; chapter 2 keeps them green."""

import pytest
import torch

from llm101.attention import MultiHeadSelfAttention
from llm101.model import GPTConfig, TinyGPT, TransformerBlock


def small_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=68,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
    )


def test_forward_shapes() -> None:
    """[B, T] ids in, [B, T, vocab] logits out."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model.forward(token_ids)

    assert logits.shape == (2, 3, 68)
    assert not logits.requires_grad


def test_attention_shapes_and_row_sums() -> None:
    attention = MultiHeadSelfAttention(
        16, 2, generator=torch.Generator().manual_seed(0)
    )
    inputs = torch.randn(5, 16, generator=torch.Generator().manual_seed(1))

    outputs, cache = attention.forward(inputs)

    assert outputs.shape == (5, 16)
    assert cache["q"].shape == (2, 5, 8)
    assert cache["weights"].shape == (2, 5, 5)
    torch.testing.assert_close(
        cache["weights"].sum(dim=-1), torch.ones(2, 5), atol=1e-5, rtol=0
    )


def test_attention_is_causal() -> None:
    attention = MultiHeadSelfAttention(
        16, 2, generator=torch.Generator().manual_seed(0)
    )
    inputs = torch.randn(5, 16, generator=torch.Generator().manual_seed(1))

    _, cache = attention.forward(inputs)
    upper = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    assert torch.all(cache["weights"][:, upper] == 0)


def test_causal_mask_hides_future() -> None:
    """TODO 1.6: changing a later token must not change earlier outputs."""
    attention = MultiHeadSelfAttention(
        16, 2, generator=torch.Generator().manual_seed(0)
    )
    inputs = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(1))
    later = inputs.clone()
    later[:, -1, :] += 1.0

    first, _ = attention.forward(inputs)
    second, _ = attention.forward(later)

    torch.testing.assert_close(first[:, :-1], second[:, :-1])


def test_context_limit() -> None:
    """1.4: sequences longer than max_seq_len are rejected."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))
    token_ids = torch.zeros(1, 17, dtype=torch.long)

    with pytest.raises(ValueError, match="context"):
        model.forward(token_ids)


def test_batch_dimension() -> None:
    """1.8: [B, T] ids produce [B, T, vocab] logits."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))
    batch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    outputs = model.forward(batch)

    assert outputs.shape == (2, 3, 68)


def test_position_changes_the_embedding() -> None:
    """1.4: the same token at different positions gets different vectors."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))
    token = model.token_embedding.forward(torch.tensor([1], dtype=torch.long))[0]
    first = token + model.position_embedding.forward(torch.tensor([0]))[0]
    second = token + model.position_embedding.forward(torch.tensor([1]))[0]

    assert not torch.allclose(first, second)


def test_block_shapes() -> None:
    """1.7: one block keeps [B, T, C] and exposes attention intermediates."""
    block = TransformerBlock(small_config(), generator=torch.Generator().manual_seed(0))
    inputs = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(1))

    outputs, cache = block.forward(inputs)

    assert outputs.shape == (2, 3, 16)
    assert "attention" in cache


def test_block_parameter_count() -> None:
    """1.7: one block has 2,224 parameters."""
    block = TransformerBlock(small_config(), generator=torch.Generator().manual_seed(0))

    assert sum(parameter.numel() for parameter in block.parameters()) == 2224


def test_parameter_count() -> None:
    """1.8: the two-layer model has exactly 5,824 parameters."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))

    assert len(model.parameters()) > 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 5824


def test_last_position_logits_and_argmax() -> None:
    """1.8: the last position has one score per vocabulary entry."""
    model = TinyGPT(small_config(), generator=torch.Generator().manual_seed(0))

    logits = model.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))
    last = logits[0, -1]

    assert logits.shape == (1, 3, 68)
    assert last.shape == (68,)
    assert 0 <= int(last.argmax()) < 68
