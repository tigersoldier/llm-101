"""Tests for chapter 1 embeddings."""

import pytest
import torch
import torch.nn.functional as F

from llm101.layers import GELU, Embedding, LayerNorm, Linear


def test_embedding_shapes_and_lookup() -> None:
    embedding = Embedding(10, 4, generator=torch.Generator().manual_seed(0))
    ids = torch.tensor([1, 2, 3])

    outputs = embedding.forward(ids)

    assert outputs.shape == (3, 4)
    torch.testing.assert_close(outputs[2], embedding.weight[3])


def test_embedding_weight_shape() -> None:
    embedding = Embedding(68, 16)

    assert embedding.weight.shape == (68, 16)


def test_embedding_rejects_float_indices() -> None:
    embedding = Embedding(10, 4)

    with pytest.raises(TypeError, match="torch.long"):
        embedding.forward(torch.tensor([1.0, 2.0]))


def test_embedding_rejects_out_of_range_ids() -> None:
    embedding = Embedding(10, 4)

    with pytest.raises(IndexError, match="vocabulary"):
        embedding.forward(torch.tensor([10]))


def test_layer_norm_normalizes_last_dim() -> None:
    layer_norm = LayerNorm(16)
    inputs = torch.randn(4, 16, generator=torch.Generator().manual_seed(1)) * 3 + 5

    outputs = layer_norm.forward(inputs)

    assert outputs.shape == inputs.shape
    torch.testing.assert_close(outputs.mean(dim=-1), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        outputs.var(dim=-1, unbiased=False), torch.ones(4), atol=1e-4, rtol=0
    )


def test_layer_norm_gain_and_bias() -> None:
    layer_norm = LayerNorm(16)
    layer_norm.gain = torch.full((16,), 2.0)
    layer_norm.bias = torch.full((16,), 1.0)
    inputs = torch.randn(3, 16, generator=torch.Generator().manual_seed(2))

    outputs = layer_norm.forward(inputs)

    torch.testing.assert_close(outputs.mean(dim=-1), torch.ones(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        outputs.var(dim=-1, unbiased=False), torch.full((3,), 4.0), atol=1e-4, rtol=0
    )


def test_layer_norm_eps_handles_constant_input() -> None:
    layer_norm = LayerNorm(4)

    outputs = layer_norm.forward(torch.ones(2, 4))

    torch.testing.assert_close(outputs, torch.zeros(2, 4))


def test_layer_norm_rejects_wrong_last_dim() -> None:
    layer_norm = LayerNorm(4)

    with pytest.raises(ValueError, match="normalized_shape"):
        layer_norm.forward(torch.zeros(2, 3))


def test_gelu_endpoints_and_midpoint() -> None:
    gelu = GELU()
    values = gelu.forward(torch.tensor([-10.0, 0.0, 1.0, 10.0]))

    assert values.shape == (4,)
    assert values[0].item() < 1e-4
    assert values[1].item() == 0.0
    assert abs(values[2].item() - 0.8413447) < 1e-3
    assert abs(values[3].item() - 10.0) < 1e-3


def test_gelu_matches_torch_reference() -> None:
    gelu = GELU()
    inputs = torch.linspace(-4, 4, 33)

    torch.testing.assert_close(
        gelu.forward(inputs), F.gelu(inputs), atol=2e-3, rtol=2e-3
    )


def test_linear_shapes_and_values() -> None:
    linear = Linear(3, 2, generator=torch.Generator().manual_seed(0))
    inputs = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))

    outputs = linear.forward(inputs)

    assert outputs.shape == (4, 2)
    torch.testing.assert_close(outputs, inputs @ linear.weight + linear.bias)


def test_linear_bias_and_last_dim_check() -> None:
    linear = Linear(3, 2)
    linear.bias = torch.tensor([1.0, -1.0])

    outputs = linear.forward(torch.zeros(1, 3))

    torch.testing.assert_close(outputs, torch.tensor([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="in_features"):
        linear.forward(torch.zeros(1, 4))
