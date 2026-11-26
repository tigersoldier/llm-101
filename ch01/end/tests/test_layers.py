"""Tests for chapter 1 parameters and the first two layers."""

import pytest
import torch

from llm101.layers import Embedding, Linear
from llm101.parameters import Parameter


def test_parameter_starts_with_zero_grad() -> None:
    parameter = Parameter(torch.ones(2, 3))

    assert parameter.value.shape == (2, 3)
    assert parameter.grad.shape == (2, 3)
    assert not parameter.grad.any()
    assert not parameter.value.requires_grad


def test_parameter_owns_its_value() -> None:
    source = torch.zeros(2)
    parameter = Parameter(source)
    source.fill_(1.0)

    assert not parameter.value.any()


def test_parameter_rejects_autograd_tensors() -> None:
    with pytest.raises(ValueError, match="autograd"):
        Parameter(torch.zeros(2, requires_grad=True))


def test_zero_grad_clears_the_buffer() -> None:
    parameter = Parameter(torch.ones(2))
    parameter.grad.fill_(3.0)

    parameter.zero_grad()

    assert not parameter.grad.any()


def test_linear_shapes_and_values() -> None:
    linear = Linear(4, 3, generator=torch.Generator().manual_seed(0))
    inputs = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(1))

    outputs = linear.forward(inputs)

    assert outputs.shape == (2, 5, 3)
    torch.testing.assert_close(
        outputs, inputs @ linear.weight.value + linear.bias.value
    )


def test_linear_parameter_count() -> None:
    linear = Linear(4, 3)

    assert sum(p.value.numel() for p in linear.parameters()) == 4 * 3 + 3


def test_linear_rejects_wrong_input_width() -> None:
    linear = Linear(4, 3)

    with pytest.raises(ValueError, match="input dimension"):
        linear.forward(torch.randn(2, 5))


def test_embedding_shapes_and_lookup() -> None:
    embedding = Embedding(10, 4, generator=torch.Generator().manual_seed(0))
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    outputs = embedding.forward(ids)

    assert outputs.shape == (2, 3, 4)
    torch.testing.assert_close(outputs[1, 2], embedding.weight.value[6])


def test_embedding_parameter_count() -> None:
    embedding = Embedding(68, 16)

    assert sum(p.value.numel() for p in embedding.parameters()) == 68 * 16


def test_embedding_rejects_float_indices() -> None:
    embedding = Embedding(10, 4)

    with pytest.raises(TypeError, match="torch.long"):
        embedding.forward(torch.tensor([1.0, 2.0]))


def test_embedding_rejects_out_of_range_ids() -> None:
    embedding = Embedding(10, 4)

    with pytest.raises(IndexError, match="vocabulary"):
        embedding.forward(torch.tensor([10]))
