"""Tests for chapter 1 embeddings."""

import pytest
import torch

from llm101.layers import Embedding


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
