"""Tests for the section-by-section placeholder pipeline."""

import pytest
import torch

from llm101.layers import Embedding
from llm101.placeholder import VOCAB_SIZE, WIDTH, argmax, run
from llm101.tokenizer import DEFAULT_VOCABULARY, CharacterTokenizer

EXAMPLE = "the cat sat"


def test_stage_one_uses_placeholder_ids() -> None:
    result = run(EXAMPLE, stage=1)

    assert result.ids == [ord(character) % VOCAB_SIZE for character in EXAMPLE]
    assert len(result.vectors) == len(EXAMPLE)
    assert len(result.vectors[0]) == WIDTH
    assert len(result.scores) == VOCAB_SIZE
    assert 0 <= result.answer < VOCAB_SIZE


def test_stage_one_is_deterministic() -> None:
    first = run(EXAMPLE, stage=1)
    second = run(EXAMPLE, stage=1)

    assert first.ids == second.ids
    assert first.answer == second.answer
    assert first.scores == second.scores


def test_stage_two_uses_the_tokenizer() -> None:
    tokenizer = CharacterTokenizer(DEFAULT_VOCABULARY)
    result = run(EXAMPLE, stage=2)

    assert result.ids == tokenizer.encode(EXAMPLE, add_bos=True, add_eos=True)
    assert len(result.ids) == len(EXAMPLE) + 2
    assert len(result.vectors) == len(result.ids)


def test_stage_three_uses_the_embedding_table() -> None:
    result = run(EXAMPLE, stage=3)
    embedding = Embedding(VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0))
    expected = embedding.forward(torch.tensor([result.ids], dtype=torch.long))

    torch.testing.assert_close(torch.tensor(result.vectors), expected[0])
    assert len(result.scores) == VOCAB_SIZE


def test_argmax_returns_the_first_largest_index() -> None:
    assert argmax([1.0, 3.0, 3.0, 2.0]) == 1


def test_future_stage_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="1.4"):
        run(EXAMPLE, stage=4)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run("", stage=1)
