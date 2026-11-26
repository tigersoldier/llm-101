"""Tests for the section-by-section placeholder pipeline."""

import pytest
import torch

from llm101.attention import MultiHeadSelfAttention
from llm101.layers import Embedding, GELU, LayerNorm
from llm101.placeholder import ALPHABET, MAX_SEQ_LEN, VOCAB_SIZE, WIDTH, run
from llm101.tokenizer import DEFAULT_VOCABULARY, CharacterTokenizer

EXAMPLE = "the cat sat"


def test_stage_one_uses_placeholder_ids_and_character() -> None:
    result = run(EXAMPLE, stage=1)

    assert result.ids == [ord(character) % VOCAB_SIZE for character in EXAMPLE]
    assert result.vectors is None
    assert result.output is None
    assert result.scores is None
    assert result.answer is not None
    assert result.character == ALPHABET[result.answer % len(ALPHABET)]


def test_stage_one_is_deterministic() -> None:
    first = run(EXAMPLE, stage=1)
    second = run(EXAMPLE, stage=1)

    assert first.ids == second.ids
    assert first.answer == second.answer
    assert first.character == second.character


def test_stage_two_decodes_the_answer() -> None:
    tokenizer = CharacterTokenizer(DEFAULT_VOCABULARY)
    result = run(EXAMPLE, stage=2)

    assert result.ids == tokenizer.encode(EXAMPLE, add_bos=True, add_eos=True)
    assert len(result.ids) == len(EXAMPLE) + 2
    assert result.vectors is None
    assert result.output is None
    assert result.scores is None
    assert result.answer is not None
    assert result.character == tokenizer.decode([result.answer])


def test_stage_three_returns_the_embedding_vectors() -> None:
    result = run(EXAMPLE, stage=3)
    embedding = Embedding(VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0))
    expected = embedding.forward(torch.tensor(result.ids, dtype=torch.long))

    assert result.answer is not None
    assert result.character == ALPHABET[result.answer % len(ALPHABET)]
    assert result.vectors is not None
    assert result.position_vectors is None
    torch.testing.assert_close(result.vectors, expected)
    assert result.output is not None
    assert result.output.shape == (WIDTH,)
    torch.testing.assert_close(result.output, expected.sum(dim=0))
    assert result.scores is not None
    assert result.scores.shape == (VOCAB_SIZE,)
    assert result.answer == int(result.scores.max(dim=0).indices)


def test_stage_four_adds_position_vectors() -> None:
    result = run(EXAMPLE, stage=4)
    token_embedding = Embedding(
        VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
    )
    position_embedding = Embedding(
        MAX_SEQ_LEN, WIDTH, generator=torch.Generator().manual_seed(1)
    )
    token_ids = torch.tensor(result.ids, dtype=torch.long)
    positions = torch.arange(len(result.ids))
    expected_tokens = token_embedding.forward(token_ids)
    expected_positions = position_embedding.forward(positions)

    assert result.position_vectors is not None
    assert result.position_vectors.shape == (len(result.ids), WIDTH)
    assert result.vectors is not None
    torch.testing.assert_close(result.vectors, expected_tokens + expected_positions)
    assert result.scores is not None
    assert result.answer == int(result.scores.max(dim=0).indices)


def test_stage_four_rejects_text_longer_than_context() -> None:
    with pytest.raises(ValueError, match="context"):
        run("x" * 17, stage=4)


def test_stage_five_normalizes_and_activates() -> None:
    result = run(EXAMPLE, stage=5)
    token_embedding = Embedding(
        VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
    )
    position_embedding = Embedding(
        MAX_SEQ_LEN, WIDTH, generator=torch.Generator().manual_seed(1)
    )
    token_ids = torch.tensor(result.ids, dtype=torch.long)
    positions = torch.arange(len(result.ids))
    base = token_embedding.forward(token_ids) + position_embedding.forward(positions)
    expected = GELU().forward(LayerNorm(WIDTH).forward(base))

    assert result.vectors is not None
    torch.testing.assert_close(result.vectors, expected)
    assert result.scores is not None
    assert result.answer == int(result.scores.max(dim=0).indices)


def test_stage_six_adds_attention() -> None:
    result = run(EXAMPLE, stage=6)
    token_embedding = Embedding(
        VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
    )
    position_embedding = Embedding(
        MAX_SEQ_LEN, WIDTH, generator=torch.Generator().manual_seed(1)
    )
    token_ids = torch.tensor(result.ids, dtype=torch.long)
    positions = torch.arange(len(result.ids))
    base = token_embedding.forward(token_ids) + position_embedding.forward(positions)
    base = GELU().forward(LayerNorm(WIDTH).forward(base))
    attention = MultiHeadSelfAttention(
        WIDTH, 2, generator=torch.Generator().manual_seed(2)
    )
    expected, cache = attention.forward(base)

    assert result.vectors is not None
    torch.testing.assert_close(result.vectors, expected)
    assert result.attention_weights is not None
    torch.testing.assert_close(result.attention_weights, cache["weights"])
    assert result.scores is not None
    assert result.answer == int(result.scores.max(dim=0).indices)


def test_future_stage_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="1.7"):
        run(EXAMPLE, stage=7)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run("", stage=1)
