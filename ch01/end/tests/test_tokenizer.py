"""Tests for the chapter 1 character-level tokenizer."""

import pytest

from llm101.tokenizer import DEFAULT_VOCABULARY, CharacterTokenizer


@pytest.fixture()
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer(DEFAULT_VOCABULARY)


def test_vocabulary_and_special_ids(tokenizer: CharacterTokenizer) -> None:
    assert tokenizer.vocab_size == 68
    assert tokenizer.bos_id != tokenizer.eos_id
    assert DEFAULT_VOCABULARY[tokenizer.bos_id] == "<bos>"
    assert DEFAULT_VOCABULARY[tokenizer.eos_id] == "<eos>"


def test_round_trip(tokenizer: CharacterTokenizer) -> None:
    text = "The little rabbit found a red ball."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_characters_map_to_unk(tokenizer: CharacterTokenizer) -> None:
    unknown_id = DEFAULT_VOCABULARY.index("<unk>")
    ids = tokenizer.encode("你好")

    assert ids == [unknown_id, unknown_id]
    assert tokenizer.decode(ids) == ""


def test_special_tokens_do_not_enter_decoded_text(
    tokenizer: CharacterTokenizer,
) -> None:
    ids = tokenizer.encode("hi", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hi"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<bos>hi<eos>"


def test_missing_special_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="special"):
        CharacterTokenizer(["a", "b"])


def test_decode_rejects_out_of_range_ids(tokenizer: CharacterTokenizer) -> None:
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size])
