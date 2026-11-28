"""Tests for the chapter 2 BPE tokenizer.

The first check turns green in 2.2; the other two need the 2.4 encode/decode
work, and the model check is part of the 2.6 acceptance. Failure messages
contain the section that owns the missing implementation.
"""

import pytest
import torch

from llm101.model import GPTConfig, TinyGPT
from llm101.tokenizer import BPETokenizer, count_pairs, learn_merges, merge_pair

TEXTS = ["low", "lower", "lowest"]


def test_learn_merges_shrinks_symbols() -> None:
    """TODO 2.2: frequent adjacent pairs become one symbol."""
    merges = learn_merges(TEXTS, 3, min_frequency=1)

    assert merges == [("l", "o"), ("lo", "w"), ("low", "e")]
    symbols = merge_pair(list("lowest"), merges[0])
    assert symbols == ["lo", "w", "e", "s", "t"]
    assert len(symbols) < len("lowest")


def test_merge_pair_left_to_right() -> None:
    """Merges scan left to right and never overlap."""
    assert merge_pair(["a", "a", "a"], ("a", "a")) == ["aa", "a"]
    assert merge_pair(["a", "a", "a", "a"], ("a", "a")) == ["aa", "aa"]
    assert merge_pair(["a", "b", "a", "b"], ("a", "b")) == ["ab", "ab"]


def test_merge_pair_no_match() -> None:
    """A pair that does not occur leaves the symbols unchanged."""
    assert merge_pair(["a", "b"], ("b", "a")) == ["a", "b"]


def test_count_pairs_across_sequences() -> None:
    """Pairs are counted inside each sequence, never across a boundary."""
    counts = count_pairs([["l", "o", "w"], ["l", "o", "w", "e", "r"]])

    assert counts == {
        ("l", "o"): 2,
        ("o", "w"): 2,
        ("w", "e"): 1,
        ("e", "r"): 1,
    }


def test_learn_merges_order_and_ties() -> None:
    """Frequencies choose the pair; ties are broken lexicographically."""
    assert learn_merges(TEXTS, 6, min_frequency=1) == [
        ("l", "o"),
        ("lo", "w"),
        ("low", "e"),
        ("lowe", "r"),
        ("lowe", "s"),
        ("lowes", "t"),
    ]


def test_learn_merges_min_frequency() -> None:
    """A pair below the frequency threshold stops the loop."""
    assert learn_merges(TEXTS, 6, min_frequency=2) == [
        ("l", "o"),
        ("lo", "w"),
        ("low", "e"),
    ]


def test_learn_merges_stops_at_num_merges() -> None:
    """The caller's merge budget is an upper bound."""
    assert learn_merges(TEXTS, 0) == []
    assert learn_merges(TEXTS, 2, min_frequency=1) == [
        ("l", "o"),
        ("lo", "w"),
    ]


def test_vocabulary_order_and_ids() -> None:
    """The vocabulary is special tokens, then alphabet, then merges."""
    tokenizer = BPETokenizer.train(TEXTS, 12)

    assert tokenizer.vocab_size == 12
    assert tokenizer.vocabulary == (
        "<unk>",
        "<bos>",
        "<eos>",
        "e",
        "l",
        "o",
        "r",
        "s",
        "t",
        "w",
        "lo",
        "low",
    )
    assert (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)


def test_train_rejects_small_vocab_size() -> None:
    """The target must fit the special tokens and the observed alphabet."""
    with pytest.raises(ValueError, match="at least 10"):
        BPETokenizer.train(TEXTS, 9)


def test_train_merge_count_matches_target() -> None:
    """The target size determines how many merges are attempted."""
    assert len(BPETokenizer.train(TEXTS, 12).merges) == 2
    assert len(BPETokenizer.train(TEXTS, 13).merges) == 3


def test_train_stops_at_min_frequency() -> None:
    """A high target can still stop when no pair reaches the threshold."""
    tokenizer = BPETokenizer.train(TEXTS, 512, min_frequency=2)

    assert tokenizer.vocab_size == 13
    assert len(tokenizer.merges) == 3


def test_train_rejects_duplicate_special_tokens() -> None:
    """Special tokens must be unique."""
    with pytest.raises(ValueError, match="unique"):
        BPETokenizer.train(
            TEXTS,
            12,
            special_tokens=("<unk>", "<bos>", "<eos>", "<eos>"),
        )


def test_duplicate_merge_result() -> None:
    """A merged string that already exists does not grow the vocabulary."""
    tokenizer = BPETokenizer(
        ("a", "b", "c"),
        (("a", "b"), ("b", "c"), ("ab", "c"), ("a", "bc")),
    )

    assert tokenizer.vocab_size == 9
    assert tokenizer.vocabulary == (
        "<unk>",
        "<bos>",
        "<eos>",
        "a",
        "b",
        "c",
        "ab",
        "bc",
        "abc",
    )


def test_tokenizer_contract_round_trip() -> None:
    """TODO 2.4: the five-member tokenizer contract survives BPE."""
    tokenizer = BPETokenizer.train(TEXTS, 12)
    ids = tokenizer.encode("lower", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "lower"
    assert tokenizer.vocab_size == 12


def test_vocab_size_matches_model() -> None:
    """2.6 acceptance: the tokenizer vocabulary feeds the model config."""
    tokenizer = BPETokenizer.train(TEXTS, 12)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
    )
    model = TinyGPT(config, generator=torch.Generator().manual_seed(0))
    ids = tokenizer.encode("low")

    logits = model.forward(torch.tensor([ids]))

    assert logits.shape == (1, len(ids), tokenizer.vocab_size)
