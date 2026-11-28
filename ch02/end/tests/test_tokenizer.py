"""Tests for the chapter 2 BPE tokenizer.

The first check turns green in 2.2; the other two need the 2.4 encode/decode
work, and the model check is part of the 2.6 acceptance. Failure messages
contain the section that owns the missing implementation.
"""

import torch

from llm101.model import GPTConfig, TinyGPT
from llm101.tokenizer import BPETokenizer, learn_merges, merge_pair

TEXTS = ["low", "lower", "lowest"]


def test_learn_merges_shrinks_symbols() -> None:
    """TODO 2.2: frequent adjacent pairs become one symbol."""
    merges = learn_merges(TEXTS, 3, min_frequency=1)

    assert merges == [("l", "o"), ("lo", "w"), ("low", "e")]
    symbols = merge_pair(list("lowest"), merges[0])
    assert symbols == ["lo", "w", "e", "s", "t"]
    assert len(symbols) < len("lowest")


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
