"""BPE tokenizer skeleton (chapter 2).

The five members used by the rest of the course are ``encode``, ``decode``,
``vocab_size``, ``bos_id`` and ``eos_id``. Chapter 2 adds ``train``, ``save``
and ``load``; the merge helpers are introduced in section 2.2.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

Pair = tuple[str, str]
SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>")
UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS


def merge_pair(symbols: Sequence[str], pair: Pair) -> list[str]:
    """Merge non-overlapping occurrences of ``pair`` from left to right."""
    raise NotImplementedError("TODO 2.2: merge one adjacent pair")


def count_pairs(sequences: Iterable[Sequence[str]]) -> Counter[Pair]:
    """Count adjacent symbol pairs within each sequence."""
    raise NotImplementedError("TODO 2.2: count adjacent pairs")


def learn_merges(
    texts: Iterable[str],
    num_merges: int,
    *,
    min_frequency: int = 2,
) -> list[Pair]:
    """Learn up to ``num_merges`` pairs, most frequent first."""
    raise NotImplementedError("TODO 2.2: learn adjacent-pair merges")


class BPETokenizer:
    """BPE tokenizer with three special tokens and a JSON save format."""

    def __init__(
        self,
        initial_tokens: Sequence[str],
        merges: Sequence[Pair],
        *,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        raise NotImplementedError("TODO 2.3: build the vocabulary")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        *,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Learn a vocabulary from ``texts`` and return a tokenizer."""
        raise NotImplementedError("TODO 2.3: train the vocabulary")

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Tokens ordered by their integer ids."""
        raise NotImplementedError("TODO 2.3: expose the vocabulary")

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary."""
        raise NotImplementedError("TODO 2.3: expose the vocabulary size")

    @property
    def unk_id(self) -> int:
        """Id of the ``<unk>`` token."""
        raise NotImplementedError("TODO 2.3: expose the unknown-token id")

    @property
    def bos_id(self) -> int:
        """Id of the ``<bos>`` token."""
        raise NotImplementedError("TODO 2.3: expose the beginning-of-sequence id")

    @property
    def eos_id(self) -> int:
        """Id of the ``<eos>`` token."""
        raise NotImplementedError("TODO 2.3: expose the end-of-sequence id")

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text, optionally adding beginning/end-of-sequence ids."""
        raise NotImplementedError("TODO 2.4: encode text")

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode ids to text, dropping special tokens by default."""
        raise NotImplementedError("TODO 2.4: decode ids")

    def save(self, path: str | Path) -> None:
        """Save the tokenizer ingredients as UTF-8 JSON."""
        raise NotImplementedError("TODO 2.4: save the tokenizer")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer saved by :meth:`save`."""
        raise NotImplementedError("TODO 2.4: load the tokenizer")
