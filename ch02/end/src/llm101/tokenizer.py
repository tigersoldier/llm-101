"""BPE tokenizer (chapter 2).

The five members used by the rest of the course are ``encode``, ``decode``,
``vocab_size``, ``bos_id`` and ``eos_id``. Section 2.2 implements the merge
helpers; section 2.3 implements vocabulary construction and training. Encoding,
decoding and persistence arrive in section 2.4.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

Pair = tuple[str, str]
SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>")
UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS


def merge_pair(symbols: Sequence[str], pair: Pair) -> list[str]:
    """Merge non-overlapping occurrences of ``pair`` from left to right."""
    left, right = pair
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == left
            and symbols[index + 1] == right
        ):
            merged.append(left + right)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def count_pairs(sequences: Iterable[Sequence[str]]) -> Counter[Pair]:
    """Count adjacent symbol pairs within each sequence."""
    counts: Counter[Pair] = Counter()
    for sequence in sequences:
        counts.update(pairwise(sequence))
    return counts


def learn_merges(
    texts: Iterable[str],
    num_merges: int,
    *,
    min_frequency: int = 2,
) -> list[Pair]:
    """Learn up to ``num_merges`` pairs, most frequent first.

    Ties are broken by the lexicographic order of ``(left, right)``, so the
    same corpus always produces the same merge list.
    """
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    sequences = [list(text) for text in texts]
    if not sequences or not any(sequences):
        raise ValueError("cannot learn merges without non-empty texts")

    merges: list[Pair] = []
    for _ in range(num_merges):
        counts = count_pairs(sequences)
        if not counts:
            break
        best_pair, frequency = min(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if frequency < min_frequency:
            break
        merges.append(best_pair)
        sequences = [merge_pair(sequence, best_pair) for sequence in sequences]
    return merges


class BPETokenizer:
    """BPE tokenizer with three special tokens and a JSON save format."""

    def __init__(
        self,
        initial_tokens: Sequence[str],
        merges: Sequence[Pair],
        *,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        special_tokens = tuple(special_tokens)
        initial_tokens = tuple(initial_tokens)
        merges = tuple((left, right) for left, right in merges)

        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")
        if any(not token for token in special_tokens):
            raise ValueError("special tokens cannot be empty")
        required = {UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN}
        if not required.issubset(special_tokens):
            raise ValueError(f"special_tokens must include {sorted(required)}")
        if len(set(initial_tokens)) != len(initial_tokens):
            raise ValueError("initial_tokens must be unique")
        if any(not token for token in initial_tokens):
            raise ValueError("initial tokens cannot be empty")
        if set(initial_tokens) & set(special_tokens):
            raise ValueError("initial tokens cannot collide with special tokens")

        vocabulary = list(special_tokens) + list(initial_tokens)
        known_tokens = set(vocabulary)
        special_set = set(special_tokens)
        for left, right in merges:
            if not left or not right:
                raise ValueError("merge pairs cannot contain empty tokens")
            if left not in known_tokens or right not in known_tokens:
                raise ValueError(
                    f"merge references an unknown token: {(left, right)!r}"
                )
            merged = left + right
            if merged in special_set:
                raise ValueError("a BPE merge cannot create a special token")
            known_tokens.add(merged)
            if merged not in vocabulary:
                vocabulary.append(merged)

        self.special_tokens = special_tokens
        self.initial_tokens = initial_tokens
        self.merges = merges
        self._vocabulary = tuple(vocabulary)
        self._token_to_id = {
            token: index for index, token in enumerate(vocabulary)
        }

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
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        materialized = list(texts)
        if not materialized or not any(materialized):
            raise ValueError("cannot train without non-empty texts")
        alphabet = sorted(set("".join(materialized)))
        special_tokens = tuple(special_tokens)
        minimum_size = len(special_tokens) + len(alphabet)
        if vocab_size < minimum_size:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for the observed alphabet; "
                f"need at least {minimum_size}"
            )

        merges = learn_merges(
            materialized,
            vocab_size - minimum_size,
            min_frequency=min_frequency,
        )
        return cls(
            initial_tokens=tuple(alphabet),
            merges=merges,
            special_tokens=special_tokens,
        )

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Tokens ordered by their integer ids."""
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary."""
        return len(self._vocabulary)

    @property
    def unk_id(self) -> int:
        """Id of the ``<unk>`` token."""
        return self._token_to_id[UNKNOWN_TOKEN]

    @property
    def bos_id(self) -> int:
        """Id of the ``<bos>`` token."""
        return self._token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        """Id of the ``<eos>`` token."""
        return self._token_to_id[EOS_TOKEN]

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
