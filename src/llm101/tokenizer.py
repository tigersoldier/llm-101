"""A deliberately small, Unicode-codepoint BPE tokenizer for teaching.

This is not GPT's byte-level tokenizer: it starts from Unicode code points and
applies learned adjacent-pair merges in rank order. The trainer is intentionally
clear rather than fast and keeps its input sequences in memory.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
_FORMAT_VERSION = 1
_UNKNOWN_SYMBOL = "\0"
_Pair = tuple[str, str]


def _merge_pair(sequence: Sequence[str], pair: _Pair) -> list[str]:
    """Merge non-overlapping occurrences of ``pair`` from left to right."""
    left, right = pair
    merged: list[str] = []
    index = 0

    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left
            and sequence[index + 1] == right
        ):
            merged.append(left + right)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1

    return merged


@dataclass
class BPETokenizer:
    """BPE tokenizer with a transparent training and serialization format."""

    initial_tokens: tuple[str, ...]
    merges: tuple[_Pair, ...]
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    _vocabulary: tuple[str, ...] = field(init=False, repr=False)
    _token_to_id: dict[str, int] = field(init=False, repr=False)
    _initial_token_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.initial_tokens = tuple(self.initial_tokens)
        self.merges = tuple((left, right) for left, right in self.merges)
        self.special_tokens = tuple(self.special_tokens)

        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        if any(not token for token in self.special_tokens):
            raise ValueError("special tokens cannot be empty")
        if len(set(self.initial_tokens)) != len(self.initial_tokens):
            raise ValueError("initial_tokens must be unique")
        if set(self.special_tokens) & set(self.initial_tokens):
            raise ValueError("initial tokens cannot collide with special tokens")

        known_tokens = set(self.initial_tokens)
        vocabulary = list(self.special_tokens) + list(self.initial_tokens)
        vocabulary_set = set(vocabulary)
        for left, right in self.merges:
            if left not in known_tokens or right not in known_tokens:
                raise ValueError(f"merge references an unknown token: {(left, right)!r}")
            merged = left + right
            if merged in self.special_tokens:
                raise ValueError("a BPE merge cannot create a special token")
            known_tokens.add(merged)
            if merged not in vocabulary_set:
                vocabulary.append(merged)
                vocabulary_set.add(merged)

        self._vocabulary = tuple(vocabulary)
        self._token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self._initial_token_set = frozenset(self.initial_tokens)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        *,
        min_frequency: int = 2,
        max_texts: int | None = None,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> BPETokenizer:
        """Learn frequent adjacent code-point merges from ``texts``.

        ``max_texts`` limits only tokenizer training, not later LM training. It
        is useful because this intentionally simple implementation repeatedly
        scans its input; a deterministic text sample is often sufficient for
        teaching BPE on a large corpus.
        """
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if max_texts is not None and max_texts < 1:
            raise ValueError("max_texts must be positive when provided")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        sequences: list[list[str]] = []
        alphabet: set[str] = set()
        reserved = tuple(special_tokens)

        for index, text in enumerate(texts):
            if max_texts is not None and index >= max_texts:
                break
            if not isinstance(text, str):
                raise TypeError("BPE training data must contain strings")
            if any(token in text for token in reserved):
                raise ValueError("training text contains a reserved special-token spelling")
            if _UNKNOWN_SYMBOL in text:
                raise ValueError("training text contains the reserved NUL code point")
            if text:
                sequence = list(text)
                sequences.append(sequence)
                alphabet.update(sequence)

        if not sequences:
            raise ValueError("cannot train BPE without non-empty texts")

        initial_tokens = tuple(sorted(alphabet))
        minimum_size = len(reserved) + len(initial_tokens)
        if vocab_size < minimum_size:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for the observed alphabet; "
                f"need at least {minimum_size} (special tokens + code points)"
            )

        vocabulary = set(initial_tokens)
        merges: list[_Pair] = []

        while len(reserved) + len(vocabulary) < vocab_size:
            pair_counts: Counter[_Pair] = Counter()
            for sequence in sequences:
                pair_counts.update(pairwise(sequence))

            if not pair_counts:
                break

            # Stable tie-breaking makes repeated runs on the same corpus exact.
            best_pair, frequency = min(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_frequency:
                break

            merges.append(best_pair)
            vocabulary.add(best_pair[0] + best_pair[1])
            sequences = [_merge_pair(sequence, best_pair) for sequence in sequences]

        return cls(
            initial_tokens=initial_tokens,
            merges=tuple(merges),
            special_tokens=tuple(special_tokens),
        )

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Tokens ordered by their integer IDs."""
        return self._vocabulary

    @property
    def token_to_id(self) -> dict[str, int]:
        """A copy of the token-to-ID mapping."""
        return self._token_to_id.copy()

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def special_token_id(self, token: str) -> int:
        try:
            return self._token_to_id[token]
        except KeyError as error:
            raise ValueError(f"unknown special token: {token!r}") from error

    @property
    def pad_id(self) -> int:
        return self.special_token_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self.special_token_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.special_token_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.special_token_id("<eos>")

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text, optionally adding beginning/end-of-sequence IDs."""
        if any(token in text for token in self.special_tokens):
            raise ValueError("input text contains a reserved special-token spelling")

        if _UNKNOWN_SYMBOL in text:
            raise ValueError("input text contains the reserved NUL code point")

        # The reserved NUL symbol marks an unknown code point and blocks merges.
        sequence = [
            char if char in self._initial_token_set else _UNKNOWN_SYMBOL for char in text
        ]
        for pair in self.merges:
            sequence = _merge_pair(sequence, pair)

        ids = [
            self.unk_id if piece == _UNKNOWN_SYMBOL else self._token_to_id.get(piece, self.unk_id)
            for piece in sequence
        ]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text; unknown IDs decode as U+FFFD."""
        output: list[str] = []
        skippable = {"<pad>", "<bos>", "<eos>"}

        for token_id in ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id!r}")
            piece = self._vocabulary[token_id]
            if piece == "<unk>":
                output.append("�")
            elif piece in self.special_tokens and skip_special_tokens and piece in skippable:
                continue
            else:
                output.append(piece)

        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Save the small, inspectable tokenizer model as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "special_tokens": list(self.special_tokens),
            "initial_tokens": list(self.initial_tokens),
            "merges": [list(pair) for pair in self.merges],
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported BPE tokenizer format version")
        return cls(
            initial_tokens=tuple(payload["initial_tokens"]),
            merges=tuple(tuple(pair) for pair in payload["merges"]),
            special_tokens=tuple(payload["special_tokens"]),
        )
