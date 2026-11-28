"""Frozen chapter 1 character baseline for the section 2.1 demo.

This module is chapter-specific: it is not part of the tokenizer interface
used by the model. The tokenizer interface lives in :mod:`llm101.tokenizer`;
this file only keeps the old 68-symbol table visible for the before/after
comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()\n"
SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>")
CHARACTER_VOCABULARY = (*_CHARACTERS, *SPECIAL_TOKENS)

_TOKEN_TO_ID = {token: index for index, token in enumerate(CHARACTER_VOCABULARY)}
UNK_ID = _TOKEN_TO_ID["<unk>"]
BOS_ID = _TOKEN_TO_ID["<bos>"]
EOS_ID = _TOKEN_TO_ID["<eos>"]

CONTEXT_LIMIT = 16
ENGLISH_SAMPLE = "the cat sat on the mat"
CHINESE_SAMPLE = "小猫坐在垫子上"

# The hand merge is a 2.1 placeholder, not the algorithm from section 2.2.
_HAND_MERGES: tuple[tuple[str, str], ...] = (
    ("a", "t"),
    ("s", "at"),
    ("m", "at"),
    ("t", "h"),
    ("th", "e"),
)


def character_encode(
    text: str,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Encode text one code point at a time with the frozen 68-symbol table."""
    ids = [_TOKEN_TO_ID.get(character, UNK_ID) for character in text]
    if add_bos:
        ids.insert(0, BOS_ID)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def character_decode(
    ids: Sequence[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode ids with the frozen table, dropping special tokens by default."""
    pieces: list[str] = []
    for token_id in ids:
        if token_id < 0 or token_id >= len(CHARACTER_VOCABULARY):
            raise ValueError(f"token id out of range: {token_id}")
        token = CHARACTER_VOCABULARY[token_id]
        if skip_special_tokens and token in SPECIAL_TOKENS:
            continue
        pieces.append(token)
    return "".join(pieces)


def _merge_symbols(symbols: Sequence[str], pair: tuple[str, str]) -> list[str]:
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


def hand_merge(text: str) -> list[str]:
    """Apply the five section-2.1 merges to a string of code points."""
    symbols = list(text)
    for pair in _HAND_MERGES:
        symbols = _merge_symbols(symbols, pair)
    return symbols


@dataclass(frozen=True)
class BaselineReport:
    """The character-level baseline plus the hand-merge demonstration."""

    text: str
    ids: list[int]
    unknown_count: int
    decoded: str
    hand_symbols: list[str]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def token_count(self) -> int:
        return len(self.ids)

    @property
    def over_context(self) -> bool:
        return self.token_count > CONTEXT_LIMIT

    @property
    def hand_token_count(self) -> int:
        return len(self.hand_symbols) + 2


def build_report(text: str) -> BaselineReport:
    """Run the frozen character baseline and the hand merge for one text."""
    ids = character_encode(text, add_bos=True, add_eos=True)
    return BaselineReport(
        text=text,
        ids=ids,
        unknown_count=sum(1 for token_id in ids if token_id == UNK_ID),
        decoded=character_decode(ids),
        hand_symbols=hand_merge(text),
    )


def context_error(text: str) -> str | None:
    """Return the model's context error for ``text``, or ``None`` if it fits."""
    import torch

    from llm101.model import GPTConfig, TinyGPT

    ids = character_encode(text, add_bos=True, add_eos=True)
    config = GPTConfig(
        vocab_size=len(CHARACTER_VOCABULARY),
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        max_seq_len=CONTEXT_LIMIT,
    )
    model = TinyGPT(config, generator=torch.Generator().manual_seed(0))
    try:
        model.forward(torch.tensor([ids], dtype=torch.long))
    except ValueError as error:
        return str(error)
    return None
