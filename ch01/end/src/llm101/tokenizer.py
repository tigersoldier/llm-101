"""Character-level tokenizer (chapter 1).

One code point maps to one id. Chapter 2 replaces the splitter behind the
same five-member interface: ``encode``, ``decode``, ``vocab_size``, ``bos_id``
and ``eos_id``.
"""

from __future__ import annotations

from collections.abc import Sequence

_SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>")
UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN = _SPECIAL_TOKENS

# 26 lowercase + 26 uppercase + space, newline and 11 punctuation marks.
_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:'\"-()\n"

DEFAULT_VOCABULARY = (*_CHARACTERS, UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN)


class CharacterTokenizer:
    """Map text to token ids one code point at a time."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        unknown_token: str = UNKNOWN_TOKEN,
    ) -> None:
        tokens = tuple(vocabulary)
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulary entries must be unique")

        self._vocabulary = tokens
        self._token_to_id = {token: index for index, token in enumerate(tokens)}
        self._unknown_token = unknown_token
        self._special_tokens = frozenset({unknown_token, BOS_TOKEN, EOS_TOKEN})

        missing = self._special_tokens - self._token_to_id.keys()
        if missing:
            raise ValueError(f"vocabulary is missing special tokens: {sorted(missing)}")

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def bos_id(self) -> int:
        return self._token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[EOS_TOKEN]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert every code point to an id, mapping unknown ones to ``<unk>``."""
        unknown_id = self._token_to_id[self._unknown_token]
        ids = [self._token_to_id.get(character, unknown_id) for character in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Turn ids back into text, dropping special tokens by default."""
        pieces: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self._vocabulary):
                raise ValueError(f"token id out of range: {token_id}")
            token = self._vocabulary[token_id]
            if skip_special_tokens and token in self._special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)
