"""Character-level tokenizer (chapter 1 scaffold).

Chapter 2 replaces this with BPE behind the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class CharacterTokenizer:
    """Map text to token ids one code point at a time."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        unknown_token: str = "<unk>",
    ) -> None:
        raise NotImplementedError("TODO 1.2: store the vocabulary and id maps")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("TODO 1.2")

    @property
    def bos_id(self) -> int:
        raise NotImplementedError("TODO 1.2")

    @property
    def eos_id(self) -> int:
        raise NotImplementedError("TODO 1.2")

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError("TODO 1.2: encode text to ids")

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError("TODO 1.2: decode ids back to text")
