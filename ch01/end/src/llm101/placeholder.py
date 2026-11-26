"""The chapter 1 placeholder pipeline, upgraded one section at a time.

Sections 1.1 and 1.2 map text to a placeholder answer directly. Section 1.3
opens the box and returns the 16-number vector per position; the model output
scores and argmax arrive with the full model in section 1.7. Later sections
replace the remaining placeholders and this module is removed once the model
takes over.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VOCAB_SIZE = 68
WIDTH = 16


@dataclass(frozen=True)
class PipelineResult:
    """Everything the demo prints for one input and one stage."""

    stage: int
    text: str
    ids: list[int]
    answer: int | None = None
    vectors: torch.Tensor | None = None


def _placeholder_ids(text: str) -> list[int]:
    return [ord(character) % VOCAB_SIZE for character in text]


def _placeholder_answer(ids: list[int]) -> int:
    return (sum(ids) * 31) % VOCAB_SIZE


def run(text: str, stage: int = 1) -> PipelineResult:
    """Run the pipeline as it stands at the end of section ``stage``."""
    if not 1 <= stage <= 11:
        raise ValueError("stage must be between 1 and 11")
    if not text:
        raise ValueError("text must contain at least one character")
    if stage >= 4:
        raise NotImplementedError("position embeddings land in section 1.4")

    if stage >= 2:
        from llm101.tokenizer import DEFAULT_VOCABULARY, CharacterTokenizer

        tokenizer = CharacterTokenizer(DEFAULT_VOCABULARY)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    else:
        ids = _placeholder_ids(text)

    if stage >= 3:
        from llm101.layers import Embedding

        embedding = Embedding(
            VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
        )
        token_ids = torch.tensor([ids], dtype=torch.long)
        vectors = embedding.forward(token_ids)
        return PipelineResult(stage=stage, text=text, ids=ids, vectors=vectors)

    return PipelineResult(
        stage=stage,
        text=text,
        ids=ids,
        answer=_placeholder_answer(ids),
    )
