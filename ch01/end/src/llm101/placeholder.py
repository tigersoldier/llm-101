"""The chapter 1 placeholder pipeline, upgraded one section at a time.

``run(text, stage)`` uses real components for sections up to ``stage`` and
simple placeholders afterwards. The placeholders exist so the pipeline can run
before every component is written; later sections replace them and this module
is removed once the full model takes over.
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
    vectors: torch.Tensor
    mixed: torch.Tensor
    scores: torch.Tensor
    top: tuple[tuple[int, float], ...]


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _placeholder_ids(text: str) -> list[int]:
    return [ord(character) % VOCAB_SIZE for character in text]


def _placeholder_vectors(ids: list[int]) -> torch.Tensor:
    rows: list[list[float]] = []
    total = 0
    for token_id in ids:
        total += token_id
        rows.append([float((total + index) % WIDTH) for index in range(WIDTH)])
    return torch.tensor([rows], dtype=torch.float32)


def _placeholder_mix(vectors: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    running = torch.zeros(vectors.shape[-1], dtype=vectors.dtype)
    for position in range(vectors.shape[1]):
        running = torch.remainder(running + vectors[0, position], WIDTH)
        rows.append(running.clone())
    if not rows:
        return vectors
    return torch.stack(rows).unsqueeze(0)


def _placeholder_scores(mixed: torch.Tensor) -> torch.Tensor:
    last = mixed[0, -1].tolist()
    seed = sum(int((index + 1) * value) for index, value in enumerate(last))
    return torch.tensor(
        [
            float((seed * (index + 1) + index) % VOCAB_SIZE)
            for index in range(VOCAB_SIZE)
        ],
        dtype=torch.float32,
    )


def _top(scores: torch.Tensor, count: int = 3) -> tuple[tuple[int, float], ...]:
    values = scores.tolist()
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return tuple((index, values[index]) for index in order[:count])


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

        embedding = Embedding(VOCAB_SIZE, WIDTH, generator=_generator(0))
        token_ids = torch.tensor([ids], dtype=torch.long)
        vectors = embedding.forward(token_ids)
    else:
        vectors = _placeholder_vectors(ids)

    mixed = _placeholder_mix(vectors)

    if stage >= 3:
        from llm101.layers import Linear

        output = Linear(WIDTH, VOCAB_SIZE, generator=_generator(1))
        scores = output.forward(mixed)[0, -1]
    else:
        scores = _placeholder_scores(mixed)

    return PipelineResult(
        stage=stage,
        text=text,
        ids=ids,
        vectors=vectors,
        mixed=mixed,
        scores=scores,
        top=_top(scores),
    )
