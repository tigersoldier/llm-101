"""The chapter 1 placeholder pipeline, upgraded one section at a time.

Stages 1 and 2 are pure Python lists; ``torch`` appears only when the embedding
and linear layers arrive in section 1.3. Later sections replace the remaining
placeholders and this module is removed once the full model takes over.
"""

from __future__ import annotations

from dataclasses import dataclass

VOCAB_SIZE = 68
WIDTH = 16


@dataclass(frozen=True)
class PipelineResult:
    """Everything the demo prints for one input and one stage."""

    stage: int
    text: str
    ids: list[int]
    vectors: list[list[float]]
    scores: list[float]
    answer: int


def _placeholder_ids(text: str) -> list[int]:
    return [ord(character) % VOCAB_SIZE for character in text]


def _placeholder_vectors(ids: list[int]) -> list[list[float]]:
    rows: list[list[float]] = []
    total = 0
    for token_id in ids:
        total += token_id
        rows.append([float((total + index) % WIDTH) for index in range(WIDTH)])
    return rows


def _placeholder_scores(vectors: list[list[float]]) -> list[float]:
    last = vectors[-1]
    seed = sum(int((index + 1) * value) for index, value in enumerate(last))
    return [
        float((seed * 31 + index * 17 + index * index * 7) % VOCAB_SIZE)
        for index in range(VOCAB_SIZE)
    ]


def argmax(values: list[float]) -> int:
    """Return the index of the largest value; ties go to the first one."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


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
        import torch

        from llm101.layers import Embedding, Linear

        embedding = Embedding(
            VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
        )
        token_ids = torch.tensor([ids], dtype=torch.long)
        vectors = embedding.forward(token_ids)[0].tolist()

        output = Linear(WIDTH, VOCAB_SIZE, generator=torch.Generator().manual_seed(1))
        hidden = torch.tensor([vectors], dtype=torch.float32)
        scores = output.forward(hidden)[0, -1].tolist()
    else:
        vectors = _placeholder_vectors(ids)
        scores = _placeholder_scores(vectors)

    return PipelineResult(
        stage=stage,
        text=text,
        ids=ids,
        vectors=vectors,
        scores=scores,
        answer=argmax(scores),
    )
