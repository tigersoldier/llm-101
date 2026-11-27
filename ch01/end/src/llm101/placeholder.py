"""The chapter 1 placeholder pipeline, upgraded one section at a time.

Sections 1.1 and 1.2 map text to a placeholder answer id and then to a
character. Section 1.3 embeds every position, sums the vectors, and uses a
fixed output matrix to score the 68 vocabulary entries; the full model arrives
in section 1.8. Stage 3 already takes the max score for the placeholder
answer; later sections fill the remaining placeholders. This module stays as
the per-stage demo entry point through section 1.9, and the full-model branch
simply stops calling the placeholder functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VOCAB_SIZE = 68
WIDTH = 16
MAX_SEQ_LEN = 16
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class PipelineResult:
    """Everything the demo prints for one input and one stage."""

    stage: int
    text: str
    ids: list[int]
    answer: int | None = None
    character: str | None = None
    vectors: torch.Tensor | None = None
    position_vectors: torch.Tensor | None = None
    output: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None


_OUTPUT_MATRIX: torch.Tensor = (
    torch.randn(WIDTH, VOCAB_SIZE, generator=torch.Generator().manual_seed(24)) * 0.1
)


def _placeholder_ids(text: str) -> list[int]:
    return [ord(character) % VOCAB_SIZE for character in text]


def _placeholder_answer(ids: list[int]) -> int:
    return (sum(ids) * 31) % VOCAB_SIZE


def _placeholder_character(answer: int) -> str:
    return ALPHABET[answer % len(ALPHABET)]


def run(text: str, stage: int = 1) -> PipelineResult:
    """Run the pipeline as it stands at the end of section ``stage``."""
    if not 1 <= stage <= 7:
        raise ValueError("stage must be between 1 and 7")
    if not text:
        raise ValueError("text must contain at least one character")

    if stage >= 2:
        from llm101.tokenizer import DEFAULT_VOCABULARY, CharacterTokenizer

        tokenizer = CharacterTokenizer(DEFAULT_VOCABULARY)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)

        if stage == 7:
            from llm101.model import GPTConfig, TinyGPT

            config = GPTConfig(
                vocab_size=VOCAB_SIZE,
                d_model=WIDTH,
                n_layers=2,
                n_heads=2,
                d_ff=32,
                max_seq_len=MAX_SEQ_LEN,
            )
            model = TinyGPT(
                config,
                generator=torch.Generator().manual_seed(3),
            )
            logits = model.forward(torch.tensor(ids, dtype=torch.long))
            scores = logits[-1]
            answer = int(scores.argmax(dim=0))
            return PipelineResult(
                stage=stage,
                text=text,
                ids=ids,
                answer=answer,
                character=tokenizer.decode([answer]),
                scores=scores,
            )

        if stage >= 3:
            from llm101.layers import Embedding

            embedding = Embedding(
                VOCAB_SIZE, WIDTH, generator=torch.Generator().manual_seed(0)
            )
            token_ids = torch.tensor(ids, dtype=torch.long)
            vectors = embedding.forward(token_ids)
            position_vectors = None
            if stage >= 4:
                if len(ids) > MAX_SEQ_LEN:
                    raise ValueError(
                        f"text has {len(ids)} positions, longer than the model "
                        f"context of {MAX_SEQ_LEN}"
                    )
                position_embedding = Embedding(
                    MAX_SEQ_LEN,
                    WIDTH,
                    generator=torch.Generator().manual_seed(1),
                )
                positions = torch.arange(len(ids), dtype=torch.long)
                position_vectors = position_embedding.forward(positions)
                vectors = vectors + position_vectors
            if stage >= 5:
                from llm101.layers import GELU, LayerNorm

                layer_norm = LayerNorm(WIDTH)
                gelu = GELU()
                vectors = gelu.forward(layer_norm.forward(vectors))
            attention_weights = None
            if stage >= 6:
                from llm101.attention import MultiHeadSelfAttention

                attention = MultiHeadSelfAttention(
                    WIDTH, 2, generator=torch.Generator().manual_seed(2)
                )
                vectors, attention_cache = attention.forward(vectors)
                attention_weights = attention_cache["weights"]
            output = vectors.sum(dim=0)
            scores = output @ _OUTPUT_MATRIX
            answer = int(scores.max(dim=0).indices)
            return PipelineResult(
                stage=stage,
                text=text,
                ids=ids,
                answer=answer,
                character=_placeholder_character(answer),
                vectors=vectors,
                position_vectors=position_vectors,
                output=output,
                scores=scores,
                attention_weights=attention_weights,
            )

        answer = _placeholder_answer(ids)
        return PipelineResult(
            stage=stage,
            text=text,
            ids=ids,
            answer=answer,
            character=tokenizer.decode([answer]),
        )

    ids = _placeholder_ids(text)
    answer = _placeholder_answer(ids)
    return PipelineResult(
        stage=stage,
        text=text,
        ids=ids,
        answer=answer,
        character=_placeholder_character(answer),
    )
