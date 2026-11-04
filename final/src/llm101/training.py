"""CPU data sampling and small utilities for explicit-gradient training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch

from llm101.losses import softmax_cross_entropy
from llm101.model import TinyGPT
from llm101.optim import AdamW

LanguageLengths: TypeAlias = list[tuple[str, int]]


@dataclass(frozen=True)
class LanguageSegment:
    """A half-open token range in a concatenated training stream."""

    name: str
    start: int
    stop: int

    @property
    def token_count(self) -> int:
        return self.stop - self.start


class TokenStream:
    """Memory-mapped int32 tokens with reproducible random next-token batches."""

    def __init__(self, path: str | Path, language_lengths: LanguageLengths) -> None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        byte_count = source.stat().st_size
        if byte_count % 4:
            raise ValueError(f"token stream size is not divisible by four: {source}")
        token_count = byte_count // 4
        if token_count < 2:
            raise ValueError("a token stream needs at least two token IDs")
        if not language_lengths:
            raise ValueError("at least one language segment is required")

        segments: list[LanguageSegment] = []
        position = 0
        names: set[str] = set()
        for name, length in language_lengths:
            if not name or name in names:
                raise ValueError("language segment names must be non-empty and unique")
            if length <= 0:
                raise ValueError("language segment lengths must be positive")
            names.add(name)
            segments.append(LanguageSegment(name, position, position + length))
            position += length
        if position != token_count:
            raise ValueError(
                f"language segments cover {position} tokens, but stream contains {token_count}"
            )

        self.path = source
        self.tokens = torch.from_file(
            str(source), shared=False, size=token_count, dtype=torch.int32
        )
        self.segments = tuple(segments)

    @property
    def token_count(self) -> int:
        return self.tokens.numel()

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Sample contiguous windows and count input-token coverage by language."""
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        max_start = self.token_count - sequence_length - 1
        if max_start < 0:
            raise ValueError("token stream is shorter than sequence_length + 1")

        starts = torch.randint(
            max_start + 1, (batch_size,), generator=generator, dtype=torch.long
        )
        starts_list = starts.tolist()
        inputs = torch.stack(
            [
                self.tokens[start : start + sequence_length].to(torch.long)
                for start in starts_list
            ]
        )
        targets = torch.stack(
            [
                self.tokens[start + 1 : start + sequence_length + 1].to(torch.long)
                for start in starts_list
            ]
        )

        coverage = {segment.name: 0 for segment in self.segments}
        for start in starts_list:
            stop = start + sequence_length
            for segment in self.segments:
                overlap = max(0, min(stop, segment.stop) - max(start, segment.start))
                coverage[segment.name] += overlap
        return inputs, targets, coverage


def clip_grad_norm_(model: TinyGPT, max_norm: float) -> float:
    """Clip explicit parameter gradients and return their pre-clip global norm."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    parameters = model.parameters()
    norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.norm() for parameter in parameters])
    )
    total_norm = float(norm.item())
    if not math.isfinite(total_norm):
        raise FloatingPointError("explicit gradient norm is not finite")
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return total_norm


def train_step(
    model: TinyGPT,
    optimizer: AdamW,
    token_stream: TokenStream,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    max_grad_norm: float,
) -> tuple[float, dict[str, int], float]:
    """Run one explicit-gradient next-token update and return loss and coverage."""
    optimizer.zero_grad()
    inputs, targets, coverage = token_stream.sample_batch(
        batch_size, sequence_length, generator=generator
    )
    logits = model.forward(inputs)
    loss, grad_logits = softmax_cross_entropy(logits, targets)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("training loss is not finite")
    model.backward(grad_logits)
    gradient_norm = clip_grad_norm_(model, max_grad_norm)
    optimizer.step()
    return float(loss.item()), coverage, gradient_norm
