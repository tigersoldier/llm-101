"""Atomic serialization for training checkpoints and inference-ready models."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from llm101.model import GPTConfig, TinyGPT
from llm101.optim import AdamW, AdamWState

_FORMAT_VERSION = 1


def _model_weights(model: TinyGPT) -> list[torch.Tensor]:
    return [parameter.value.detach().cpu().clone() for parameter in model.parameters()]


def load_model_weights(model: TinyGPT, weights: object) -> None:
    """Copy a serialized parameter list into a model after shape validation."""
    if not isinstance(weights, list) or len(weights) != len(model.parameters()):
        raise ValueError("saved model parameter count does not match model")
    for parameter, saved in zip(model.parameters(), weights, strict=True):
        if not isinstance(saved, torch.Tensor):
            raise TypeError("saved model parameters must be tensors")
        if saved.shape != parameter.value.shape:
            raise ValueError("saved model parameter shape does not match model")
        parameter.value.copy_(
            saved.to(device=parameter.value.device, dtype=parameter.value.dtype)
        )


def _save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        torch.save(payload, partial)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def save_model_artifact(
    path: str | Path,
    model: TinyGPT,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save only architecture, weights, and JSON-like release metadata."""
    _save_payload(
        path,
        {
            "format_version": _FORMAT_VERSION,
            "artifact_kind": "model",
            "config": asdict(model.config),
            "model_state": _model_weights(model),
            "metadata": metadata or {},
        },
    )


def save_training_checkpoint(
    path: str | Path,
    model: TinyGPT,
    optimizer: AdamW,
    *,
    step: int,
    elapsed_seconds: float,
    tokens_seen: int,
    language_tokens_seen: dict[str, int],
    batch_generator: torch.Generator,
    metadata: dict[str, Any],
) -> None:
    """Save model, optimizer, counters, and sampler RNG for exact resumption."""
    _save_payload(
        path,
        {
            "format_version": _FORMAT_VERSION,
            "artifact_kind": "training_checkpoint",
            "config": asdict(model.config),
            "model_state": _model_weights(model),
            "optimizer_state": optimizer.state_dict(),
            "training_state": {
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                "tokens_seen": tokens_seen,
                "language_tokens_seen": language_tokens_seen,
                "batch_generator_state": batch_generator.get_state().cpu(),
            },
            "metadata": metadata,
        },
    )


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load a weights-only checkpoint and reject unsupported formats."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a dictionary")
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    return cast(dict[str, Any], payload)


def load_model_artifact(path: str | Path) -> tuple[TinyGPT, dict[str, Any]]:
    """Create a model from a model artifact or a training checkpoint."""
    payload = load_payload(path)
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    config = GPTConfig(**cast(dict[str, int], raw_config))
    model = TinyGPT(config)
    load_model_weights(model, payload.get("model_state"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a dictionary")
    return model, cast(dict[str, Any], metadata)


def load_adamw_state(optimizer: AdamW, state: object) -> None:
    """Restore optimizer state from the dictionary held by a checkpoint."""
    if not isinstance(state, dict):
        raise TypeError("saved optimizer state must be a dictionary")
    optimizer.load_state_dict(cast(AdamWState, state))
