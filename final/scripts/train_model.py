"""Run one CPU-only, time-bounded bilingual TinyGPT pretraining experiment.

Example::

    uv run python scripts/train_model.py --hours 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from llm101.checkpoint import (
    load_adamw_state,
    load_model_weights,
    load_payload,
    save_model_artifact,
    save_training_checkpoint,
)
from llm101.inference import generate
from llm101.losses import softmax_cross_entropy
from llm101.model import GPTConfig, TinyGPT
from llm101.optim import AdamW
from llm101.tokenizer import BPETokenizer
from llm101.training import TokenStream, train_step

_ADAMW_BETAS = (0.9, 0.999)
_ADAMW_EPS = 1e-8

# The course trains a single joint Chinese/English model; no language tag is added.
CORPUS = "bilingual"

_PROMPTS = [
    "Once upon a time, there was a little girl named",
    "The little rabbit went to the",
    "从前，有一个小女孩，名叫",
    "小兔子走进了",
]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()


def _signature(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_streams(
    data_dir: Path, stats: dict[str, Any]
) -> list[tuple[str, TokenStream]]:
    english_count = int(stats["english_validation_tokens"])
    chinese_count = int(stats["chinese_validation_tokens"])
    return [
        (
            "mixed",
            TokenStream(
                data_dir / CORPUS / "validation.bin",
                [("english", english_count), ("chinese", chinese_count)],
            ),
        ),
        (
            "english",
            TokenStream(
                data_dir / CORPUS / "english_validation.bin",
                [("english", english_count)],
            ),
        ),
        (
            "chinese",
            TokenStream(
                data_dir / CORPUS / "chinese_validation.bin",
                [("chinese", chinese_count)],
            ),
        ),
    ]


def _estimate_loss(
    model: TinyGPT,
    stream: TokenStream,
    *,
    batch_size: int,
    sequence_length: int,
    batches: int,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_loss = 0.0
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets, _ = stream.sample_batch(
                batch_size, sequence_length, generator=generator
            )
            logits = model.forward(inputs)
            loss, _ = softmax_cross_entropy(logits, targets)
            total_loss += float(loss.item())
    return total_loss / batches


def _evaluate_all(
    model: TinyGPT,
    validation_streams: list[tuple[str, TokenStream]],
    *,
    batch_size: int,
    sequence_length: int,
    batches: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for index, (name, stream) in enumerate(validation_streams):
        loss = _estimate_loss(
            model,
            stream,
            batch_size=batch_size,
            sequence_length=sequence_length,
            batches=batches,
            seed=seed + index * 104_729,
        )
        results[name] = {"loss": loss, "perplexity": math.exp(loss)}
    return results


def _language_lengths(stats: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        ("english", int(stats["english_train_tokens"])),
        ("chinese", int(stats["chinese_train_tokens"])),
    ]


def _load_run_payload(
    path: Path,
    *,
    model: TinyGPT,
    optimizer: AdamW,
    batch_generator: torch.Generator,
    run_config: dict[str, Any],
) -> tuple[int, float, int, dict[str, int], float]:
    payload = load_payload(path)
    if payload.get("artifact_kind") != "training_checkpoint":
        raise ValueError(f"not a training checkpoint: {path}")
    if payload.get("config") != asdict(model.config):
        raise ValueError("checkpoint model config does not match this experiment")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("run_config") != run_config:
        raise ValueError("checkpoint data or training settings do not match this run")

    load_model_weights(model, payload.get("model_state"))
    load_adamw_state(optimizer, payload.get("optimizer_state"))
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise TypeError("checkpoint training_state must be a dictionary")
    generator_state = training_state.get("batch_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise TypeError("checkpoint is missing the batch generator state")
    batch_generator.set_state(generator_state)
    language_tokens = training_state.get("language_tokens_seen", {})
    if not isinstance(language_tokens, dict):
        raise TypeError("checkpoint language coverage must be a dictionary")
    return (
        int(training_state["step"]),
        float(training_state["elapsed_seconds"]),
        int(training_state["tokens_seen"]),
        {str(name): int(count) for name, count in language_tokens.items()},
        float(metadata.get("best_validation_loss", math.inf)),
    )


def _save_checkpoint(
    path: Path,
    model: TinyGPT,
    optimizer: AdamW,
    *,
    step: int,
    elapsed_seconds: float,
    tokens_seen: int,
    language_tokens_seen: dict[str, int],
    batch_generator: torch.Generator,
    run_config: dict[str, Any],
    best_validation_loss: float,
) -> None:
    save_training_checkpoint(
        path,
        model,
        optimizer,
        step=step,
        elapsed_seconds=elapsed_seconds,
        tokens_seen=tokens_seen,
        language_tokens_seen=language_tokens_seen,
        batch_generator=batch_generator,
        metadata={
            "run_config": run_config,
            "best_validation_loss": best_validation_loss,
        },
    )


def _final_samples(
    model: TinyGPT, tokenizer: BPETokenizer, seed: int
) -> list[dict[str, str]]:
    return [
        {
            "prompt": prompt,
            "output": generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=64,
                temperature=0.8,
                top_k=40,
                seed=seed + index,
            ),
        }
        for index, prompt in enumerate(_PROMPTS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every-seconds", type=float, default=300)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=300)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    if not math.isfinite(args.hours) or args.hours <= 0 or args.hours > 24:
        parser.error("--hours must be finite, positive, and no more than 24")
    if args.batch_size <= 0 or args.sequence_length <= 0:
        parser.error("batch size and sequence length must be positive")
    if args.eval_batches <= 0 or args.log_every_steps <= 0:
        parser.error("evaluation batches and log interval must be positive")
    if (
        not math.isfinite(args.eval_every_seconds)
        or not math.isfinite(args.checkpoint_every_seconds)
        or args.eval_every_seconds <= 0
        or args.checkpoint_every_seconds <= 0
    ):
        parser.error("evaluation and checkpoint intervals must be finite and positive")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("learning rate must be positive and weight decay non-negative")
    if (
        not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0
        or args.threads <= 0
    ):
        parser.error("gradient clip and thread count must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data_manifest_path = args.data_dir / "experiment_manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {data_manifest_path}; first run scripts/prepare_training_data.py"
        )
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    corpora = data_manifest["corpora"]
    stats = corpora[CORPUS]
    config = GPTConfig(
        vocab_size=int(stats["vocab_size"]),
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_seq_len=args.sequence_length,
    )
    train_lengths = _language_lengths(stats)
    train_stream = TokenStream(args.data_dir / CORPUS / "train.bin", train_lengths)
    validation_streams = _validation_streams(args.data_dir, stats)
    tokenizer_relative_path = Path(data_manifest["tokenizer_files"][CORPUS])
    if tokenizer_relative_path.is_absolute() or ".." in tokenizer_relative_path.parts:
        raise ValueError(
            "tokenizer path in data manifest must stay inside the data directory"
        )
    data_root = args.data_dir.resolve()
    tokenizer_path = (data_root / tokenizer_relative_path).resolve()
    if not tokenizer_path.is_relative_to(data_root):
        raise ValueError("tokenizer path in data manifest escapes the data directory")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            "saved tokenizer and experiment manifest disagree on vocabulary size"
        )
    tokenizer_sha256 = _sha256_file(tokenizer_path)

    signature_data = {
        "english_revision": data_manifest["english_revision"],
        "chinese_revision": data_manifest["chinese_revision"],
        "source_file_sha256": data_manifest.get("source_file_sha256", {}),
        "chinese_validation_fraction": data_manifest.get("chinese_validation_fraction"),
        "corpus_stats": stats,
        "tokenizer_path": tokenizer_relative_path.as_posix(),
        "tokenizer_sha256": tokenizer_sha256,
    }
    run_config: dict[str, Any] = {
        "corpus": CORPUS,
        "data_signature": _signature(signature_data),
        "source_file_sha256": data_manifest.get("source_file_sha256", {}),
        "tokenizer_path": tokenizer_relative_path.as_posix(),
        "tokenizer_sha256": tokenizer_sha256,
        "model": asdict(config),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "adam_betas": list(_ADAMW_BETAS),
        "adam_epsilon": _ADAMW_EPS,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "eval_batches": args.eval_batches,
        "eval_every_seconds": args.eval_every_seconds,
        "checkpoint_every_seconds": args.checkpoint_every_seconds,
        "log_every_steps": args.log_every_steps,
        "seed": args.seed,
        "threads": args.threads,
    }
    run_dir = args.run_dir or Path(
        f"artifacts/runs/{CORPUS}-{args.hours:g}h-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    metrics_path = run_dir / "metrics.jsonl"
    run_manifest_path = run_dir / "run_manifest.json"
    if not args.resume and (checkpoint_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"run directory already contains a run: {run_dir}; choose a new path or use --resume"
        )

    init_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    model = TinyGPT(config, generator=init_generator)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=_ADAMW_BETAS,
        eps=_ADAMW_EPS,
        weight_decay=args.weight_decay,
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    step = 0
    elapsed_before = 0.0
    tokens_seen = 0
    language_tokens_seen = {name: 0 for name, _ in train_lengths}
    best_validation_loss = math.inf
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"cannot resume without {checkpoint_path}")
        (
            step,
            elapsed_before,
            tokens_seen,
            language_tokens_seen,
            best_validation_loss,
        ) = _load_run_payload(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            batch_generator=batch_generator,
            run_config=run_config,
        )
        if set(language_tokens_seen) != {language for language, _ in train_lengths}:
            raise ValueError("checkpoint language counters do not match this corpus")
        if args.hours * 3600 <= elapsed_before:
            raise ValueError(
                "resume target must exceed the checkpoint's accumulated training time"
            )

    now_iso = datetime.now().astimezone().isoformat()
    previous_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if args.resume and run_manifest_path.exists()
        else {}
    )
    run_manifest = {
        "created_at": previous_manifest.get("created_at", now_iso),
        "updated_at": now_iso,
        "corpus": CORPUS,
        "target_hours": args.hours,
        "run_config": run_config,
        "train_tokens_available": train_stream.token_count,
        "training_segments": [asdict(segment) for segment in train_stream.segments],
        "validation_streams": [name for name, _ in validation_streams],
    }
    _write_json(run_manifest_path, run_manifest)
    _append_event(
        metrics_path,
        {
            "event": "started" if not args.resume else "resumed",
            "timestamp": now_iso,
            "target_hours": args.hours,
            "step": step,
            "elapsed_seconds_before_run": elapsed_before,
        },
    )

    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"received signal {signum}; will checkpoint after the current step",
            flush=True,
        )

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    run_started = time.monotonic()
    target_seconds = args.hours * 3600
    last_checkpoint = 0.0
    last_evaluation = -args.eval_every_seconds
    recent_losses: list[float] = []
    try:
        while not stop_requested:
            elapsed = elapsed_before + time.monotonic() - run_started
            if elapsed >= target_seconds:
                break

            loss, coverage, gradient_norm = train_step(
                model,
                optimizer,
                train_stream,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                generator=batch_generator,
                max_grad_norm=args.gradient_clip,
            )
            step += 1
            step_tokens = args.batch_size * args.sequence_length
            tokens_seen += step_tokens
            for language, count in coverage.items():
                language_tokens_seen[language] += count
            recent_losses.append(loss)

            if step % args.log_every_steps == 0:
                event = {
                    "event": "train",
                    "step": step,
                    "elapsed_seconds": elapsed_before + time.monotonic() - run_started,
                    "tokens_seen": tokens_seen,
                    "language_tokens_seen": language_tokens_seen.copy(),
                    "mean_loss": sum(recent_losses) / len(recent_losses),
                    "gradient_norm": gradient_norm,
                }
                _append_event(metrics_path, event)
                recent_losses.clear()

            elapsed = elapsed_before + time.monotonic() - run_started
            if elapsed - last_evaluation >= args.eval_every_seconds:
                validation = _evaluate_all(
                    model,
                    validation_streams,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    batches=args.eval_batches,
                    seed=args.seed + step,
                )
                primary_name = "mixed"
                primary_loss = validation[primary_name]["loss"]
                if primary_loss < best_validation_loss:
                    best_validation_loss = primary_loss
                    save_model_artifact(
                        run_dir / "best-model.pt",
                        model,
                        metadata={
                            "corpus": CORPUS,
                            "step": step,
                            "elapsed_seconds": elapsed,
                            "validation": validation,
                        },
                    )
                _append_event(
                    metrics_path,
                    {
                        "event": "validation",
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "tokens_seen": tokens_seen,
                        "language_tokens_seen": language_tokens_seen.copy(),
                        "metrics": validation,
                    },
                )
                last_evaluation = elapsed
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    elapsed_seconds=elapsed,
                    tokens_seen=tokens_seen,
                    language_tokens_seen=language_tokens_seen,
                    batch_generator=batch_generator,
                    run_config=run_config,
                    best_validation_loss=best_validation_loss,
                )
                last_checkpoint = elapsed

            if elapsed - last_checkpoint >= args.checkpoint_every_seconds:
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    elapsed_seconds=elapsed,
                    tokens_seen=tokens_seen,
                    language_tokens_seen=language_tokens_seen,
                    batch_generator=batch_generator,
                    run_config=run_config,
                    best_validation_loss=best_validation_loss,
                )
                last_checkpoint = elapsed
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    elapsed = elapsed_before + time.monotonic() - run_started
    final_validation = _evaluate_all(
        model,
        validation_streams,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        batches=max(args.eval_batches, 50),
        seed=args.seed + step + 1,
    )
    primary_name = "mixed"
    if final_validation[primary_name]["loss"] < best_validation_loss:
        best_validation_loss = final_validation[primary_name]["loss"]
        save_model_artifact(
            run_dir / "best-model.pt",
            model,
            metadata={
                "corpus": CORPUS,
                "step": step,
                "elapsed_seconds": elapsed,
                "validation": final_validation,
            },
        )
    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=step,
        elapsed_seconds=elapsed,
        tokens_seen=tokens_seen,
        language_tokens_seen=language_tokens_seen,
        batch_generator=batch_generator,
        run_config=run_config,
        best_validation_loss=best_validation_loss,
    )
    final_model_path = run_dir / f"model-{args.hours:g}h.pt"
    final_metadata = {
        "corpus": CORPUS,
        "step": step,
        "training_seconds": elapsed,
        "tokens_seen": tokens_seen,
        "language_tokens_seen": language_tokens_seen,
        "train_tokens_available": train_stream.token_count,
        "validation": final_validation,
        "run_config": run_config,
    }
    save_model_artifact(final_model_path, model, metadata=final_metadata)
    _write_json(
        run_dir / f"samples-{args.hours:g}h.json",
        {
            "model": final_model_path.name,
            "tokenizer": str(tokenizer_relative_path),
            "samples": _final_samples(model, tokenizer, args.seed),
        },
    )
    _append_event(
        metrics_path,
        {
            "event": "finished" if not stop_requested else "stopped",
            "step": step,
            "training_seconds": elapsed,
            "tokens_seen": tokens_seen,
            "language_tokens_seen": language_tokens_seen,
            "validation": final_validation,
            "model": final_model_path.name,
        },
    )
    _write_json(
        run_manifest_path,
        {
            **run_manifest,
            "updated_at": datetime.now().astimezone().isoformat(),
            "training_seconds": elapsed,
            "steps": step,
            "tokens_seen": tokens_seen,
            "language_tokens_seen": language_tokens_seen,
            "final_validation": final_validation,
            "model_artifact": final_model_path.name,
        },
    )
    print(
        f"Finished {CORPUS}: {step:,} steps, {tokens_seen:,} tokens, "
        f"{elapsed / 3600:.3f} training hours; artifacts: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
