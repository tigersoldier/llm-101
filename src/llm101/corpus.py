"""Streaming helpers for the pinned TinyStories training corpora."""

from __future__ import annotations

import array
import hashlib
import json
import random
import re
import sys
import tarfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TextIO

from llm101.tokenizer import BPETokenizer

_END_OF_TEXT = "<|endoftext|>"
_TOKEN_ITEM_SIZE = array.array("I").itemsize
_WORKER_TOKENIZER: BPETokenizer | None = None


def _initialize_encoder(tokenizer: BPETokenizer) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_story_batch(stories: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return [
        _WORKER_TOKENIZER.encode(story, add_bos=True, add_eos=True) for story in stories
    ]


def _story_batches(stories: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for story in stories:
        batch.append(story)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ordered_parallel_batches(
    executor: ProcessPoolExecutor,
    batches: Iterable[list[str]],
    *,
    max_in_flight: int,
) -> Iterator[list[list[int]]]:
    pending: deque[Future[list[list[int]]]] = deque()
    for batch in batches:
        pending.append(executor.submit(_encode_story_batch, batch))
        if len(pending) >= max_in_flight:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def iter_delimited_stories(
    path: str | Path,
    *,
    delimiter: str = _END_OF_TEXT,
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    """Yield stories from a large UTF-8 file without reading it all at once."""
    if not delimiter:
        raise ValueError("delimiter cannot be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pending = ""
    with Path(path).open("r", encoding="utf-8") as stream:
        while chunk := stream.read(chunk_size):
            pieces = (pending + chunk).split(delimiter)
            pending = pieces.pop()
            for story in pieces:
                story = story.strip()
                if story:
                    yield story

    pending = pending.strip()
    if pending:
        yield pending


def split_chinese_archive(
    archive_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    *,
    validation_fraction: float = 0.01,
    progress: Callable[[int], None] | None = None,
    progress_every_stories: int = 100_000,
) -> tuple[int, int]:
    """Split the 52AI JSON-array shards deterministically by story content.

    Hash-based assignment keeps duplicate stories in the same split and spreads
    validation examples throughout the source shards. The source archive is read
    as a stream; it is never extracted to paths from the archive header.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if progress_every_stories <= 0:
        raise ValueError("progress interval must be positive")

    train_target = Path(train_path)
    validation_target = Path(validation_path)
    train_target.parent.mkdir(parents=True, exist_ok=True)
    validation_target.parent.mkdir(parents=True, exist_ok=True)
    train_partial = train_target.with_name(train_target.name + ".partial")
    validation_partial = validation_target.with_name(
        validation_target.name + ".partial"
    )
    train_count = 0
    validation_count = 0
    processed_count = 0
    threshold = int(validation_fraction * (1 << 64))

    try:
        with (
            train_partial.open("w", encoding="utf-8", newline="\n") as train_out,
            validation_partial.open(
                "w", encoding="utf-8", newline="\n"
            ) as validation_out,
            tarfile.open(archive_path, mode="r|gz") as archive,
        ):
            for member in archive:
                if not member.isfile():
                    raise ValueError(
                        f"unexpected non-file in Chinese archive: {member.name}"
                    )
                if re.fullmatch(r"data\d+\.json", member.name) is None:
                    raise ValueError(
                        f"unexpected path in Chinese archive: {member.name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"unable to read archive member: {member.name}")

                with source:
                    records = json.load(source)
                if not isinstance(records, list):
                    raise TypeError(f"expected a JSON array in {member.name}")
                for record in records:
                    if not isinstance(record, dict) or not isinstance(
                        record.get("story"), str
                    ):
                        raise TypeError(f"invalid story record in {member.name}")
                    story = record["story"].strip()
                    if not story:
                        continue
                    processed_count += 1
                    if (
                        progress is not None
                        and processed_count % progress_every_stories == 0
                    ):
                        progress(processed_count)
                    digest = hashlib.blake2b(
                        story.encode("utf-8"), digest_size=8
                    ).digest()
                    is_validation = int.from_bytes(digest, "big") < threshold
                    output: TextIO = validation_out if is_validation else train_out
                    output.write(json.dumps(story, ensure_ascii=False) + "\n")
                    if is_validation:
                        validation_count += 1
                    else:
                        train_count += 1

        if train_count == 0 or validation_count == 0:
            raise ValueError(
                "Chinese split produced an empty training or validation set"
            )
        if progress is not None and processed_count % progress_every_stories:
            progress(processed_count)
        train_partial.replace(train_target)
        validation_partial.replace(validation_target)
    except BaseException:
        train_partial.unlink(missing_ok=True)
        validation_partial.unlink(missing_ok=True)
        raise

    return train_count, validation_count


def iter_jsonl_stories(path: str | Path) -> Iterator[str]:
    """Yield UTF-8 JSON-string records written by :func:`split_chinese_archive`."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            story = json.loads(line)
            if not isinstance(story, str):
                raise TypeError(
                    f"expected a JSON string on line {line_number} of {path}"
                )
            if story:
                yield story


def collect_alphabet_and_sample(
    stories: Iterable[str], *, sample_size: int, seed: int
) -> tuple[set[str], list[str], int]:
    """Scan a split for its full code-point alphabet and a reservoir sample."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    alphabet: set[str] = set()
    sample: list[str] = []
    rng = random.Random(seed)
    story_count = 0
    for story_count, story in enumerate(stories, start=1):
        alphabet.update(story)
        if len(sample) < sample_size:
            sample.append(story)
        else:
            replacement = rng.randrange(story_count)
            if replacement < sample_size:
                sample[replacement] = story
    return alphabet, sample, story_count


def write_token_stream(
    stories: Iterable[str],
    tokenizer: BPETokenizer,
    path: str | Path,
    *,
    flush_tokens: int = 262_144,
    workers: int = 1,
    batch_size: int = 128,
    progress_every_stories: int = 100_000,
) -> tuple[int, int]:
    """Encode stories to little-endian int32, optionally using CPU worker processes."""
    if _TOKEN_ITEM_SIZE != 4 or sys.byteorder != "little":
        raise RuntimeError("this binary format requires little-endian 32-bit integers")
    if flush_tokens <= 0 or batch_size <= 0 or progress_every_stories < 0:
        raise ValueError("flush and batch sizes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    buffer = array.array("I")
    token_count = 0
    story_count = 0
    batches = _story_batches(stories, batch_size)

    try:
        with partial.open("wb") as output:
            if workers == 1:
                encoded_batches = (
                    [
                        tokenizer.encode(story, add_bos=True, add_eos=True)
                        for story in batch
                    ]
                    for batch in batches
                )
                for encoded_batch in encoded_batches:
                    for ids in encoded_batch:
                        buffer.extend(ids)
                        token_count += len(ids)
                        story_count += 1
                    if len(buffer) >= flush_tokens:
                        buffer.tofile(output)
                        buffer = array.array("I")
                    _report_encoding_progress(story_count, progress_every_stories)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_encoder,
                    initargs=(tokenizer,),
                ) as executor:
                    for encoded_batch in _ordered_parallel_batches(
                        executor, batches, max_in_flight=workers * 2
                    ):
                        for ids in encoded_batch:
                            buffer.extend(ids)
                            token_count += len(ids)
                            story_count += 1
                        if len(buffer) >= flush_tokens:
                            buffer.tofile(output)
                            buffer = array.array("I")
                        _report_encoding_progress(story_count, progress_every_stories)
            if buffer:
                buffer.tofile(output)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return token_count, story_count


def _report_encoding_progress(story_count: int, interval: int) -> None:
    if interval and story_count and story_count % interval < 128:
        print(f"  encoded {story_count:,} stories", flush=True)


def concatenate_token_streams(sources: Iterable[str | Path], target: str | Path) -> int:
    """Concatenate int32 token streams and return their total token count."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    total_bytes = 0
    try:
        with partial.open("wb") as output:
            for source_path in sources:
                source_path = Path(source_path)
                size = source_path.stat().st_size
                if size % _TOKEN_ITEM_SIZE:
                    raise ValueError(
                        f"token stream has a partial integer: {source_path}"
                    )
                with source_path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                total_bytes += size
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return total_bytes // _TOKEN_ITEM_SIZE
