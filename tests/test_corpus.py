import io
import json
import tarfile
from collections import Counter

import pytest
import torch

from llm101.corpus import (
    collect_alphabet_and_sample,
    concatenate_token_streams,
    iter_delimited_stories,
    iter_jsonl_stories,
    split_chinese_archive,
    write_token_stream,
)
from llm101.tokenizer import BPETokenizer


def test_delimited_story_reader_handles_marker_across_chunks(tmp_path) -> None:
    path = tmp_path / "stories.txt"
    path.write_text("\nfirst<|endoftext|> second <|endoftext|>\nlast", encoding="utf-8")

    stories = list(iter_delimited_stories(path, chunk_size=7))

    assert stories == ["first", "second", "last"]


def test_chinese_split_is_deterministic_and_keeps_duplicates_together(tmp_path) -> None:
    archive_path = tmp_path / "zh.tar.gz"
    records = [{"story": f"故事 {index}"} for index in range(200)]
    records.extend([{"story": "重复故事"}, {"story": "重复故事"}])
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("data11.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    first_train = tmp_path / "first-train.jsonl"
    first_valid = tmp_path / "first-valid.jsonl"
    second_train = tmp_path / "second-train.jsonl"
    second_valid = tmp_path / "second-valid.jsonl"
    progress_counts: list[int] = []
    first_counts = split_chinese_archive(
        archive_path,
        first_train,
        first_valid,
        validation_fraction=0.2,
        progress=progress_counts.append,
        progress_every_stories=50,
    )
    second_counts = split_chinese_archive(
        archive_path, second_train, second_valid, validation_fraction=0.2
    )

    train_stories = list(iter_jsonl_stories(first_train))
    valid_stories = list(iter_jsonl_stories(first_valid))
    assert first_counts == second_counts
    assert progress_counts == [50, 100, 150, 200, 202]
    assert first_train.read_bytes() == second_train.read_bytes()
    assert first_valid.read_bytes() == second_valid.read_bytes()
    assert len(train_stories) == first_counts[0]
    assert len(valid_stories) == first_counts[1]
    assert not set(train_stories) & set(valid_stories)
    assert Counter(train_stories + valid_stories)["重复故事"] == 2


def test_chinese_split_rejects_archive_member_paths(tmp_path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    payload = b"[]"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("../escape.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    with pytest.raises(ValueError, match="unexpected path"):
        split_chinese_archive(archive_path, train_path, validation_path)

    assert not train_path.exists()
    assert not validation_path.exists()
    assert not (tmp_path.parent / "escape.json").exists()


def test_sample_scan_keeps_full_alphabet_and_is_reproducible() -> None:
    stories = [f"story-{index}-中" for index in range(50)]

    first = collect_alphabet_and_sample(stories, sample_size=5, seed=17)
    second = collect_alphabet_and_sample(stories, sample_size=5, seed=17)

    assert first == second
    assert first[2] == len(stories)
    assert "中" in first[0]
    assert set().union(*(set(story) for story in stories)) == first[0]
    assert len(first[1]) == 5


def test_token_streams_are_int32_and_concatenate_without_changing_ids(tmp_path) -> None:
    tokenizer = BPETokenizer.train(["hello world", "hello there"], vocab_size=32)
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    combined_path = tmp_path / "combined.bin"
    first_count, _ = write_token_stream(["hello world"], tokenizer, first_path)
    second_count, _ = write_token_stream(["hello there"], tokenizer, second_path)

    combined_count = concatenate_token_streams([first_path, second_path], combined_path)
    combined = torch.from_file(
        str(combined_path), shared=False, size=combined_count, dtype=torch.int32
    )

    assert combined_count == first_count + second_count
    assert combined.dtype == torch.int32
    assert combined[0].item() == tokenizer.bos_id
    assert combined[first_count - 1].item() == tokenizer.eos_id
    assert combined[first_count].item() == tokenizer.bos_id
    assert combined[-1].item() == tokenizer.eos_id


def test_parallel_token_stream_matches_single_process(tmp_path) -> None:
    stories = [f"story number {index}" for index in range(12)]
    tokenizer = BPETokenizer.train(stories, vocab_size=40)
    sequential_path = tmp_path / "sequential.bin"
    parallel_path = tmp_path / "parallel.bin"

    sequential_result = write_token_stream(stories, tokenizer, sequential_path)
    parallel_result = write_token_stream(
        stories, tokenizer, parallel_path, workers=2, batch_size=3
    )

    assert parallel_result == sequential_result
    assert parallel_path.read_bytes() == sequential_path.read_bytes()
