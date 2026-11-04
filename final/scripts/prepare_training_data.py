"""Build tokenizers and full-corpus int32 streams for all three experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from llm101.corpus import (
    collect_alphabet_and_sample,
    concatenate_token_streams,
    iter_delimited_stories,
    iter_jsonl_stories,
    split_chinese_archive,
    write_token_stream,
)
from llm101.tokenizer import BPETokenizer

ENGLISH_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
CHINESE_REVISION = "96e998aab7ce3311f8a8497b2aa6ed5088630672"
CHINESE_ARCHIVE_SHA256 = (
    "d7ae2d3e99a67c2c26f735c6e2d53d30b141febbabfe1b689b2e89a6cbce593c"
)
ENGLISH_FILES = {
    "train": "TinyStories-train.txt",
    "validation": "TinyStories-valid.txt",
}
ENGLISH_SHA256 = {
    "train": "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
    "validation": "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
}


def with_story_progress(
    stories: Iterable[str], label: str, *, interval: int = 100_000
) -> Iterator[str]:
    for count, story in enumerate(stories, start=1):
        if count % interval == 0:
            print(f"  {label}: {count:,} stories", flush=True)
        yield story


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def ensure_chinese_split(
    archive_path: Path,
    output_dir: Path,
    *,
    validation_fraction: float,
    force: bool,
    progress: Callable[[int], None] | None = None,
) -> tuple[Path, Path, int, int]:
    split_dir = output_dir / "chinese" / "stories"
    train_path = split_dir / "train.jsonl"
    validation_path = split_dir / "validation.jsonl"
    manifest_path = split_dir / "split.json"
    archive_hash = sha256_file(archive_path)
    if archive_hash != CHINESE_ARCHIVE_SHA256:
        raise ValueError(f"Chinese source archive SHA-256 mismatch: {archive_hash}")

    if (
        not force
        and manifest_path.exists()
        and train_path.exists()
        and validation_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("archive_sha256") == archive_hash
            and manifest.get("validation_fraction") == validation_fraction
        ):
            print("Using cached deterministic Chinese story split.", flush=True)
            return (
                train_path,
                validation_path,
                int(manifest["train_stories"]),
                int(manifest["validation_stories"]),
            )

    print("Splitting Chinese story archive by content hash...", flush=True)
    train_count, validation_count = split_chinese_archive(
        archive_path,
        train_path,
        validation_path,
        validation_fraction=validation_fraction,
        progress=progress,
    )
    write_json(
        manifest_path,
        {
            "dataset": "52AI/TinyStoriesZh",
            "revision": CHINESE_REVISION,
            "archive_sha256": archive_hash,
            "split_method": "blake2b-64 of normalized story text",
            "validation_fraction": validation_fraction,
            "train_stories": train_count,
            "validation_stories": validation_count,
        },
    )
    return train_path, validation_path, train_count, validation_count


def train_tokenizer(
    sample: list[str],
    alphabet: set[str],
    target_vocab_size: int,
) -> BPETokenizer:
    minimum_size = len(alphabet) + 4
    actual_target = max(target_vocab_size, minimum_size)
    return BPETokenizer.train(
        sample,
        actual_target,
        min_frequency=2,
        initial_alphabet=alphabet,
    )


def encode_split(
    tokenizer: BPETokenizer,
    train_stories,
    validation_stories,
    output_dir: Path,
    *,
    workers: int,
) -> dict[str, int]:
    train_tokens, train_story_count = write_token_stream(
        train_stories, tokenizer, output_dir / "train.bin", workers=workers
    )
    validation_tokens, validation_story_count = write_token_stream(
        validation_stories, tokenizer, output_dir / "validation.bin", workers=workers
    )
    return {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_stories": train_story_count,
        "validation_stories": validation_story_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-stories", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--english-vocab-size", type=int, default=4096)
    parser.add_argument("--chinese-vocab-size", type=int, default=8192)
    parser.add_argument("--bilingual-vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--reuse-tokenizers", action="store_true")
    parser.add_argument("--force-rebuild-splits", action="store_true")
    args = parser.parse_args()
    if args.sample_stories <= 0:
        parser.error("--sample-stories must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    print("Verifying pinned source file checksums...", flush=True)
    english_train = args.raw_dir / "english" / ENGLISH_FILES["train"]
    english_valid = args.raw_dir / "english" / ENGLISH_FILES["validation"]
    chinese_archive = args.raw_dir / "chinese" / "TinyStories_all_data_zh_2M.tar.gz"
    for path in (english_train, english_valid, chinese_archive):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; first run uv run python scripts/download_datasets.py"
            )
    english_hashes = {
        "train": sha256_file(english_train),
        "validation": sha256_file(english_valid),
    }
    for split, expected in ENGLISH_SHA256.items():
        if english_hashes[split] != expected:
            raise ValueError(
                f"English {split} SHA-256 mismatch: {english_hashes[split]}"
            )

    chinese_train, chinese_valid, chinese_train_count, chinese_valid_count = (
        ensure_chinese_split(
            chinese_archive,
            args.output_dir,
            validation_fraction=args.validation_fraction,
            force=args.force_rebuild_splits,
            progress=lambda count: print(
                f"  split {count:,} Chinese stories", flush=True
            ),
        )
    )

    tokenizer_dir = args.output_dir / "tokenizers"
    tokenizer_paths = {
        "english": tokenizer_dir / "english.json",
        "chinese": tokenizer_dir / "chinese.json",
        "bilingual": tokenizer_dir / "bilingual.json",
    }
    if args.reuse_tokenizers:
        missing = [path for path in tokenizer_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cannot reuse missing tokenizers: {missing}")
        english_tokenizer = BPETokenizer.load(tokenizer_paths["english"])
        chinese_tokenizer = BPETokenizer.load(tokenizer_paths["chinese"])
        bilingual_tokenizer = BPETokenizer.load(tokenizer_paths["bilingual"])
        english_alphabet = set(english_tokenizer.initial_tokens)
        chinese_alphabet = set(chinese_tokenizer.initial_tokens)
        joint_alphabet = set(bilingual_tokenizer.initial_tokens)
        tokenizer_training_path = tokenizer_dir / "manifest.json"
        tokenizer_training = (
            json.loads(tokenizer_training_path.read_text(encoding="utf-8"))
            if tokenizer_training_path.exists()
            else {}
        )
        tokenizer_training["mode"] = "reused-existing-tokenizers"
        print("Reusing the saved tokenizers and resuming corpus encoding...")
    else:
        print("Scanning full training splits for code points and tokenizer samples...")
        english_alphabet, english_sample, english_story_count = (
            collect_alphabet_and_sample(
                with_story_progress(
                    iter_delimited_stories(english_train), "English alphabet scan"
                ),
                sample_size=args.sample_stories,
                seed=args.seed,
            )
        )
        chinese_alphabet, chinese_sample, scanned_chinese_count = (
            collect_alphabet_and_sample(
                with_story_progress(
                    iter_jsonl_stories(chinese_train), "Chinese alphabet scan"
                ),
                sample_size=args.sample_stories,
                seed=args.seed + 1,
            )
        )
        if scanned_chinese_count != chinese_train_count:
            raise RuntimeError("Chinese training story count changed after the split")

        joint_alphabet = english_alphabet | chinese_alphabet
        joint_sample = english_sample + chinese_sample
        print(
            "story counts: "
            f"English train={english_story_count:,}; "
            f"Chinese train={chinese_train_count:,}, validation={chinese_valid_count:,}"
        )
        print("Training English BPE merges...", flush=True)
        english_tokenizer = train_tokenizer(
            english_sample, english_alphabet, args.english_vocab_size
        )
        print("Training Chinese BPE merges...", flush=True)
        chinese_tokenizer = train_tokenizer(
            chinese_sample, chinese_alphabet, args.chinese_vocab_size
        )
        print("Training bilingual BPE merges...", flush=True)
        bilingual_tokenizer = train_tokenizer(
            joint_sample, joint_alphabet, args.bilingual_vocab_size
        )
        english_tokenizer.save(tokenizer_paths["english"])
        chinese_tokenizer.save(tokenizer_paths["chinese"])
        bilingual_tokenizer.save(tokenizer_paths["bilingual"])
        tokenizer_training = {
            "mode": "trained-from-reservoir-samples",
            "sample_stories_per_language": args.sample_stories,
            "seed": args.seed,
            "min_frequency": 2,
            "target_vocab_sizes": {
                "english": args.english_vocab_size,
                "chinese": args.chinese_vocab_size,
                "bilingual": args.bilingual_vocab_size,
            },
        }

    tokenizer_training["actual_vocab_sizes"] = {
        "english": english_tokenizer.vocab_size,
        "chinese": chinese_tokenizer.vocab_size,
        "bilingual": bilingual_tokenizer.vocab_size,
    }
    write_json(tokenizer_dir / "manifest.json", tokenizer_training)
    print(
        "alphabet sizes: "
        f"English={len(english_alphabet):,}; "
        f"Chinese={len(chinese_alphabet):,}; bilingual={len(joint_alphabet):,}"
    )
    print(
        "vocabulary sizes: "
        f"English={english_tokenizer.vocab_size:,}; "
        f"Chinese={chinese_tokenizer.vocab_size:,}; "
        f"bilingual={bilingual_tokenizer.vocab_size:,}"
    )

    print("Encoding all training and validation stories...")
    english_stats = encode_split(
        english_tokenizer,
        iter_delimited_stories(english_train),
        iter_delimited_stories(english_valid),
        args.output_dir / "english",
        workers=args.workers,
    )
    chinese_stats = encode_split(
        chinese_tokenizer,
        iter_jsonl_stories(chinese_train),
        iter_jsonl_stories(chinese_valid),
        args.output_dir / "chinese",
        workers=args.workers,
    )
    if (
        chinese_stats["train_stories"] != chinese_train_count
        or chinese_stats["validation_stories"] != chinese_valid_count
    ):
        raise RuntimeError(
            "Chinese token streams do not match the persisted story split"
        )

    bilingual_dir = args.output_dir / "bilingual"
    bilingual_dir.mkdir(parents=True, exist_ok=True)
    bilingual_parts = {
        "english_train": (iter_delimited_stories(english_train), "english_train.bin"),
        "chinese_train": (iter_jsonl_stories(chinese_train), "chinese_train.bin"),
        "english_validation": (
            iter_delimited_stories(english_valid),
            "english_validation.bin",
        ),
        "chinese_validation": (
            iter_jsonl_stories(chinese_valid),
            "chinese_validation.bin",
        ),
    }
    bilingual_stats: dict[str, int] = {"vocab_size": bilingual_tokenizer.vocab_size}
    for name, (stories, filename) in bilingual_parts.items():
        count, story_count = write_token_stream(
            stories, bilingual_tokenizer, bilingual_dir / filename, workers=args.workers
        )
        bilingual_stats[f"{name}_tokens"] = count
        bilingual_stats[f"{name}_stories"] = story_count

    bilingual_stats["train_tokens"] = concatenate_token_streams(
        [bilingual_dir / "english_train.bin", bilingual_dir / "chinese_train.bin"],
        bilingual_dir / "train.bin",
    )
    bilingual_stats["validation_tokens"] = concatenate_token_streams(
        [
            bilingual_dir / "english_validation.bin",
            bilingual_dir / "chinese_validation.bin",
        ],
        bilingual_dir / "validation.bin",
    )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "tokenizer_training": tokenizer_training,
        "english_revision": ENGLISH_REVISION,
        "chinese_revision": CHINESE_REVISION,
        "source_file_sha256": {
            "english_train": english_hashes["train"],
            "english_validation": english_hashes["validation"],
            "chinese_archive": CHINESE_ARCHIVE_SHA256,
        },
        "chinese_validation_fraction": args.validation_fraction,
        "alphabet_sizes": {
            "english": len(english_alphabet),
            "chinese": len(chinese_alphabet),
            "bilingual": len(joint_alphabet),
        },
        "corpora": {
            "english": english_stats,
            "chinese": chinese_stats,
            "bilingual": bilingual_stats,
        },
        "tokenizer_files": {
            "english": "tokenizers/english.json",
            "chinese": "tokenizers/chinese.json",
            "bilingual": "tokenizers/bilingual.json",
        },
        "tokenizer_sha256": {
            name: sha256_file(path) for name, path in tokenizer_paths.items()
        },
    }
    write_json(args.output_dir / "experiment_manifest.json", manifest)
    print(json.dumps(manifest["corpora"], ensure_ascii=False, indent=2))
    print(f"Prepared data at {args.output_dir}")


if __name__ == "__main__":
    main()
