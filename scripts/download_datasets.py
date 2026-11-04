"""Download the pinned TinyStories corpora and verify their SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DatasetFile:
    dataset: str
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str


FILES = (
    DatasetFile(
        dataset="english",
        repository="roneneldan/TinyStories",
        revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        filename="TinyStories-train.txt",
        size_bytes=1_924_281_556,
        sha256="c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
    ),
    DatasetFile(
        dataset="english",
        repository="roneneldan/TinyStories",
        revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        filename="TinyStories-valid.txt",
        size_bytes=19_447_282,
        sha256="94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
    ),
    DatasetFile(
        dataset="chinese",
        repository="52AI/TinyStoriesZh",
        revision="96e998aab7ce3311f8a8497b2aa6ed5088630672",
        filename="TinyStories_all_data_zh_2M.tar.gz",
        size_bytes=509_820_547,
        sha256="d7ae2d3e99a67c2c26f735c6e2d53d30b141febbabfe1b689b2e89a6cbce593c",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(spec: DatasetFile, output_dir: Path) -> Path:
    target_dir = output_dir / spec.dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / spec.filename

    if target.exists():
        if (
            target.stat().st_size == spec.size_bytes
            and sha256_file(target) == spec.sha256
        ):
            print(f"Verified existing file: {target}")
            return target
        print(f"Removing file with unexpected size or digest: {target}")
        target.unlink()

    url = (
        f"https://huggingface.co/datasets/{spec.repository}/resolve/"
        f"{spec.revision}/{spec.filename}?download=true"
    )
    parsed_url = urlsplit(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "huggingface.co":
        raise ValueError(f"refusing unexpected dataset URL: {url!r}")
    partial = target.with_name(target.name + ".partial")
    print(f"Downloading {spec.repository}/{spec.filename} ({spec.size_bytes:,} bytes)")
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "60",
                "--progress-bar",
                "--output",
                str(partial),
                parsed_url.geturl(),
            ],
            check=True,
        )
        received = partial.stat().st_size
        if received != spec.size_bytes:
            raise RuntimeError(
                f"size mismatch for {target}: expected {spec.size_bytes}, got {received}"
            )
        actual_digest = sha256_file(partial)
        if actual_digest != spec.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {target}: expected {spec.sha256}, "
                f"got {actual_digest}"
            )
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"Verified {target} ({actual_digest})")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("english", "chinese", "all"),
        default="all",
        help="which corpus to download (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    for spec in FILES:
        if args.dataset != "all" and args.dataset != spec.dataset:
            continue
        download_file(spec, args.output_dir)


if __name__ == "__main__":
    main()
