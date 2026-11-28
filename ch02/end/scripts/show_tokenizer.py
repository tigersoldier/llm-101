"""Print the tokenizer pipeline state after a given section.

Stages 1–3 are implemented through section 2.3; later sections add stages 4–6.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from llm101.baseline import (
    CHINESE_SAMPLE,
    ENGLISH_SAMPLE,
    BaselineReport,
    build_report,
    context_error,
)

from llm101.tokenizer import BPETokenizer, Pair, learn_merges, merge_pair

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
DEMO_TEXTS: tuple[str, ...] = (
    "the cat sat on the mat",
    "the cat sat on the mat",
    "the cat sat on the mat",
    "the cat sat on a mat",
    "the cat is on the mat",
    "小猫坐在垫子上",
    "小猫坐在垫子上",
    "小猫坐在垫子上",
    "小猫坐在垫子旁边",
    "小猫坐在垫子上玩",
)


def _line(label: str, value: object) -> None:
    print(f"{label:<16}{value}")


def _print_report(report: BaselineReport) -> None:
    error = context_error(report.text)
    _line("text", report.text)
    _line("chars", report.char_count)
    _line(f"ids ({report.token_count})", report.ids)
    _line("unknown", report.unknown_count)
    _line("decoded", repr(report.decoded))
    _line(
        "hand merge",
        f"{len(report.hand_symbols)} symbols; + specials = {report.hand_token_count}",
    )
    _line("symbols", report.hand_symbols)
    _line("model", "-" if error is None else f"ValueError: {error}")


def _apply_merges(text: str, merges: Sequence[Pair]) -> list[str]:
    symbols = list(text)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols


def _print_stage_two() -> None:
    merges = learn_merges(DEMO_TEXTS, 16, min_frequency=2)
    print(f"stage 2 | adjacent-pair merges ({len(merges)})")
    _line("corpus", f"{len(DEMO_TEXTS)} short lines")
    for rank, pair in enumerate(merges, start=1):
        print(f"{rank:>2}. {pair[0]!r} + {pair[1]!r} -> {pair[0] + pair[1]!r}")

    for text in (ENGLISH_SAMPLE, CHINESE_SAMPLE):
        symbols = _apply_merges(text, merges)
        print()
        _line("text", text)
        _line(f"symbols ({len(symbols)})", symbols)


def _load_training_texts() -> list[str]:
    texts: list[str] = []
    for name in ("english.txt", "chinese.txt"):
        lines = (SAMPLES_DIR / name).read_text(encoding="utf-8").splitlines()
        texts.extend(lines)
    return texts


def _print_stage_three() -> None:
    texts = _load_training_texts()
    tokenizer = BPETokenizer.train(texts, 512, min_frequency=2)
    print("stage 3 | vocabulary")
    _line("corpus", f"{len(texts)} lines")
    _line("vocab size", tokenizer.vocab_size)
    _line("alphabet", len(tokenizer.initial_tokens))
    _line("merges", len(tokenizer.merges))
    _line("specials", tokenizer.special_tokens)

    print("first merges:")
    for rank, pair in enumerate(tokenizer.merges[:10], start=1):
        print(f"{rank:>2}. {pair[0]!r} + {pair[1]!r} -> {pair[0] + pair[1]!r}")

    print("first vocabulary:")
    for token_id, token in enumerate(tokenizer.vocabulary[:10]):
        print(f"{token_id:>2}. {token!r}")

    token_to_id = {
        token: token_id for token_id, token in enumerate(tokenizer.vocabulary)
    }
    for text in (ENGLISH_SAMPLE, CHINESE_SAMPLE):
        symbols = _apply_merges(text, tokenizer.merges)
        ids = [token_to_id.get(piece, tokenizer.unk_id) for piece in symbols]
        print()
        _line("text", text)
        _line(f"symbols ({len(symbols)})", symbols)
        _line(f"ids ({len(ids)})", ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    if args.stage not in (1, 2, 3):
        parser.error("only --stage 1 to 3 are implemented through section 2.3")

    if args.stage == 3:
        _print_stage_three()
        return
    if args.stage == 2:
        _print_stage_two()
        return

    print("stage 1 | character baseline + hand merge")
    for text in (ENGLISH_SAMPLE, CHINESE_SAMPLE):
        print()
        _print_report(build_report(text))


if __name__ == "__main__":
    main()
