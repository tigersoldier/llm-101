"""Print the tokenizer pipeline state after a given section.

Section 2.1 implements stage 1 only. Later sections add the BPE stages.
"""

from __future__ import annotations

import argparse

from llm101.baseline import (
    CHINESE_SAMPLE,
    ENGLISH_SAMPLE,
    BaselineReport,
    build_report,
    context_error,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    if args.stage != 1:
        parser.error("only --stage 1 is implemented in section 2.1")

    print("stage 1 | character baseline + hand merge")
    for text in (ENGLISH_SAMPLE, CHINESE_SAMPLE):
        print()
        _print_report(build_report(text))


if __name__ == "__main__":
    main()
