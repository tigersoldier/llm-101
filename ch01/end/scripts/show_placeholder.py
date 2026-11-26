"""Print the placeholder pipeline state after a given section.

Example:
    uv run python scripts/show_placeholder.py "the cat sat"
"""

from __future__ import annotations

import argparse

from llm101.placeholder import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="the cat sat")
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    result = run(args.text, stage=args.stage)

    print(f"stage {result.stage} | text: {result.text}")
    print(f"ids ({len(result.ids)})        {result.ids}")
    if result.vectors is not None:
        print(f"vectors[-1]     {[round(value, 3) for value in result.vectors[0, -1].tolist()]}")
    if result.answer is not None:
        print(f"answer          {result.answer}")


if __name__ == "__main__":
    main()
