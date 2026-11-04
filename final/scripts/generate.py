"""Load a saved TinyGPT artifact and continue a text prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm101.checkpoint import load_model_artifact
from llm101.inference import generate
from llm101.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, _ = load_model_artifact(args.model)
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
