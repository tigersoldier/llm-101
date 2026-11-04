import pytest
import torch

from llm101.inference import generate
from llm101.model import GPTConfig, TinyGPT
from llm101.tokenizer import BPETokenizer


def _tokenizer_and_model() -> tuple[BPETokenizer, TinyGPT]:
    tokenizer = BPETokenizer.train(
        ["hello small world", "hello little story"], vocab_size=32
    )
    model = TinyGPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_seq_len=8,
        ),
        generator=torch.Generator().manual_seed(5),
    )
    return tokenizer, model


def test_greedy_generation_is_deterministic_and_keeps_prompt() -> None:
    tokenizer, model = _tokenizer_and_model()

    first = generate(model, tokenizer, "h", max_new_tokens=5, temperature=0)
    second = generate(model, tokenizer, "h", max_new_tokens=5, temperature=0)

    assert first == second
    assert first.startswith("h")


def test_top_k_sampling_is_seeded() -> None:
    tokenizer, model = _tokenizer_and_model()

    first = generate(model, tokenizer, "hello", max_new_tokens=6, top_k=4, seed=12)
    second = generate(model, tokenizer, "hello", max_new_tokens=6, top_k=4, seed=12)

    assert first == second
    assert first.startswith("hello")


def test_generation_rejects_incompatible_vocabulary_size() -> None:
    tokenizer, model = _tokenizer_and_model()
    model.config = GPTConfig(vocab_size=tokenizer.vocab_size + 1)

    with pytest.raises(ValueError, match="vocabulary sizes"):
        generate(model, tokenizer, "hello")
