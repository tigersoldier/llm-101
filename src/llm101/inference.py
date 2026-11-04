"""Autoregressive decoding for the educational TinyGPT model."""

from __future__ import annotations

import math

import torch

from llm101.model import TinyGPT
from llm101.tokenizer import BPETokenizer


def generate(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
) -> str:
    """Generate text with top-k sampling; set temperature to zero for greedy."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes do not match")

    token_ids = tokenizer.encode(prompt, add_bos=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = token_ids[-model.config.max_seq_len :]
            input_ids = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            logits = model.forward(input_ids)[0, -1].clone()
            if temperature == 0:
                next_token = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature
                if top_k:
                    count = min(top_k, logits.numel())
                    threshold = torch.topk(logits, count).values[-1]
                    logits[logits < threshold] = -torch.inf
                probabilities = torch.softmax(logits, dim=-1)
                next_token = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
            token_ids.append(next_token)
            if next_token == tokenizer.eos_id:
                break

    return tokenizer.decode(token_ids)
