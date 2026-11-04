import array
import math

import pytest
import torch

from llm101.model import GPTConfig, TinyGPT
from llm101.optim import AdamW
from llm101.training import TokenStream, clip_grad_norm_, train_step


def _write_ids(path, ids: list[int]) -> None:
    values = array.array("I", ids)
    with path.open("wb") as stream:
        values.tofile(stream)


def test_token_stream_batches_shift_targets_and_count_language_coverage(
    tmp_path,
) -> None:
    path = tmp_path / "tokens.bin"
    _write_ids(path, list(range(100)))
    stream = TokenStream(path, [("english", 50), ("chinese", 50)])

    inputs, targets, coverage = stream.sample_batch(
        64, 3, generator=torch.Generator().manual_seed(41)
    )

    assert inputs.shape == targets.shape == (64, 3)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert sum(coverage.values()) == 64 * 3
    expected = {"english": 0, "chinese": 0}
    for start in inputs[:, 0].tolist():
        for token_position in range(start, start + 3):
            expected["english" if token_position < 50 else "chinese"] += 1
    assert coverage == expected


def test_token_stream_rejects_inconsistent_segments_and_short_windows(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    _write_ids(path, [1, 2, 3, 4])

    with pytest.raises(ValueError, match="cover"):
        TokenStream(path, [("english", 3)])

    stream = TokenStream(path, [("english", 4)])
    with pytest.raises(ValueError, match="shorter"):
        stream.sample_batch(1, 4, generator=torch.Generator().manual_seed(1))


def test_gradient_clipping_bounds_global_explicit_gradient_norm() -> None:
    model = TinyGPT(
        GPTConfig(
            vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=8
        )
    )
    for parameter in model.parameters():
        parameter.grad.fill_(1.0)
    expected_norm = math.sqrt(
        sum(parameter.grad.numel() for parameter in model.parameters())
    )

    actual_norm = clip_grad_norm_(model, max_norm=1.0)
    clipped_norm = math.sqrt(
        sum(
            float(torch.sum(parameter.grad * parameter.grad))
            for parameter in model.parameters()
        )
    )

    assert actual_norm == pytest.approx(expected_norm)
    assert clipped_norm == pytest.approx(1.0, rel=1e-6)


def test_gradient_clipping_rejects_nonfinite_norm() -> None:
    model = TinyGPT(
        GPTConfig(
            vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=8
        )
    )
    model.parameters()[0].grad.fill_(float("inf"))

    with pytest.raises(FloatingPointError, match="not finite"):
        clip_grad_norm_(model, max_norm=1.0)


def test_training_step_updates_model_without_autograd(tmp_path) -> None:
    path = tmp_path / "train.bin"
    _write_ids(path, [index % 16 for index in range(80)])
    stream = TokenStream(path, [("english", 80)])
    model = TinyGPT(
        GPTConfig(
            vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=8
        ),
        generator=torch.Generator().manual_seed(3),
    )
    optimizer = AdamW(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(7)
    initial_values = [parameter.value.clone() for parameter in model.parameters()]

    loss, coverage, gradient_norm = train_step(
        model,
        optimizer,
        stream,
        batch_size=2,
        sequence_length=4,
        generator=generator,
        max_grad_norm=1.0,
    )

    assert math.isfinite(loss)
    assert gradient_norm > 0
    assert coverage == {"english": 8}
    assert optimizer.step_count == 1
    assert any(
        not torch.equal(parameter.value, initial)
        for parameter, initial in zip(model.parameters(), initial_values, strict=True)
    )
    assert all(not parameter.value.requires_grad for parameter in model.parameters())
