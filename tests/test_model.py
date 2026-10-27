from itertools import product

import torch

from llm101.model import GPTConfig, TinyGPT


def finite_difference(tensor: torch.Tensor, function, eps: float = 1e-6) -> torch.Tensor:
    numerical = torch.empty_like(tensor)
    for index in product(*(range(size) for size in tensor.shape)):
        original = tensor[index].item()
        tensor[index] = original + eps
        positive = function().item()
        tensor[index] = original - eps
        negative = function().item()
        tensor[index] = original
        numerical[index] = (positive - negative) / (2.0 * eps)
    return numerical


def tiny_config(**overrides: int) -> GPTConfig:
    settings = {
        "vocab_size": 5,
        "d_model": 2,
        "n_layers": 1,
        "n_heads": 1,
        "d_ff": 3,
        "max_seq_len": 3,
    }
    settings.update(overrides)
    return GPTConfig(**settings)


def test_tiny_gpt_shape_tying_and_parameter_count() -> None:
    config = tiny_config(vocab_size=7, d_model=4, n_heads=2, d_ff=8)
    model = TinyGPT(config, dtype=torch.float64)
    input_ids = torch.tensor([[1, 2, 3], [2, 4, 1]], dtype=torch.long)

    logits = model.forward(input_ids)

    assert logits.shape == (2, 3, config.vocab_size)
    assert not logits.requires_grad
    parameters = model.parameters()
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert all(not parameter.value.requires_grad for parameter in parameters)
    assert model.token_embedding.weight in parameters
    assert model.parameter_count() == 220


def test_tied_vocabulary_projection_accumulates_gradient_into_embedding() -> None:
    model = TinyGPT(tiny_config(vocab_size=6), dtype=torch.float64)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    logits = model.forward(input_ids)
    grad_logits = torch.tensor(
        [[[0.1, 0.2, -0.3, 0.4, -0.2, 0.5],
          [-0.4, 0.3, 0.2, -0.1, 0.6, -0.2]]],
        dtype=torch.float64,
    )
    assert model._final_hidden_states is not None
    expected_output_head_grad = (
        grad_logits.reshape(-1, model.config.vocab_size).T
        @ model._final_hidden_states.reshape(-1, model.config.d_model)
    )

    model.backward(grad_logits)

    # Token 5 is absent from the input IDs, so its embedding gradient can only
    # come from the shared vocabulary-projection matrix.
    torch.testing.assert_close(
        model.token_embedding.weight.grad[5], expected_output_head_grad[5]
    )
    assert logits.shape[-1] == model.config.vocab_size


def test_tiny_gpt_all_parameter_gradients_match_finite_differences() -> None:
    generator = torch.Generator().manual_seed(17)
    model = TinyGPT(tiny_config(), dtype=torch.float64, generator=generator)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    grad_logits = torch.tensor(
        [[[0.4, -0.2, 0.3, 0.1, -0.5], [-0.1, 0.5, -0.4, 0.2, 0.3]]],
        dtype=torch.float64,
    )

    model.forward(input_ids)
    model.backward(grad_logits)

    def projected_loss() -> torch.Tensor:
        return (model.forward(input_ids) * grad_logits).sum()

    for parameter in model.parameters():
        analytic_gradient = parameter.grad.clone()
        numerical_gradient = finite_difference(parameter.value, projected_loss)
        torch.testing.assert_close(
            analytic_gradient, numerical_gradient, atol=2e-6, rtol=2e-4
        )


def test_tiny_gpt_rejects_sequences_beyond_position_table() -> None:
    model = TinyGPT(tiny_config(max_seq_len=2))
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    try:
        model.forward(input_ids)
    except ValueError as error:
        assert "exceeds configured maximum" in str(error)
    else:
        raise AssertionError("expected an overlong sequence to be rejected")
