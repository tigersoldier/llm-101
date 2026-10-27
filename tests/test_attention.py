from itertools import product

import torch

from llm101.attention import MultiHeadSelfAttention


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


def test_causal_attention_forward_and_backward_match_finite_differences() -> None:
    attention = MultiHeadSelfAttention(4, 2, dtype=torch.float64)
    inputs = torch.tensor(
        [[[0.2, -0.4, 0.1, 0.7], [0.5, 0.3, -0.2, 0.8], [-0.1, 0.6, 0.4, -0.3]]],
        dtype=torch.float64,
    )
    grad_output = torch.tensor(
        [[[0.4, -0.2, 0.1, 0.5], [0.3, 0.6, -0.7, 0.2], [-0.5, 0.1, 0.8, -0.4]]],
        dtype=torch.float64,
    )

    output = attention.forward(inputs)
    grad_input = attention.backward(grad_output)

    def projected_loss() -> torch.Tensor:
        return (attention.forward(inputs) * grad_output).sum()

    numerical_input = finite_difference(inputs, projected_loss)
    torch.testing.assert_close(grad_input, numerical_input, atol=2e-7, rtol=2e-5)

    for parameter in attention.parameters():
        analytic_gradient = parameter.grad.clone()
        numerical_gradient = finite_difference(parameter.value, projected_loss)
        torch.testing.assert_close(
            analytic_gradient, numerical_gradient, atol=2e-7, rtol=2e-5
        )
    assert output.shape == inputs.shape
    assert not output.requires_grad


def test_changing_future_tokens_does_not_change_earlier_outputs() -> None:
    attention = MultiHeadSelfAttention(4, 2, dtype=torch.float64)
    original = torch.tensor(
        [[[0.2, -0.4, 0.1, 0.7], [0.5, 0.3, -0.2, 0.8], [-0.1, 0.6, 0.4, -0.3]]],
        dtype=torch.float64,
    )
    changed_future = original.clone()
    changed_future[:, 2, :] = torch.tensor([10.0, -8.0, 6.0, 4.0], dtype=torch.float64)

    original_output = attention.forward(original)
    changed_output = attention.forward(changed_future)

    torch.testing.assert_close(original_output[:, :2], changed_output[:, :2])


def test_attention_requires_divisible_head_width() -> None:
    try:
        MultiHeadSelfAttention(d_model=5, n_heads=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected invalid head dimensions to be rejected")
