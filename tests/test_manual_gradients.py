from itertools import product

import torch

from llm101.layers import GELU, Embedding, LayerNorm, Linear
from llm101.losses import softmax_cross_entropy


def finite_difference(tensor: torch.Tensor, function, eps: float = 1e-6) -> torch.Tensor:
    """Central finite differences for tiny tensors; deliberately not an autograd tool."""
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


def test_linear_backward_matches_finite_differences() -> None:
    layer = Linear(2, 3, dtype=torch.float64)
    inputs = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float64)
    grad_output = torch.tensor(
        [[0.1, -0.2, 0.5], [0.4, 0.3, -0.1]], dtype=torch.float64
    )

    output = layer.forward(inputs)
    assert not output.requires_grad
    grad_input = layer.backward(grad_output)

    numerical_input = finite_difference(
        inputs, lambda: ((inputs @ layer.weight.value + layer.bias.value) * grad_output).sum()
    )
    numerical_weight = finite_difference(
        layer.weight.value,
        lambda: ((inputs @ layer.weight.value + layer.bias.value) * grad_output).sum(),
    )
    numerical_bias = finite_difference(
        layer.bias.value,
        lambda: ((inputs @ layer.weight.value + layer.bias.value) * grad_output).sum(),
    )

    torch.testing.assert_close(grad_input, numerical_input, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(layer.weight.grad, numerical_weight, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(layer.bias.grad, numerical_bias, atol=1e-8, rtol=1e-6)


def test_embedding_backward_accumulates_repeated_token_ids() -> None:
    layer = Embedding(5, 2, dtype=torch.float64)
    indices = torch.tensor([[1, 2, 1]], dtype=torch.long)
    grad_output = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64
    )

    layer.forward(indices)
    layer.backward(grad_output)

    expected = torch.zeros_like(layer.weight.value)
    expected[1] = torch.tensor([6.0, 8.0], dtype=torch.float64)
    expected[2] = torch.tensor([3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(layer.weight.grad, expected)


def test_layer_norm_supplied_backward_matches_finite_differences() -> None:
    layer = LayerNorm(3, dtype=torch.float64)
    inputs = torch.tensor([[0.2, -0.4, 0.7], [0.1, 0.9, -0.3]], dtype=torch.float64)
    grad_output = torch.tensor([[0.4, -0.2, 0.3], [0.5, 0.1, -0.6]], dtype=torch.float64)

    output = layer.forward(inputs)
    assert torch.isfinite(output).all()
    grad_input = layer.backward(grad_output)

    def projected_loss() -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        centered = inputs - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + layer.eps)
        return ((normalized * layer.weight.value + layer.bias.value) * grad_output).sum()

    numerical_input = finite_difference(inputs, projected_loss)
    numerical_weight = finite_difference(layer.weight.value, projected_loss)
    numerical_bias = finite_difference(layer.bias.value, projected_loss)

    torch.testing.assert_close(grad_input, numerical_input, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(layer.weight.grad, numerical_weight, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(layer.bias.grad, numerical_bias, atol=1e-7, rtol=1e-5)


def test_gelu_supplied_backward_matches_finite_differences() -> None:
    activation = GELU()
    inputs = torch.tensor([-1.2, -0.1, 0.5, 1.4], dtype=torch.float64)
    grad_output = torch.tensor([0.3, -0.5, 0.7, 0.2], dtype=torch.float64)

    output = activation.forward(inputs)
    grad_input = activation.backward(grad_output)
    numerical = finite_difference(inputs, lambda: (activation_formula(inputs) * grad_output).sum())

    torch.testing.assert_close(output, activation_formula(inputs))
    torch.testing.assert_close(grad_input, numerical, atol=1e-8, rtol=1e-6)


def activation_formula(inputs: torch.Tensor) -> torch.Tensor:
    return 0.5 * inputs * (1.0 + torch.erf(inputs / (2.0**0.5)))


def test_softmax_cross_entropy_gradient_matches_finite_differences() -> None:
    logits = torch.tensor(
        [[0.2, -0.3, 1.1], [1.5, 0.4, -0.2]], dtype=torch.float64
    )
    targets = torch.tensor([2, 0], dtype=torch.long)

    loss, gradient = softmax_cross_entropy(logits, targets)
    numerical = finite_difference(
        logits, lambda: softmax_cross_entropy(logits, targets)[0]
    )

    assert loss.ndim == 0
    torch.testing.assert_close(gradient, numerical, atol=1e-8, rtol=1e-6)


def test_softmax_cross_entropy_is_stable_for_large_logits() -> None:
    logits = torch.tensor([[10000.0, 9999.0, -10000.0]], dtype=torch.float64)
    targets = torch.tensor([1], dtype=torch.long)

    loss, gradient = softmax_cross_entropy(logits, targets)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
