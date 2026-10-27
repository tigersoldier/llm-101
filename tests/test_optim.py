import torch

from llm101.optim import SGD, AdamW
from llm101.parameters import Parameter


def test_sgd_updates_once_when_parameter_is_listed_twice() -> None:
    parameter = Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad.copy_(torch.tensor([0.5, -1.0]))
    optimizer = SGD([parameter, parameter], lr=0.1)

    optimizer.step()

    torch.testing.assert_close(parameter.value, torch.tensor([0.95, -1.9]))


def test_adamw_uses_bias_correction_and_decoupled_weight_decay() -> None:
    parameter = Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameter.grad.fill_(1.0)
    optimizer = AdamW(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    optimizer.step()

    expected = 1.0 * (1.0 - 0.1 * 0.01) - 0.1 / (1.0 + 1e-8)
    torch.testing.assert_close(
        parameter.value,
        torch.tensor([expected], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert optimizer.step_count == 1


def test_optimizer_zero_grad_clears_accumulated_gradient() -> None:
    parameter = Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad.fill_(3.0)
    optimizer = SGD([parameter], lr=0.1)

    optimizer.zero_grad()

    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))
