import pytest
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


def test_adamw_state_round_trip_continues_identically() -> None:
    original = Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = AdamW([original], lr=0.02, weight_decay=0.03)
    original.grad.copy_(torch.tensor([0.5, -0.25], dtype=torch.float64))
    optimizer.step()

    resumed_parameter = Parameter(original.value.clone())
    resumed_optimizer = AdamW([resumed_parameter], lr=0.5)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    original.grad.copy_(torch.tensor([-0.1, 0.4], dtype=torch.float64))
    resumed_parameter.grad.copy_(original.grad)

    optimizer.step()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed_parameter.value, original.value)
    assert resumed_optimizer.step_count == optimizer.step_count
    assert resumed_optimizer.lr == optimizer.lr


def test_optimizers_reject_non_finite_hyperparameters() -> None:
    parameter = Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="finite"):
        SGD([parameter], lr=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        AdamW([parameter], lr=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        AdamW([parameter], eps=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        AdamW([parameter], weight_decay=float("nan"))


def test_adamw_rejects_malformed_state_without_partial_restore() -> None:
    source_parameters = [
        Parameter(torch.tensor([1.0, 2.0])),
        Parameter(torch.tensor([3.0, 4.0])),
    ]
    saved_state = AdamW(source_parameters, lr=0.01).state_dict()
    saved_state["first_moments"][1] = torch.zeros(1)

    target_parameters = [
        Parameter(torch.tensor([1.0, 2.0])),
        Parameter(torch.tensor([3.0, 4.0])),
    ]
    target_optimizer = AdamW(target_parameters, lr=0.2)
    target_optimizer.first_moments[0].fill_(5.0)
    target_optimizer.second_moments[0].fill_(7.0)

    with pytest.raises(ValueError, match="moment shape"):
        target_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        target_optimizer.first_moments[0], torch.tensor([5.0, 5.0])
    )
    torch.testing.assert_close(
        target_optimizer.second_moments[0], torch.tensor([7.0, 7.0])
    )
    assert target_optimizer.lr == 0.2
