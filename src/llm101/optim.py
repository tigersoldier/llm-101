"""Minimal SGD and AdamW optimizers over explicit-gradient PyTorch parameters."""

import math
from collections.abc import Iterable
from typing import TypedDict

import torch

from llm101.parameters import Parameter


class AdamWState(TypedDict):
    step_count: int
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    first_moments: list[torch.Tensor]
    second_moments: list[torch.Tensor]


def _unique_parameters(parameters: Iterable[Parameter]) -> tuple[Parameter, ...]:
    unique: list[Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if not isinstance(parameter, Parameter):
            raise TypeError("optimizers accept llm101.Parameter objects")
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            unique.append(parameter)
    if not unique:
        raise ValueError("an optimizer needs at least one parameter")
    return tuple(unique)


class SGD:
    """Stochastic gradient descent with a single learning rate."""

    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.parameters = _unique_parameters(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.value.add_(parameter.grad, alpha=-self.lr)


class AdamW:
    """A compact AdamW implementation with bias correction and decoupled decay."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        beta1, beta2 = betas
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")

        self.parameters = _unique_parameters(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.first_moments = [torch.zeros_like(p.value) for p in self.parameters]
        self.second_moments = [torch.zeros_like(p.value) for p in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self.step_count += 1
        first_correction = 1.0 - self.beta1**self.step_count
        second_correction = 1.0 - self.beta2**self.step_count

        for parameter, first, second in zip(
            self.parameters,
            self.first_moments,
            self.second_moments,
            strict=True,
        ):
            first.mul_(self.beta1).add_(parameter.grad, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                parameter.grad, parameter.grad, value=1.0 - self.beta2
            )
            corrected_first = first / first_correction
            corrected_second = second / second_correction

            if self.weight_decay:
                parameter.value.mul_(1.0 - self.lr * self.weight_decay)
            denominator = corrected_second.sqrt().add_(self.eps)
            parameter.value.addcdiv_(corrected_first, denominator, value=-self.lr)

    def state_dict(self) -> AdamWState:
        """Return CPU copies of optimizer state for a durable checkpoint."""
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "first_moments": [
                moment.detach().cpu().clone() for moment in self.first_moments
            ],
            "second_moments": [
                moment.detach().cpu().clone() for moment in self.second_moments
            ],
        }

    def load_state_dict(self, state: AdamWState) -> None:
        """Restore moments and hyperparameters after validating parameter shapes."""
        if state["step_count"] < 0:
            raise ValueError("optimizer step_count cannot be negative")
        if not all(
            math.isfinite(value)
            for value in (
                state["lr"],
                state["beta1"],
                state["beta2"],
                state["eps"],
                state["weight_decay"],
            )
        ):
            raise ValueError("saved AdamW hyperparameters must be finite")
        if state["lr"] <= 0 or state["eps"] <= 0 or state["weight_decay"] < 0:
            raise ValueError("invalid AdamW hyperparameter in saved state")
        if not 0 <= state["beta1"] < 1 or not 0 <= state["beta2"] < 1:
            raise ValueError("invalid AdamW beta in saved state")

        first_moments = state["first_moments"]
        second_moments = state["second_moments"]
        if len(first_moments) != len(self.parameters) or len(second_moments) != len(
            self.parameters
        ):
            raise ValueError("saved AdamW moment count does not match parameter count")
        for parameter, saved_first, saved_second in zip(
            self.parameters, first_moments, second_moments, strict=True
        ):
            if (
                saved_first.shape != parameter.value.shape
                or saved_second.shape != parameter.value.shape
            ):
                raise ValueError(
                    "saved AdamW moment shape does not match parameter shape"
                )
        for first, second, saved_first, saved_second in zip(
            self.first_moments,
            self.second_moments,
            first_moments,
            second_moments,
            strict=True,
        ):
            first.copy_(saved_first.to(device=first.device, dtype=first.dtype))
            second.copy_(saved_second.to(device=second.device, dtype=second.dtype))

        self.step_count = state["step_count"]
        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]
