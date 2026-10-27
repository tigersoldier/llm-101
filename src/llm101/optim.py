"""Minimal SGD and AdamW optimizers over explicit-gradient PyTorch parameters."""

from collections.abc import Iterable

import torch

from llm101.parameters import Parameter


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
        if lr <= 0:
            raise ValueError("lr must be positive")
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
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

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
