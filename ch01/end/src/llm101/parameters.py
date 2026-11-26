"""Trainable tensors (chapter 1).

Chapter 1 never touches ``grad``; chapter 3 starts writing into it.
"""

from __future__ import annotations

import torch


class Parameter:
    """A tensor plus a same-shaped ``grad`` buffer for later updates."""

    def __init__(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("parameter value must be a torch.Tensor")
        if value.requires_grad:
            raise ValueError("LLM 101 does not use autograd")
        if not value.is_floating_point():
            raise TypeError("parameter values must use a floating-point dtype")

        self.value = value.clone()
        self.grad = torch.zeros_like(self.value)

    def zero_grad(self) -> None:
        self.grad.zero_()
