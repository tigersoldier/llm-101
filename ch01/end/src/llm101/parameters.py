"""A named holder for one tensor (chapter 1).

Chapter 1 only reads ``value``. Chapter 3 adds the ``grad`` buffer and
``zero_grad`` when backpropagation needs them.
"""

from __future__ import annotations

import torch


class Parameter:
    """A named holder for one tensor value."""

    def __init__(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("parameter value must be a torch.Tensor")
        if value.requires_grad:
            raise ValueError("LLM 101 does not use autograd")
        if not value.is_floating_point():
            raise TypeError("parameter values must use a floating-point dtype")

        self.value = value.clone()
