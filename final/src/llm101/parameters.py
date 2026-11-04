"""Small parameter objects for explicit, non-autograd training."""

from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class Parameter:
    """A tensor value and an explicitly accumulated gradient."""

    value: torch.Tensor
    grad: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("parameter value must be a torch.Tensor")
        if self.value.requires_grad:
            raise ValueError("LLM 101 parameters must not use autograd")
        if not self.value.is_floating_point():
            raise TypeError("parameter values must use a floating-point dtype")

        self.value = self.value.clone()
        self.grad = torch.zeros_like(self.value)

    def zero_grad(self) -> None:
        """Clear accumulated gradients in place."""
        self.grad.zero_()
