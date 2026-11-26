"""A named holder for one tensor (chapter 3).

Chapter 1 does not use ``Parameter``. Chapter 3 adds the gradient buffer and
``zero_grad`` when backpropagation needs them.
"""

from __future__ import annotations

import torch


class Parameter:
    """A named holder for one tensor value."""

    def __init__(self, value: torch.Tensor) -> None:
        raise NotImplementedError("TODO 3: hold the value")

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO 3")
