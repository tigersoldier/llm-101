"""Trainable tensors (chapter 1 scaffold).

Chapter 1 never touches ``grad``; chapter 3 starts writing into it.
"""

from __future__ import annotations

import torch


class Parameter:
    """A tensor plus a same-shaped gradient buffer."""

    def __init__(self, value: torch.Tensor) -> None:
        raise NotImplementedError("TODO 1.3: hold value and a zeroed grad")

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO 1.3")
