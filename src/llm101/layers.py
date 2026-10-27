"""Readable PyTorch layers with explicit backward methods and no autograd."""

from __future__ import annotations

import math

import torch

from llm101.parameters import Parameter


class Linear:
    """An affine layer whose final input dimension is ``in_features``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        scale = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.randn(
            in_features, out_features, dtype=dtype, generator=generator
        ) * scale
        self.weight = Parameter(weight)
        self.bias = Parameter(torch.zeros(out_features, dtype=dtype))
        self.in_features = in_features
        self.out_features = out_features
        self._input: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected final input dimension {self.in_features}, "
                f"got {inputs.shape[-1]}"
            )
        self._input = inputs
        return inputs @ self.weight.value + self.bias.value

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Accumulate parameter gradients and return the input gradient."""
        if self._input is None:
            raise RuntimeError("Linear.backward() requires a preceding forward()")
        if grad_output.shape[:-1] != self._input.shape[:-1]:
            raise ValueError("grad_output leading dimensions do not match the input")
        if grad_output.shape[-1] != self.out_features:
            raise ValueError(
                f"expected grad_output final dimension {self.out_features}, "
                f"got {grad_output.shape[-1]}"
            )

        flat_input = self._input.reshape(-1, self.in_features)
        flat_grad = grad_output.reshape(-1, self.out_features)
        self.weight.grad.add_(flat_input.T @ flat_grad)
        self.bias.grad.add_(flat_grad.sum(dim=0))
        grad_input = (flat_grad @ self.weight.value.T).reshape_as(self._input)
        self._input = None
        return grad_input

    def parameters(self) -> tuple[Parameter, Parameter]:
        return self.weight, self.bias


class Embedding:
    """A token lookup table with explicit gradient accumulation."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        initial = torch.randn(
            num_embeddings, embedding_dim, dtype=dtype, generator=generator
        ) * 0.02
        self.weight = Parameter(initial)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._indices: torch.Tensor | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            raise TypeError("embedding indices must have dtype torch.long")
        if indices.numel() and (
            bool(torch.any(indices < 0))
            or bool(torch.any(indices >= self.num_embeddings))
        ):
            raise IndexError("embedding index is outside the vocabulary")
        self._indices = indices.clone()
        return self.weight.value[indices]

    def backward(self, grad_output: torch.Tensor) -> None:
        """Add each output gradient to its token row; repeated IDs accumulate."""
        if self._indices is None:
            raise RuntimeError("Embedding.backward() requires a preceding forward()")
        expected_shape = (*self._indices.shape, self.embedding_dim)
        if grad_output.shape != expected_shape:
            raise ValueError(
                f"expected grad_output shape {expected_shape}, got {tuple(grad_output.shape)}"
            )

        flat_indices = self._indices.reshape(-1)
        flat_grad = grad_output.reshape(-1, self.embedding_dim)
        self.weight.grad.index_add_(0, flat_indices, flat_grad)
        self._indices = None

    def parameters(self) -> tuple[Parameter]:
        return (self.weight,)


class LayerNorm:
    """Layer normalization over the last dimension, with supplied backward code."""

    def __init__(
        self,
        features: int,
        *,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if features <= 0:
            raise ValueError("features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = Parameter(torch.ones(features, dtype=dtype))
        self.bias = Parameter(torch.zeros(features, dtype=dtype))
        self.features = features
        self.eps = eps
        self._normalized: torch.Tensor | None = None
        self._inverse_std: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.features:
            raise ValueError(
                f"expected final input dimension {self.features}, "
                f"got {inputs.shape[-1]}"
            )
        mean = inputs.mean(dim=-1, keepdim=True)
        centered = inputs - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        inverse_std = torch.rsqrt(variance + self.eps)
        normalized = centered * inverse_std
        self._normalized = normalized
        self._inverse_std = inverse_std
        return normalized * self.weight.value + self.bias.value

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Apply the standard LayerNorm gradient; derivation is supplied in the lesson."""
        if self._normalized is None or self._inverse_std is None:
            raise RuntimeError("LayerNorm.backward() requires a preceding forward()")
        if grad_output.shape != self._normalized.shape:
            raise ValueError("grad_output shape does not match the LayerNorm input")

        reduce_dims = tuple(range(grad_output.ndim - 1))
        self.weight.grad.add_((grad_output * self._normalized).sum(dim=reduce_dims))
        self.bias.grad.add_(grad_output.sum(dim=reduce_dims))

        grad_normalized = grad_output * self.weight.value
        feature_count = self.features
        grad_input = (self._inverse_std / feature_count) * (
            feature_count * grad_normalized
            - grad_normalized.sum(dim=-1, keepdim=True)
            - self._normalized
            * (grad_normalized * self._normalized).sum(dim=-1, keepdim=True)
        )
        self._normalized = None
        self._inverse_std = None
        return grad_input

    def parameters(self) -> tuple[Parameter, Parameter]:
        return self.weight, self.bias


class GELU:
    """Exact GELU activation with a supplied analytic derivative."""

    def __init__(self) -> None:
        self._input: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._input = inputs
        return 0.5 * inputs * (1.0 + torch.erf(inputs / math.sqrt(2.0)))

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input is None:
            raise RuntimeError("GELU.backward() requires a preceding forward()")
        if grad_output.shape != self._input.shape:
            raise ValueError("grad_output shape does not match the GELU input")

        scaled = self._input / math.sqrt(2.0)
        cdf = 0.5 * (1.0 + torch.erf(scaled))
        density = torch.exp(-0.5 * self._input.square()) / math.sqrt(2.0 * math.pi)
        grad_input = grad_output * (cdf + self._input * density)
        self._input = None
        return grad_input
