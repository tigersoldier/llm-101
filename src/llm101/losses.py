"""PyTorch loss functions with explicit, non-autograd gradients."""

import torch


def softmax_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean next-token cross entropy and its gradient with respect to logits.

    ``logits`` has shape ``(..., vocabulary_size)`` and ``targets`` has shape
    ``(...)``. The implementation uses the stable log-sum-exp identity and
    computes ``(softmax(logits) - one_hot(targets)) / number_of_targets``
    directly, without autograd.
    """
    if logits.ndim < 2:
        raise ValueError("logits must include batch/position and vocabulary dimensions")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape must be {tuple(logits.shape[:-1])}, "
            f"got {tuple(targets.shape)}"
        )
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    if logits.shape[-1] == 0:
        raise ValueError("vocabulary dimension must be non-empty")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if flat_targets.numel() == 0:
        raise ValueError("targets must contain at least one token")
    if bool(torch.any(flat_targets < 0)) or bool(torch.any(flat_targets >= logits.shape[-1])):
        raise IndexError("target ID is outside the vocabulary")

    shifted = flat_logits - flat_logits.max(dim=-1, keepdim=True).values
    exponentials = shifted.exp()
    normalizer = exponentials.sum(dim=-1, keepdim=True)
    probabilities = exponentials / normalizer

    correct_logits = shifted.gather(1, flat_targets[:, None]).squeeze(1)
    log_partition = normalizer.squeeze(1).log()
    loss = (log_partition - correct_logits).mean()

    grad_logits = probabilities
    row_ids = torch.arange(flat_targets.numel(), device=logits.device)
    grad_logits[row_ids, flat_targets] -= 1.0
    grad_logits /= flat_targets.numel()
    return loss, grad_logits.reshape_as(logits)
