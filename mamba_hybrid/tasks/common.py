"""Shared task-target utilities."""

import torch


def shift_targets_right(
    targets: torch.Tensor, *, bos_token_id: int, pad_token_id: int
) -> torch.Tensor:
    """Build teacher-forcing inputs from target token sequences. [B,L]."""
    if targets.ndim != 2 or targets.shape[1] == 0:
        raise ValueError("targets must have shape [batch_size, positive_length]")
    shifted = torch.full_like(targets, pad_token_id)
    shifted[:, 0] = bos_token_id
    shifted[:, 1:] = targets[:, :-1]
    return shifted


__all__ = ["shift_targets_right"]
