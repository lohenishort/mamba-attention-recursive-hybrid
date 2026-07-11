"""Unit tests for the joint loss calculations."""

import torch
from mamba_hybrid.loss import compute_bce_joint_loss


def test_bce_joint_loss() -> None:
    # y_final shape: [batch_size, seq_len, d_model]
    y_final: torch.Tensor = torch.randn(2, 8, 64)
    # target_ids shape: [batch_size, seq_len]
    target_ids: torch.Tensor = torch.randint(0, 64, (2, 8))
    # list of tensors, each of shape [batch_size]
    bce_probs: list[torch.Tensor] = [torch.tensor([0.1, 0.2]) for _ in range(6)]
    # correct_mask shape: [batch_size]
    correct_mask: torch.Tensor = torch.tensor([1.0, 0.0])

    loss: torch.Tensor = compute_bce_joint_loss(
        y_final, target_ids, bce_probs, correct_mask
    )
    assert loss > 0
