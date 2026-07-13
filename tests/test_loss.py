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


def test_bce_uses_per_cycle_targets_and_respects_minimum_cycles() -> None:
    first_probability = torch.tensor([0.5], requires_grad=True)
    second_probability = torch.tensor([0.5], requires_grad=True)
    logits = torch.randn(1, 1, 3, requires_grad=True)
    targets = torch.tensor([[1]])
    cycle_correct = torch.tensor([[1.0], [1.0]])

    loss = compute_bce_joint_loss(
        logits,
        targets,
        [first_probability, second_probability],
        cycle_correct,
        min_cycles=2,
    )
    loss.backward()  # type: ignore[no-untyped-call]

    assert first_probability.grad is not None
    assert second_probability.grad is not None
    assert first_probability.grad.item() > 0  # halt forbidden before M_min
    assert second_probability.grad.item() < 0  # correct answer should halt
