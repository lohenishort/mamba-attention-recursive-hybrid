"""Loss functions for the Mamba-Attention Recursive Reasoning Hybrid framework."""

import torch
import torch.nn.functional as F


def compute_bce_joint_loss(
    y_final: torch.Tensor,
    target_ids: torch.Tensor,
    bce_probs: list[torch.Tensor],
    correct_mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Computes the joint loss combining sparse task CE and BCE halting head loss.

    Args:
        y_final: Final sequence predictions. Shape: [batch_size, seq_len, d_model]
        target_ids: Target token IDs. Shape: [batch_size, seq_len]
        bce_probs: Probabilities from the halting head for each recursive step.
            List of length `n_steps`, where each tensor has shape [batch_size].
        correct_mask: Binary mask indicating whether each sample was answered correctly.
            Shape: [batch_size]
        alpha: Weighting factor for the halting loss component.

    Returns:
        The combined joint loss tensor (scalar).
    """
    # y_final: [batch_size, seq_len, d_model]
    # target_ids: [batch_size, seq_len]
    # correct_mask: [batch_size]
    # Each prob in bce_probs: [batch_size]

    B, L_ans, D = y_final.shape
    loss_task = F.cross_entropy(y_final.view(-1, D), target_ids.view(-1))

    loss_bce = torch.tensor(0.0, device=y_final.device)
    n_steps = len(bce_probs)
    if n_steps > 0:
        for prob in bce_probs:
            # prob: [batch_size], correct_mask: [batch_size]
            loss_bce = loss_bce + F.binary_cross_entropy(
                prob.to(y_final.device), correct_mask.to(y_final.device)
            )
        loss_bce = loss_bce / n_steps

    # Return combined loss: task loss + alpha * halting loss
    # [1] (scalar)
    return loss_task + alpha * loss_bce
