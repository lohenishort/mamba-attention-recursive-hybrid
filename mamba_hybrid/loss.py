"""Loss functions for the Mamba-Attention Recursive Reasoning Hybrid framework."""

import torch
import torch.nn.functional as F
from typing import Any


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
    loss_task = F.cross_entropy(y_final.view(-1, D), target_ids.view(-1), ignore_index=0)

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


def compute_q_joint_loss(
    y_final: torch.Tensor,
    target_ids: torch.Tensor,
    q_preds: list[torch.Tensor],
    correct_mask: torch.Tensor,
    target_model: Any,
    alpha: float = 1.0,
    gamma: float = 1.0,
    states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Computes the joint loss combining sparse task CE and Q-learning halting loss.

    Args:
        y_final: Final sequence predictions. Shape: [batch_size, seq_len, d_model]
        target_ids: Target token IDs. Shape: [batch_size, seq_len]
        q_preds: Q-value predictions for each step.
            List of length `n_steps`, where each tensor has shape [batch_size, 2].
        correct_mask: Binary mask indicating whether each sample was answered correctly.
            Shape: [batch_size]
        target_model: Target model (MambaAttentionHybrid) for Q-learning bootstrapping.
        alpha: Weighting factor for the halting Q-loss component.
        gamma: Discount factor for bootstrapping.
        states: Optional list of tuples (z, y) at each step of the supervision cycle.
            Used for true target network bootstrapping.

    Returns:
        The combined joint loss tensor (scalar).
    """
    # y_final: [batch_size, seq_len, d_model]
    # target_ids: [batch_size, seq_len]
    # correct_mask: [batch_size]
    # Each pred in q_preds: [batch_size, 2]
    B, L_ans, D = y_final.shape
    loss_task = F.cross_entropy(y_final.view(-1, D), target_ids.view(-1), ignore_index=0)

    loss_q = torch.tensor(0.0, device=y_final.device)
    n_steps = len(q_preds)

    if n_steps > 0:
        for t in range(n_steps):
            q_halt_target = correct_mask.to(y_final.device)
            if t < n_steps - 1:
                # Get next state and run target model
                if states is not None and t + 1 < len(states):
                    next_z, next_y = states[t + 1]
                else:
                    # fallback simulation if states not provided or not fully populated
                    # next_z: [B, n_meta, D], next_y: [B, l_ans, D]
                    B = y_final.shape[0]
                    D = y_final.shape[2]
                    next_z = torch.zeros(B, target_model.n_meta, D, device=y_final.device)
                    next_y = torch.zeros(B, target_model.l_ans, D, device=y_final.device)

                with torch.no_grad():
                    # target_model.q_head.get_q_values: [batch_size, 2]
                    q_next = target_model.q_head.get_q_values(
                        next_z.to(y_final.device), next_y.to(y_final.device)
                    )
                q_cont_target = gamma * torch.max(q_next, dim=-1)[0]
            else:
                q_cont_target = correct_mask.to(y_final.device)

            # q_preds[t]: [batch_size, 2]
            # Column 1 represents Q(s_t, halt), Column 0 represents Q(s_t, continue)
            loss_q = (
                loss_q
                + (q_preds[t][:, 1].to(y_final.device) - q_halt_target).pow(2).mean()
            )
            if t < n_steps - 1:
                loss_q = (
                    loss_q
                    + (q_preds[t][:, 0].to(y_final.device) - q_cont_target)
                    .pow(2)
                    .mean()
                )

        loss_q = loss_q / n_steps

    return loss_task + alpha * loss_q
