"""Loss functions for the Mamba-Attention Recursive Reasoning Hybrid framework."""

import torch
import torch.nn.functional as F
from typing import Protocol


class _QHead(Protocol):
    def get_q_values(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


class _TargetModel(Protocol):
    @property
    def q_head(self) -> _QHead: ...


def compute_bce_joint_loss(
    y_final: torch.Tensor,
    target_ids: torch.Tensor,
    bce_probs: list[torch.Tensor],
    correct_mask: torch.Tensor,
    alpha: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes the joint loss combining sparse task CE and BCE halting head loss.

    Args:
        y_final: Final vocabulary logits. Shape: [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs. Shape: [batch_size, seq_len]
        bce_probs: Probabilities from the halting head for each recursive step.
            List of length `n_steps`, where each tensor has shape [batch_size].
        correct_mask: Binary mask indicating whether each sample was answered correctly.
            Shape: [batch_size]
        alpha: Weighting factor for the halting loss component.
        ignore_index: Token ID to ignore in cross-entropy calculation.

    Returns:
        The combined joint loss tensor (scalar).
    """
    # y_final: [batch_size, seq_len, d_model]
    # target_ids: [batch_size, seq_len]
    # correct_mask: [batch_size]
    # Each prob in bce_probs: [batch_size]

    _, _, vocab_size = y_final.shape
    loss_task = F.cross_entropy(
        y_final.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )

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
    target_model: _TargetModel,
    alpha: float = 1.0,
    gamma: float = 1.0,
    states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ignore_index: int = -100,
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
        ignore_index: Token ID to ignore in cross-entropy calculation.

    Returns:
        The combined joint loss tensor (scalar).
    """
    # y_final: [batch_size, seq_len, d_model]
    # target_ids: [batch_size, seq_len]
    # correct_mask: [batch_size]
    # Each pred in q_preds: [batch_size, 2]
    _, _, vocab_size = y_final.shape
    loss_task = F.cross_entropy(
        y_final.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )

    loss_q = torch.tensor(0.0, device=y_final.device)
    n_steps = len(q_preds)

    if n_steps > 0:
        if states is None or len(states) != n_steps:
            raise ValueError("states must contain one state for every Q prediction")
        if correct_mask.ndim == 1:
            rewards = correct_mask.unsqueeze(0).expand(n_steps, -1)
        elif correct_mask.shape == (n_steps, y_final.shape[0]):
            rewards = correct_mask
        else:
            raise ValueError("correct_mask must have shape [B] or [steps, B]")
        for t in range(n_steps):
            q_halt_target = rewards[t].to(y_final.device)
            q_cont_target: torch.Tensor | None = None
            if t < n_steps - 1:
                next_z, next_y = states[t + 1]
                with torch.no_grad():
                    q_next = target_model.q_head.get_q_values(
                        next_z.to(y_final.device), next_y.to(y_final.device)
                    )
                q_cont_target = gamma * torch.max(q_next, dim=-1)[0]

            # q_preds[t]: [batch_size, 2]
            # Column 1 represents Q(s_t, halt), Column 0 represents Q(s_t, continue)
            loss_q = (
                loss_q
                + (q_preds[t][:, 1].to(y_final.device) - q_halt_target).pow(2).mean()
            )
            if t < n_steps - 1:
                assert q_cont_target is not None
                loss_q = (
                    loss_q
                    + (q_preds[t][:, 0].to(y_final.device) - q_cont_target)
                    .pow(2)
                    .mean()
                )

        loss_q = loss_q / n_steps

    return loss_task + alpha * loss_q
