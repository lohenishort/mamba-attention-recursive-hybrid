import math
import torch
from mamba_hybrid.model import MambaAttentionHybrid


def ptrm_inference(
    input_ids: torch.Tensor,
    model: MambaAttentionHybrid,
    K: int = 5,
    sigma_base: float = 0.05,
    max_noise_step: int = 20,
) -> torch.Tensor:
    """Probabilistic Tiny Recursive Model (PTRM) inference with majority-consensus voting.

    Args:
        input_ids: Raw input context of shape [B, L_raw, D]
        model: The trained MambaAttentionHybrid model.
        K: Number of stochastic rollouts to sample.
        sigma_base: Base standard deviation for Gaussian noise injection.
        max_noise_step: Maximum step index (1-based, across cycles) to inject noise.

    Returns:
        best_output: Consensus-selected answer predictions of shape [B, L_ans, D]
    """
    # input_ids: [batch_size, seq_len, d_model]
    B, L_raw, D = input_ids.shape

    if K == 1:
        with torch.no_grad():
            y_final: torch.Tensor
            y_final, _ = model(input_ids)
            return y_final

    candidates: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []

    with torch.no_grad():
        for k in range(K):
            # z: [batch_size, n_meta, d_model]
            z: torch.Tensor = model.M_meta.expand(B, -1, -1)
            # y: [batch_size, l_ans, d_model]
            y: torch.Tensor = model.init_answer(input_ids)

            for c in range(1, model.t_cycles + 1):
                for i in range(1, model.n_steps + 1):
                    step_idx: int = (c - 1) * model.n_steps + i
                    if step_idx <= max_noise_step:
                        # Compute step-dependent noise scale
                        sigma: float = sigma_base * math.sqrt(1.0 - i / model.n_steps)
                        z = z + torch.randn_like(z) * sigma

                    # X_concat: [batch_size, n_meta + l_ans + seq_len, d_model]
                    X_concat: torch.Tensor = torch.cat([z, y, input_ids], dim=1)
                    # Update planning state z: [batch_size, n_meta, d_model]
                    z = model.planning_loop.planning_block(X_concat, causal=False)[
                        :, : model.n_meta, :
                    ]
                # Update answer state y: [batch_size, l_ans, d_model]
                y = model.planning_loop.answer_update_block(z, y)

            # prob: [batch_size]
            prob: torch.Tensor = model.q_head(z, y)
            candidates.append(y)
            scores.append(prob)

    # Consensus Voting Selection
    stacked_cand: torch.Tensor = torch.stack(candidates, dim=0)  # [K, B, L_ans, D]
    stacked_scores: torch.Tensor = torch.stack(scores, dim=0)  # [K, B]

    best_outputs: list[torch.Tensor] = []
    for b in range(B):
        # batch_cands: [K, L_ans, D]
        batch_cands: torch.Tensor = stacked_cand[:, b]
        # batch_scores: [K]
        batch_scores: torch.Tensor = stacked_scores[:, b]

        # Group candidates based on similarity using pairwise MSE/L2 distance
        groups: dict[int, list[int]] = {}
        unique_representatives: list[int] = []
        epsilon: float = 1e-4
        for k in range(K):
            matched: bool = False
            for idx, rep in enumerate(unique_representatives):
                mse: float = float(
                    torch.mean((batch_cands[k] - batch_cands[rep]) ** 2).item()
                )
                if mse < epsilon:
                    groups[idx].append(k)
                    matched = True
                    break
            if not matched:
                new_idx: int = len(unique_representatives)
                unique_representatives.append(k)
                groups[new_idx] = [k]

        # Find the group with the largest number of members
        largest_group_idx: int = max(groups.keys(), key=lambda idx: len(groups[idx]))
        consensus_idx: list[int] = groups[largest_group_idx]

        # Select candidate within the consensus group with the highest score
        best_k: int = consensus_idx[0]
        best_s: torch.Tensor = batch_scores[best_k]
        for k in consensus_idx:
            if batch_scores[k] > best_s:
                best_s = batch_scores[k]
                best_k = k
        best_outputs.append(batch_cands[best_k])

    # [batch_size, l_ans, d_model]
    return torch.stack(best_outputs, dim=0)
