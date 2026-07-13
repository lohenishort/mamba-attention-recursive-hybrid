import math
from typing import cast

import torch

from mamba_hybrid.model import MambaAttentionHybrid


def select_consensus(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Select per-batch logits by exact-token majority, then score ties. [K,B,L,V]."""
    rollouts, batch_size, _, _ = logits.shape
    token_ids = logits.argmax(dim=-1)
    selected: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        groups: dict[tuple[int, ...], list[int]] = {}
        for rollout in range(rollouts):
            key = tuple(
                int(token) for token in token_ids[rollout, batch_index].tolist()
            )
            groups.setdefault(key, []).append(rollout)
        largest_size = max(len(indices) for indices in groups.values())
        eligible = [
            rollout
            for indices in groups.values()
            if len(indices) == largest_size
            for rollout in indices
        ]
        best = max(
            eligible,
            key=lambda rollout: float(scores[rollout, batch_index].item()),
        )
        selected.append(logits[best, batch_index])
    return torch.stack(selected, dim=0)


def ptrm_inference(
    input_ids: torch.Tensor,
    model: MambaAttentionHybrid,
    K: int = 5,
    sigma_base: float = 0.05,
    max_noise_step: int = 20,
    task_names: list[str] | None = None,
) -> torch.Tensor:
    """Run batched stochastic rollouts and vote on exact decoded token sequences."""
    if K <= 0:
        raise ValueError("K must be positive")
    if sigma_base < 0.0 or max_noise_step < 0:
        raise ValueError("noise parameters must be non-negative")
    batch_size = input_ids.shape[0]
    model._validate_tasks(task_names, batch_size)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if K == 1:
                logits, _ = model(input_ids, task_names=task_names)
                return cast(torch.Tensor, logits)

            expanded_input = (
                input_ids.unsqueeze(0)
                .expand(K, -1, -1, -1)
                .reshape(K * batch_size, input_ids.shape[1], input_ids.shape[2])
            )
            expanded_tasks = task_names * K if task_names is not None else None
            z = model.M_meta.expand(K * batch_size, -1, -1)
            y = model.init_answer(expanded_input)
            for cycle in range(1, model.t_cycles + 1):
                for step in range(1, model.n_steps + 1):
                    global_step = (cycle - 1) * model.n_steps + step
                    if global_step <= max_noise_step:
                        fraction = 1.0 - (step / model.n_steps)
                        z = z + torch.randn_like(z) * sigma_base * math.sqrt(fraction)
                    x_concat = torch.cat([z, y, expanded_input], dim=1)
                    z = model.planning_loop.planning_block(
                        x_concat, causal=False, task_names=expanded_tasks
                    )[:, : model.n_meta, :]
                y = model._update_answer(z, y, expanded_tasks)

            logits = model.decode_answer(y).reshape(
                K, batch_size, model.l_ans, model.config.vocab_size
            )
            scores = model.q_head(z, y).reshape(K, batch_size)

        return select_consensus(logits, scores)
    finally:
        model.train(was_training)
