import math
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from mamba_hybrid.evaluation import select_consensus_array
from mamba_hybrid.model import MambaAttentionHybrid


def select_consensus(logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Select per-batch logits by exact-token majority, then score ties. [K,B,L,V]."""
    if logits.ndim != 4:
        raise ValueError("logits must have shape [rollouts, batch, length, vocab]")
    rollouts, batch_size, _, _ = logits.shape
    if scores.shape != (rollouts, batch_size):
        raise ValueError("scores must have shape [rollouts, batch]")
    token_ids = cast(
        npt.NDArray[np.int64],
        logits.argmax(dim=-1).detach().cpu().contiguous().numpy(),
    )
    score_values = cast(
        npt.NDArray[np.float32],
        scores.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy(),
    )
    selected = select_consensus_array(token_ids, score_values)
    rollout_indices = torch.tensor(selected, dtype=torch.long, device=logits.device)
    batch_indices = torch.arange(batch_size, device=logits.device)
    return logits[rollout_indices, batch_indices]


def ptrm_inference(
    input_ids: torch.Tensor,
    model: MambaAttentionHybrid,
    K: int = 5,
    sigma_base: float = 0.05,
    max_noise_step: int = 20,
    task_names: list[str] | None = None,
    x_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run batched stochastic rollouts and vote on exact decoded token sequences."""
    if K <= 0:
        raise ValueError("K must be positive")
    if sigma_base < 0.0 or max_noise_step < 0:
        raise ValueError("noise parameters must be non-negative")
    if K == 1:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits, _ = model(input_ids, task_names=task_names, x_mask=x_mask)
            return cast(torch.Tensor, logits)
        finally:
            model.train(was_training)
    candidates, scores = ptrm_state_rollouts(
        input_ids,
        model,
        K=K,
        sigma_base=sigma_base,
        max_noise_step=max_noise_step,
        task_names=task_names,
        x_mask=x_mask,
    )
    rollouts, batch_size, answer_length, d_model = candidates.shape
    logits = model.decode_answer(
        candidates.reshape(rollouts * batch_size, answer_length, d_model)
    ).reshape(rollouts, batch_size, answer_length, model.config.vocab_size)
    return select_consensus(logits, scores)


def ptrm_state_rollouts(
    input_ids: torch.Tensor,
    model: MambaAttentionHybrid,
    K: int = 5,
    sigma_base: float = 0.05,
    max_noise_step: int = 20,
    task_names: list[str] | None = None,
    x_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched stochastic answer memories and ACT scores. [K,B,L,D], [K,B]."""
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
            expanded_input = (
                input_ids.unsqueeze(0)
                .expand(K, -1, -1, -1)
                .reshape(K * batch_size, input_ids.shape[1], input_ids.shape[2])
            )
            expanded_tasks = task_names * K if task_names is not None else None
            expanded_mask = None
            if x_mask is not None:
                if x_mask.shape != input_ids.shape[:2]:
                    raise ValueError("x_mask must match input sequence dimensions")
                expanded_mask = (
                    x_mask.unsqueeze(0)
                    .expand(K, -1, -1)
                    .reshape(K * batch_size, x_mask.shape[1])
                )
            z = model.M_meta.expand(K * batch_size, -1, -1)
            y = model.init_answer(expanded_input, expanded_mask)
            for cycle in range(1, model.M_max + 1):
                for step in range(1, model.n_steps + 1):
                    global_step = (cycle - 1) * model.n_steps + step
                    if global_step <= max_noise_step:
                        fraction = 1.0 - ((step - 1) / model.n_steps)
                        z = z + torch.randn_like(z) * sigma_base * math.sqrt(fraction)
                    x_concat = torch.cat([z, y, expanded_input], dim=1)
                    valid_mask = None
                    if expanded_mask is not None:
                        prefix_mask = torch.ones(
                            K * batch_size,
                            z.shape[1] + y.shape[1],
                            dtype=torch.bool,
                            device=input_ids.device,
                        )
                        valid_mask = torch.cat([prefix_mask, expanded_mask], dim=1)
                    z = model.planning_loop.planning_block(
                        x_concat,
                        causal=False,
                        task_names=expanded_tasks,
                        valid_mask=valid_mask,
                    )[:, : model.n_meta, :]
                y = model.planning_loop.update_answer(z, y, expanded_tasks)

            candidates = y.reshape(K, batch_size, model.l_ans, model.d_model)
            scores = model.q_head(z, y).reshape(K, batch_size)
        return candidates, scores
    finally:
        model.train(was_training)


__all__ = ["ptrm_inference", "ptrm_state_rollouts", "select_consensus"]
