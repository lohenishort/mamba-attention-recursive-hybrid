from typing import List, cast

import torch
import torch.nn as nn

from mamba_hybrid.answer_update import AnswerUpdateBlock
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.halting import ACTHaltingModule
from mamba_hybrid.planning import PlanningLoop


SUPPORTED_TASKS = frozenset({"MAZE", "SUDOKU", "DIJKSTRA", "GSM8K"})


class MambaAttentionHybrid(nn.Module):
    """Recursive latent planner with explicit answer decoding and ACT heads."""

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_meta = config.n_meta
        self.l_ans = config.l_ans
        self.n_steps = config.n_steps
        self.t_cycles = config.t_cycles
        self.M_min = config.M_min
        self.M_max = config.M_max
        self.M_meta = nn.Parameter(torch.randn(1, self.n_meta, self.d_model))
        self.y_pos_embed = nn.Parameter(torch.randn(1, self.l_ans, self.d_model))
        self.ans_init_proj = nn.Linear(self.d_model, self.d_model)
        self.vocab_decoder = nn.Linear(self.d_model, config.vocab_size)
        self.planning_loop = PlanningLoop(config)
        self.q_head = ACTHaltingModule(config)

    def _validate_tasks(self, task_names: List[str] | None, batch_size: int) -> None:
        if task_names is None:
            return
        if len(task_names) != batch_size:
            raise ValueError("task_names length must match batch size")
        unknown = set(task_names) - SUPPORTED_TASKS
        if unknown:
            raise ValueError(f"unknown task_names: {sorted(unknown)}")

    def init_answer(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Initialize latent answer states from pooled input. [B, L_ans, D]."""
        pooled = x_raw.mean(dim=1)  # [batch_size, d_model]
        answer = self.ans_init_proj(pooled).unsqueeze(1).expand(-1, self.l_ans, -1)
        return cast(torch.Tensor, answer + self.y_pos_embed)

    def decode_answer(self, answer_states: torch.Tensor) -> torch.Tensor:
        """Project latent answer states to vocabulary logits. [B, L_ans, V]."""
        return cast(torch.Tensor, self.vocab_decoder(answer_states))

    def _update_answer(
        self, z: torch.Tensor, y: torch.Tensor, task_names: List[str] | None
    ) -> torch.Tensor:
        blocks = self.planning_loop.answer_update_blocks
        if self.config.use_moe and blocks is not None:
            names = task_names if task_names is not None else ["MAZE"] * y.shape[0]
            outputs: list[torch.Tensor] = []
            for index, task in enumerate(names):
                task_block = blocks[task]
                assert isinstance(task_block, AnswerUpdateBlock)
                outputs.append(task_block(z[index : index + 1], y[index : index + 1]))
            return torch.cat(outputs, dim=0)
        answer_block = self.planning_loop.answer_update_block
        if answer_block is None:
            return y
        return cast(torch.Tensor, answer_block(z, y))

    def _initialize(
        self, x_raw: torch.Tensor, task_names: List[str] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_raw.shape[0]
        self._validate_tasks(task_names, batch_size)
        z = self.M_meta.expand(batch_size, -1, -1)
        y = self.init_answer(x_raw)
        for _ in range(1, self.t_cycles):
            z, y = self.planning_loop(x_raw, z, y, warmup=True, task_names=task_names)
        return z, y

    def forward(
        self, x_raw: torch.Tensor, task_names: List[str] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return vocabulary logits and per-step BCE halting probabilities."""
        z, y = self._initialize(x_raw, task_names)
        batch_size = x_raw.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=x_raw.device)
        probabilities: list[torch.Tensor] = []
        max_steps = self.n_steps if self.training else self.M_max

        for step in range(1, max_steps + 1):
            previous_z, previous_y = z, y
            x_concat = torch.cat([z, y, x_raw], dim=1)
            next_z = self.planning_loop.planning_block(
                x_concat, causal=False, task_names=task_names
            )[:, : self.n_meta, :]
            if self.training and torch.rand(1).item() < 0.15:
                next_z = (
                    next_z + torch.randn_like(next_z) * torch.rand(1).item() * 0.025
                )
            next_y = self._update_answer(next_z, y, task_names)
            if self.training:
                z, y = next_z, next_y
            else:
                mask = active[:, None, None]
                z = torch.where(mask, next_z, previous_z)
                y = torch.where(mask, next_y, previous_y)
            probability = self.q_head(z, y)
            probabilities.append(probability)
            if not self.training and step >= self.M_min:
                active = active & (probability < self.config.halt_threshold)
                if not bool(active.any()):
                    break

        return self.decode_answer(y), probabilities

    def forward_q(
        self, x_raw: torch.Tensor, task_names: List[str] | None = None
    ) -> tuple[
        torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]
    ]:
        """Return logits and the exact state/Q trajectory used by Q-learning."""
        z, y = self._initialize(x_raw, task_names)
        if self.training:
            num_steps = int(torch.randint(self.M_min, self.M_max + 1, (1,)).item())
        else:
            num_steps = self.M_max
        states: list[tuple[torch.Tensor, torch.Tensor]] = []
        predictions: list[torch.Tensor] = []
        for _ in range(num_steps):
            x_concat = torch.cat([z, y, x_raw], dim=1)
            z = self.planning_loop.planning_block(
                x_concat, causal=False, task_names=task_names
            )[:, : self.n_meta, :]
            y = self._update_answer(z, y, task_names)
            states.append((z, y))
            predictions.append(self.q_head.get_q_values(z, y))
        return self.decode_answer(y), states, predictions
