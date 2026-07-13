from typing import List, cast

import torch
import torch.nn as nn

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.halting import ACTHaltingModule
from mamba_hybrid.planning import PlanningLoop

SUPPORTED_TASKS = frozenset({"MAZE", "SUDOKU", "DIJKSTRA", "GSM8K"})
PlanningState = tuple[torch.Tensor, torch.Tensor]


class MambaAttentionHybrid(nn.Module):
    """Recursive planner with cycle-level adaptive computation."""

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

    def init_answer(
        self, x_raw: torch.Tensor, x_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Initialize aligned slots directly and pool unaligned inputs. [B,L_ans,D]."""
        if x_raw.shape[1] == self.l_ans:
            return cast(torch.Tensor, self.ans_init_proj(x_raw) + self.y_pos_embed)
        if x_mask is None:
            pooled = x_raw.mean(dim=1)
        else:
            if x_mask.shape != x_raw.shape[:2]:
                raise ValueError("x_mask must match the first two x_raw dimensions")
            weights = x_mask.to(device=x_raw.device, dtype=x_raw.dtype).unsqueeze(-1)
            pooled = (x_raw * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        answer = self.ans_init_proj(pooled).unsqueeze(1).expand(-1, self.l_ans, -1)
        return cast(torch.Tensor, answer + self.y_pos_embed)

    def decode_answer(self, answer_states: torch.Tensor) -> torch.Tensor:
        """Project latent answer memory to compatibility logits. [B,L_ans,V]."""
        return cast(torch.Tensor, self.vocab_decoder(answer_states))

    def build_memory_prefix(
        self,
        x_raw: torch.Tensor,
        state: PlanningState,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build printer memory as [meta, answer, raw context] and its mask."""
        z, y = state
        prefix = torch.cat([z, y, x_raw], dim=1)
        prefix_mask = torch.ones(
            x_raw.shape[0],
            z.shape[1] + y.shape[1],
            dtype=torch.bool,
            device=x_raw.device,
        )
        raw_mask = (
            torch.ones(x_raw.shape[:2], dtype=torch.bool, device=x_raw.device)
            if x_mask is None
            else x_mask.to(device=x_raw.device, dtype=torch.bool)
        )
        return prefix, torch.cat([prefix_mask, raw_mask], dim=1)

    def _initialize(
        self,
        x_raw: torch.Tensor,
        task_names: List[str] | None,
        x_mask: torch.Tensor | None,
    ) -> PlanningState:
        batch_size = x_raw.shape[0]
        self._validate_tasks(task_names, batch_size)
        if x_mask is not None and x_mask.shape != x_raw.shape[:2]:
            raise ValueError("x_mask must match the first two x_raw dimensions")
        z = self.M_meta.expand(batch_size, -1, -1)
        y = self.init_answer(x_raw, x_mask)
        return z, y

    def forward_state_trajectory(
        self,
        x_raw: torch.Tensor,
        task_names: List[str] | None = None,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[list[PlanningState], list[torch.Tensor]]:
        """Return one planning state and halt probability per completed cycle."""
        z, y = self._initialize(x_raw, task_names, x_mask)
        batch_size = x_raw.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=x_raw.device)
        states: list[PlanningState] = []
        probabilities: list[torch.Tensor] = []

        for cycle_index in range(self.M_max):
            previous_z, previous_y = z, y
            next_z, next_y = self.planning_loop(
                x_raw,
                z,
                y,
                warmup=False,
                task_names=task_names,
                x_mask=x_mask,
                cycle_index=cycle_index,
            )
            if self.training:
                z, y = next_z, next_y
            else:
                state_mask = active[:, None, None]
                z = torch.where(state_mask, next_z, previous_z)
                y = torch.where(state_mask, next_y, previous_y)
            probability = self.q_head(z, y)
            states.append((z, y))
            probabilities.append(probability)
            cycle = cycle_index + 1
            if not self.training and cycle >= self.M_min:
                active = active & (probability < self.config.halt_threshold)
                if not bool(active.any()):
                    break
        return states, probabilities

    def forward_states(
        self,
        x_raw: torch.Tensor,
        task_names: List[str] | None = None,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return final answer memory and per-cycle halt probabilities."""
        states, probabilities = self.forward_state_trajectory(x_raw, task_names, x_mask)
        return states[-1][1], probabilities

    def forward(
        self,
        x_raw: torch.Tensor,
        task_names: List[str] | None = None,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return compatibility logits and per-cycle halt probabilities."""
        answer_states, probabilities = self.forward_states(x_raw, task_names, x_mask)
        return self.decode_answer(answer_states), probabilities

    def forward_q(
        self,
        x_raw: torch.Tensor,
        task_names: List[str] | None = None,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[PlanningState], list[torch.Tensor]]:
        """Return final logits and a cycle-level state/Q trajectory."""
        z, y = self._initialize(x_raw, task_names, x_mask)
        if self.training:
            num_cycles = int(
                torch.randint(
                    self.M_min, self.M_max + 1, (1,), device=x_raw.device
                ).item()
            )
        else:
            num_cycles = self.M_max
        states: list[PlanningState] = []
        predictions: list[torch.Tensor] = []
        for cycle_index in range(num_cycles):
            z, y = self.planning_loop(
                x_raw,
                z,
                y,
                warmup=False,
                task_names=task_names,
                x_mask=x_mask,
                cycle_index=cycle_index,
            )
            states.append((z, y))
            predictions.append(self.q_head.get_q_values(z, y))
        return self.decode_answer(y), states, predictions
