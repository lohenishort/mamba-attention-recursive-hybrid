from typing import List
import torch
import torch.nn as nn
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.planning import PlanningLoop
from mamba_hybrid.halting import ACTHaltingModule


class MambaAttentionHybrid(nn.Module):
    """
    Core Mamba-Attention Recursive Reasoning Hybrid model that coordinates
    warmup planning loops (T-1 cycles with gradients disabled) and the final
    supervision cycle (T cycle with gradients enabled) along with the ACT halting module.
    """

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.config: MambaHybridConfig = config
        self.d_model: int = config.d_model
        self.n_meta: int = config.n_meta
        self.l_ans: int = config.l_ans
        self.n_steps: int = config.n_steps
        self.t_cycles: int = config.t_cycles
        self.M_min: int = config.M_min
        self.M_max: int = config.M_max

        # Learned initial planning state meta-tokens
        self.M_meta: nn.Parameter = nn.Parameter(
            torch.randn(1, self.n_meta, self.d_model)
        )

        # Positional embedding for the answer sequence y to break symmetry
        self.y_pos_embed: nn.Parameter = nn.Parameter(
            torch.randn(1, self.l_ans, self.d_model)
        )

        # Projection layer to initialize answer representation from pooled input context
        self.ans_init_proj: nn.Linear = nn.Linear(self.d_model, self.d_model)

        self.planning_loop: PlanningLoop = PlanningLoop(config)
        self.q_head: ACTHaltingModule = ACTHaltingModule(config)

    def init_answer(self, X_raw: torch.Tensor) -> torch.Tensor:
        """
        Projects average pooled raw input context to initialize the answer state y.

        Args:
            X_raw: Raw input context of shape [B, L_raw, D]

        Returns:
            ans_init: Initialized answer state of shape [B, L_ans, D]
        """
        # X_raw: [batch_size, seq_len, d_model]
        pooled: torch.Tensor = X_raw.mean(dim=1)  # [batch_size, d_model]
        ans_init: torch.Tensor = (
            self.ans_init_proj(pooled).unsqueeze(1).expand(-1, self.l_ans, -1)
        )  # [batch_size, l_ans, d_model]
        ans_init = ans_init + self.y_pos_embed
        return ans_init

    def forward(
        self, X_raw: torch.Tensor, task_names: List[str] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass coordinating warmup loops and the final supervision loop.

        Args:
            X_raw: Raw input context of shape [B, L_raw, D]
            task_names: Optional task prefix names for MoE routing.

        Returns:
            y_final: Final updated answer prediction state of shape [B, L_ans, D]
            bce_probs: List of halting probabilities from the ACT head for each step in cycle T
        """
        # X_raw: [batch_size, seq_len, d_model]
        B, L_raw, D = X_raw.shape
        z: torch.Tensor = self.M_meta.expand(B, -1, -1)  # [batch_size, n_meta, d_model]
        y: torch.Tensor = self.init_answer(X_raw)  # [batch_size, l_ans, d_model]

        # Warmup phase (T-1 cycles, no grad)
        for c in range(1, self.t_cycles):
            z, y = self.planning_loop(X_raw, z, y, warmup=True, task_names=task_names)

        # Supervision cycle (T cycle, grad enabled)

        bce_probs: list[torch.Tensor] = []
        for i in range(1, self.n_steps + 1):
            # Execute one latent step within cycle T with gradients
            X_concat: torch.Tensor = torch.cat(
                [z, y, X_raw], dim=1
            )  # [batch_size, n_meta + l_ans + seq_len, d_model]
            z = self.planning_loop.planning_block(
                X_concat, causal=False, task_names=task_names
            )[:, : self.n_meta, :]  # [batch_size, n_meta, d_model]

            # Regularization training noise
            if self.training and torch.rand(1).item() < 0.15:
                # Add training-only regularization noise
                noise: torch.Tensor = torch.randn_like(z) * torch.rand(1).item() * 0.025
                z = z + noise

            bce_prob: torch.Tensor = self.q_head(z, y)  # [batch_size]
            bce_probs.append(bce_prob)

        if self.planning_loop.config.use_moe and self.planning_loop.answer_update_blocks is not None:
            y_list = []
            for i in range(y.shape[0]):
                task = task_names[i] if task_names is not None else "MAZE"
                if task not in self.planning_loop.answer_update_blocks:
                    task = "MAZE"
                y_list.append(self.planning_loop.answer_update_blocks[task](z[i : i + 1], y[i : i + 1]))
            y_final: torch.Tensor = torch.cat(y_list, dim=0)
        elif self.planning_loop.answer_update_block is not None:
            y_final = self.planning_loop.answer_update_block(z, y)
        else:
            y_final = y
        return y_final, bce_probs

    def forward_q(
        self, X_raw: torch.Tensor, task_names: List[str] | None = None
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
    ]:
        """Forward pass coordinating warmup loops and final supervision loop for Q-learning.

        Args:
            X_raw: Raw input context of shape [B, L_raw, D]
            task_names: Optional task prefix names for MoE routing.

        Returns:
            y_final: Final updated answer prediction state of shape [B, L_ans, D]
            states: List of tuples (z, y) at each step of the supervision cycle.
            q_preds: List of Q-value predictions from the ACT head for each step.
        """
        # X_raw: [batch_size, seq_len, d_model]
        B, L_raw, D = X_raw.shape
        z: torch.Tensor = self.M_meta.expand(B, -1, -1)  # [batch_size, n_meta, d_model]
        y: torch.Tensor = self.init_answer(X_raw)  # [batch_size, l_ans, d_model]

        # Warmup phase (T-1 cycles, no grad)
        for c in range(1, self.t_cycles):
            z, y = self.planning_loop(X_raw, z, y, warmup=True, task_names=task_names)

        # Supervision cycle (T cycle, grad enabled)
        if self.training:
            num_steps: int = int(torch.randint(self.M_min, self.M_max + 1, (1,)).item())
        else:
            num_steps = self.n_steps

        q_preds: list[torch.Tensor] = []
        states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(1, num_steps + 1):
            X_concat: torch.Tensor = torch.cat(
                [z, y, X_raw], dim=1
            )  # [batch_size, n_meta + l_ans + seq_len, d_model]
            z = self.planning_loop.planning_block(
                X_concat, causal=False, task_names=task_names
            )[:, : self.n_meta, :]  # [batch_size, n_meta, d_model]

            q_vals: torch.Tensor = self.q_head.get_q_values(z, y)  # [batch_size, 2]
            q_preds.append(q_vals)
            states.append((z, y))

        if self.planning_loop.config.use_moe and self.planning_loop.answer_update_blocks is not None:
            y_list = []
            for i in range(y.shape[0]):
                task = task_names[i] if task_names is not None else "MAZE"
                if task not in self.planning_loop.answer_update_blocks:
                    task = "MAZE"
                y_list.append(self.planning_loop.answer_update_blocks[task](z[i : i + 1], y[i : i + 1]))
            y_final: torch.Tensor = torch.cat(y_list, dim=0)
        elif self.planning_loop.answer_update_block is not None:
            y_final = self.planning_loop.answer_update_block(z, y)
        else:
            y_final = y
        return y_final, states, q_preds
