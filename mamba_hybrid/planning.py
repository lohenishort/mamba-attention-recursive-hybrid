import contextlib

import torch
import torch.nn as nn
from typing import List
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.operators import MambaAttentionHybridBlock
from mamba_hybrid.answer_update import AnswerUpdateBlock


class PlanningLoop(nn.Module):
    """
    PlanningLoop module that executes recurrent steps, updates state z over n steps,
    and calls the cross-attention AnswerUpdateBlock at the end of the cycle.
    """

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.config: MambaHybridConfig = config
        self.n_steps: int = config.n_steps
        self.n_meta: int = config.n_meta
        self.planning_block: MambaAttentionHybridBlock = MambaAttentionHybridBlock(
            config
        )
        
        self.answer_update_block: AnswerUpdateBlock | None = None
        self.answer_update_blocks: nn.ModuleDict | None = None
        if config.use_moe:
            self.answer_update_blocks = nn.ModuleDict(
                {
                    "MAZE": AnswerUpdateBlock(config),
                    "SUDOKU": AnswerUpdateBlock(config),
                    "DIJKSTRA": AnswerUpdateBlock(config),
                    "GSM8K": AnswerUpdateBlock(config),
                }
            )
        else:
            self.answer_update_block = AnswerUpdateBlock(config)

    def forward(
        self,
        x_raw: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        warmup: bool = True,
        task_names: List[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the planning loop cycle.

        Args:
            x_raw: Raw input context of shape [B, L_seq, D]
            z: Latent state of shape [B, N_meta, D]
            y: Answer state of shape [B, L_ans, D]
            warmup: If True, executes under torch.no_grad() to propagate state without storing gradients.

        Returns:
            A tuple of (z_final, y_final):
                - z_final: Updated latent state of shape [B, N_meta, D]
                - y_final: Updated answer state of shape [B, L_ans, D]
        """
        # [batch_size, seq_len, d_model]
        # z: [batch_size, n_meta, d_model]
        # y: [batch_size, l_ans, d_model]
        context_manager = torch.no_grad() if warmup else contextlib.nullcontext()
        with context_manager:
            # Run one complete cycle (n latent updates + 1 answer update)
            for _ in range(1, self.n_steps + 1):
                # Concatenate along the sequence dimension: [B, N_meta + L_ans + L_seq, D]
                X_concat = torch.cat([z, y, x_raw], dim=1)
                # Apply non-causal attention to propagate constraints globally
                # Slice out the updated latent state z: [B, N_meta, D]
                z = self.planning_block(
                    X_concat, causal=False, task_names=task_names
                )[:, : self.n_meta, :]
            # Update the answer state at the end of the cycle: [B, L_ans, D]
            if self.config.use_moe and self.answer_update_blocks is not None:
                if task_names is not None:
                    y_list = []
                    for i in range(y.shape[0]):
                        task = task_names[i]
                        if task not in self.answer_update_blocks:
                            task = "MAZE"
                        block = self.answer_update_blocks[task]
                        assert isinstance(block, AnswerUpdateBlock)
                        y_list.append(block(z[i : i + 1], y[i : i + 1]))
                    y = torch.cat(y_list, dim=0)
                else:
                    block = self.answer_update_blocks["MAZE"]
                    assert isinstance(block, AnswerUpdateBlock)
                    y = block(z, y)
            elif self.answer_update_block is not None:
                y = self.answer_update_block(z, y)
        return z, y
