from typing import cast
import torch
import torch.nn as nn
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.operators import RMSNorm


class AnswerUpdateBlock(nn.Module):
    """
    AnswerUpdateBlock cross-attention module that maps latent states
    z_n and answer states y_{c-1} to the updated answer state y_c.
    """

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.d_model: int = config.d_model

        # Projection layers for Queries, Keys, Values
        self.q_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)

        # RMSNorm fallback from mamba_hybrid.operators
        self.norm_y: RMSNorm = RMSNorm(self.d_model)
        self.norm_z: RMSNorm = RMSNorm(self.d_model)

    def forward(self, z: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of AnswerUpdateBlock.

        Args:
            z: Latent states of shape [B, N_meta, D]
            y_prev: Previous answer states of shape [B, L_ans, D]

        Returns:
            Updated answer states of shape [B, L_ans, D]
        """
        # z: [batch_size, n_meta, d_model]
        # y_prev: [batch_size, l_ans, d_model]
        B: int
        L_ans: int
        D: int
        B, L_ans, D = y_prev.shape
        N_meta: int = z.shape[1]

        y_norm: torch.Tensor = self.norm_y(y_prev)  # [B, L_ans, D]
        z_norm: torch.Tensor = self.norm_z(z)  # [B, N_meta, D]

        # Project and reshape for multi-head cross-attention (8 heads)
        # q: [B, 8, L_ans, D // 8]
        q: torch.Tensor = self.q_proj(y_norm).view(B, L_ans, 8, D // 8).transpose(1, 2)
        # k: [B, 8, N_meta, D // 8]
        k: torch.Tensor = self.k_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)
        # v: [B, 8, N_meta, D // 8]
        v: torch.Tensor = self.v_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)

        # Compute scaled dot-product attention
        # scores: [B, 8, L_ans, N_meta]
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / ((D // 8) ** 0.5)
        # attn_weights: [B, 8, L_ans, N_meta]
        attn_weights: torch.Tensor = torch.softmax(scores, dim=-1)

        # y_attn: [B, L_ans, D]
        y_attn: torch.Tensor = (
            torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L_ans, D)
        )

        # Out projection and residual connection
        # return: [B, L_ans, D]
        return cast(torch.Tensor, y_prev + self.out_proj(y_attn))
