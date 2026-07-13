from typing import cast, List
import torch
import torch.nn as nn
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.attention import PrefixCausalAttention
from mamba_hybrid.ssm import Mamba2SSDScan


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) for PyTorch 2.0.0+ compatibility.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        variance: torch.Tensor = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # return: [..., d_model]
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight


class TaskPrefixedMoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer routed deterministically by task prefix.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model: int = d_model

        # FFN expert MLPs for each task
        self.experts: nn.ModuleDict = nn.ModuleDict(
            {
                "MAZE": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.SiLU(),
                    nn.Linear(4 * d_model, d_model),
                ),
                "SUDOKU": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.SiLU(),
                    nn.Linear(4 * d_model, d_model),
                ),
                "DIJKSTRA": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.SiLU(),
                    nn.Linear(4 * d_model, d_model),
                ),
                "GSM8K": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.SiLU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }
        )

    def forward(
        self, x: torch.Tensor, task_names: List[str] | None = None
    ) -> torch.Tensor:
        # x shape: [B, L, D]
        B, L, D = x.shape

        if task_names is None:
            # Fallback to MAZE expert if no task_names are provided (e.g. in tests/single-task runs)
            task_names = ["MAZE"] * B
        if len(task_names) != B:
            raise ValueError("task_names length must match batch size")
        unknown = set(task_names) - set(self.experts.keys())
        if unknown:
            raise ValueError(f"unknown task_names: {sorted(unknown)}")
        if len(set(task_names)) == 1:
            return cast(torch.Tensor, self.experts[task_names[0]](x))

        out_list = []
        for i in range(B):
            task = task_names[i]
            expert = self.experts[task]
            out_sample = expert(x[i])
            out_list.append(out_sample)

        return torch.stack(out_list, dim=0)


class MambaAttentionHybridBlock(nn.Module):
    """
    Hybrid block that combines the attention branch and the SSM branch
    using RMSNorm, learned scaling parameters (beta_1 and beta_2),
    a shared output projection layer, and a residual connection.
    """

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.config: MambaHybridConfig = config
        self.d_model: int = config.d_model

        # Projections
        # q, k, v (3*D) + ssm features/gates (2*2*D) + h_in/h_out (2*8*16) + delta (8)
        total_proj_dim: int = 3 * self.d_model + 4 * self.d_model + 2 * 8 * 16 + 8
        self.in_proj: nn.Linear = nn.Linear(self.d_model, total_proj_dim, bias=False)

        self.attn_branch: PrefixCausalAttention = PrefixCausalAttention(config)
        self.ssm_branch: Mamba2SSDScan = Mamba2SSDScan(
            self.d_model,
            expansion=2,
            num_heads=8,
            d_state=16,
            use_cuda_kernels=config.use_cuda_kernels,
        )

        self.beta_1: nn.Parameter = nn.Parameter(torch.ones(self.d_model))
        self.beta_2: nn.Parameter = nn.Parameter(torch.ones(self.d_model))
        self.norm_attn: RMSNorm = RMSNorm(self.d_model)
        self.norm_ssm: RMSNorm = RMSNorm(self.d_model)
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)

        # Mixture of Experts FFN layer
        self.moe: TaskPrefixedMoeLayer | None = None
        if config.use_moe:
            self.moe = TaskPrefixedMoeLayer(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        task_names: List[str] | None = None,
        valid_mask: torch.Tensor | None = None,
        prefix_length: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the hybrid block.

        Args:
            x: Input feature tensor of shape [B, L, D]
            causal: Whether to apply prefix-causal attention masking.
            task_names: Optional task prefix names for MoE routing.

        Returns:
            Output tensor of shape [B, L, D]
        """
        B, L, D = x.shape  # B: batch_size, L: seq_len, D: d_model

        # proj shape: [B, L, total_proj_dim]
        proj: torch.Tensor = self.in_proj(x)

        # Slice projection channels
        # split_dims: [3 * D, 4 * D, 128, 128, 8]
        # - 3 * D: attention projections (q, k, v)
        # - 4 * D: SSM projections (x_ssm, gate_ssm) where expansion = 2
        # - 128: h_in state projection (8 heads * 16 state dim)
        # - 128: h_out state projection (8 heads * 16 state dim)
        # - 8: delta step sizes (8 heads)
        split_dims = [3 * D, 4 * D, 128, 128, 8]
        p_attn: torch.Tensor
        p_ssm: torch.Tensor
        h_in: torch.Tensor
        h_out: torch.Tensor
        delta: torch.Tensor
        p_attn, p_ssm, h_in, h_out, delta = torch.split(proj, split_dims, dim=-1)

        # Attention
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = torch.split(p_attn, [D, D, D], dim=-1)
        # Reshape to [B, num_heads, L, d_head] where num_heads = 8, d_head = D // 8
        q = q.view(B, L, 8, D // 8).transpose(1, 2)
        k = k.view(B, L, 8, D // 8).transpose(1, 2)
        v = v.view(B, L, 8, D // 8).transpose(1, 2)
        # y_attn shape: [B, L, D]
        y_attn: torch.Tensor = self.attn_branch(
            q,
            k,
            v,
            causal=causal,
            valid_mask=valid_mask,
            prefix_length=prefix_length,
        )

        # SSM
        x_ssm: torch.Tensor
        g_ssm: torch.Tensor
        x_ssm, g_ssm = torch.split(p_ssm, [2 * D, 2 * D], dim=-1)
        # y_ssm shape: [B, L, D]
        y_ssm: torch.Tensor = self.ssm_branch(
            x_ssm, g_ssm, h_in, h_out, delta, valid_mask=valid_mask
        )
        if not causal:
            backward_ssm = self.ssm_branch(
                x_ssm.flip(1),
                g_ssm.flip(1),
                h_in.flip(1),
                h_out.flip(1),
                delta.flip(1),
                valid_mask=None if valid_mask is None else valid_mask.flip(1),
            ).flip(1)
            y_ssm = (y_ssm + backward_ssm) * 0.5

        # Fusion
        # Apply RMSNorm to both branches
        hat_y_attn: torch.Tensor = self.norm_attn(y_attn)
        hat_y_ssm: torch.Tensor = self.norm_ssm(y_ssm)

        # Scale with beta_1 and beta_2 and average
        # y_fused shape: [B, L, D]
        y_fused: torch.Tensor = (
            (hat_y_attn * self.beta_1) + (hat_y_ssm * self.beta_2)
        ) / 2

        # Apply Mixture of Experts FFN if enabled
        if self.moe is not None:
            y_fused = self.moe(y_fused, task_names)

        # Final projection and residual connection
        # return shape: [B, L, D]
        return cast(torch.Tensor, x + self.out_proj(y_fused))
