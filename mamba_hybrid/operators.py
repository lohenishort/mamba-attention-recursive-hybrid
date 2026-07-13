from dataclasses import dataclass
from typing import cast, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.attention import PrefixCausalAttention
from mamba_hybrid.ssm import Mamba2SSDScan, SsmState


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


@dataclass
class AttentionCache:
    """Attention K/V cache for incremental autoregressive decoding."""

    key: torch.Tensor  # [B, num_heads, capacity, d_head]
    value: torch.Tensor  # [B, num_heads, capacity, d_head]
    valid_mask: torch.Tensor  # [B, capacity]
    valid_length: int


@dataclass
class HybridLayerCache:
    """Per-layer cache for one MambaAttentionHybridBlock during incremental decoding."""

    attention: AttentionCache
    ssm: SsmState


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
        output_prefix_length: int | None = None,
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
        if output_prefix_length is not None:
            if causal:
                raise ValueError(
                    "output_prefix_length is only supported for non-causal blocks"
                )
            if not 1 <= output_prefix_length <= L:
                raise ValueError("output_prefix_length must be between 1 and seq_len")
        output_length = L if output_prefix_length is None else output_prefix_length

        q_flat: torch.Tensor
        k_flat: torch.Tensor
        v_flat: torch.Tensor
        p_ssm: torch.Tensor
        h_in: torch.Tensor
        h_out: torch.Tensor
        delta: torch.Tensor
        if output_prefix_length is None:
            # [B, L, 7 * D + 264]
            proj: torch.Tensor = self.in_proj(x)
            p_attn: torch.Tensor
            p_attn, p_ssm, h_in, h_out, delta = torch.split(
                proj, [3 * D, 4 * D, 128, 128, 8], dim=-1
            )
            q_flat, k_flat, v_flat = torch.split(p_attn, [D, D, D], dim=-1)
        else:
            # Preserve the existing projection parameter while avoiding query work for
            # rows whose block outputs are not requested.
            q_flat = F.linear(
                x[:, :output_length], self.in_proj.weight[:D]
            )  # [B, output_length, D]
            remaining_proj = F.linear(x, self.in_proj.weight[D:])  # [B, L, 6 * D + 264]
            k_flat, v_flat, p_ssm, h_in, h_out, delta = torch.split(
                remaining_proj, [D, D, 4 * D, 128, 128, 8], dim=-1
            )

        # Attention
        # q: [B, 8, output_length, D // 8]
        q = q_flat.view(B, output_length, 8, D // 8).transpose(1, 2)
        # k, v: [B, 8, L, D // 8]
        k = k_flat.view(B, L, 8, D // 8).transpose(1, 2)
        v = v_flat.view(B, L, 8, D // 8).transpose(1, 2)
        # y_attn shape: [B, output_length, D]
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
        y_ssm = y_ssm[:, :output_length]

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
        return cast(torch.Tensor, x[:, :output_length] + self.out_proj(y_fused))

    def prefill(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        capacity: int = 0,
    ) -> tuple[torch.Tensor, HybridLayerCache]:
        """Process a complete prefix and build a layer cache for incremental steps.

        Uses the forward SSM direction only (causal mode). Returns the full
        block output and a cache containing K/V for all prefix positions plus
        the terminal SSM recurrence state.
        """
        B, L, D = x.shape
        effective_capacity = max(capacity, L)

        # Project the full prefix
        proj: torch.Tensor = self.in_proj(x)
        p_attn: torch.Tensor
        p_ssm: torch.Tensor
        h_in: torch.Tensor
        h_out: torch.Tensor
        delta: torch.Tensor
        p_attn, p_ssm, h_in, h_out, delta = torch.split(
            proj, [3 * D, 4 * D, 128, 128, 8], dim=-1
        )
        q_flat: torch.Tensor
        k_flat: torch.Tensor
        v_flat: torch.Tensor
        q_flat, k_flat, v_flat = torch.split(p_attn, [D, D, D], dim=-1)

        q = q_flat.view(B, L, 8, D // 8).transpose(1, 2)
        k = k_flat.view(B, L, 8, D // 8).transpose(1, 2)
        v = v_flat.view(B, L, 8, D // 8).transpose(1, 2)
        y_attn: torch.Tensor = self.attn_branch(
            q, k, v, causal=True, valid_mask=valid_mask, prefix_length=L
        )

        x_ssm: torch.Tensor
        g_ssm: torch.Tensor
        x_ssm, g_ssm = torch.split(p_ssm, [2 * D, 2 * D], dim=-1)
        y_ssm, terminal_state = self.ssm_branch.prefill(
            x_ssm, g_ssm, h_in, h_out, delta, valid_mask=valid_mask
        )

        # Build attention K/V cache with room for future tokens
        kv_capacity: int = L if effective_capacity <= L else effective_capacity
        k_cache = torch.zeros(B, 8, kv_capacity, D // 8, dtype=k.dtype, device=k.device)
        v_cache = torch.zeros(B, 8, kv_capacity, D // 8, dtype=v.dtype, device=v.device)
        k_cache[:, :, :L] = k
        v_cache[:, :, :L] = v
        if valid_mask is not None:
            cache_mask = torch.zeros(B, kv_capacity, dtype=torch.bool, device=x.device)
            cache_mask[:, :L] = valid_mask
        else:
            cache_mask = torch.zeros(B, kv_capacity, dtype=torch.bool, device=x.device)
            cache_mask[:, :L] = True

        hat_y_attn: torch.Tensor = self.norm_attn(y_attn)
        hat_y_ssm: torch.Tensor = self.norm_ssm(y_ssm)
        y_fused: torch.Tensor = (
            (hat_y_attn * self.beta_1) + (hat_y_ssm * self.beta_2)
        ) / 2
        if self.moe is not None:
            y_fused = self.moe(y_fused)
        x_out = cast(torch.Tensor, x + self.out_proj(y_fused))

        attn_cache = AttentionCache(
            key=k_cache, value=v_cache, valid_mask=cache_mask, valid_length=L
        )
        return x_out, HybridLayerCache(attention=attn_cache, ssm=terminal_state)

    def step(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        cache: HybridLayerCache,
    ) -> tuple[torch.Tensor, HybridLayerCache]:
        """Execute one autoregressive step with cached K/V and SSM state.

        Args:
            x: Token embedding [B, 1, D].
            valid_mask: Optional [B] boolean for the new token.
            cache: Layer cache from a prior prefill or step call.

        Returns:
            (block_output [B, 1, D], updated_cache).
        """
        B, one, D = x.shape
        if one != 1:
            raise ValueError("step expects single-token input [B, 1, D]")
        if valid_mask is not None and valid_mask.shape != (B,):
            raise ValueError("step valid_mask must have shape [batch_size]")

        attn_cache = cache.attention
        pos = attn_cache.valid_length

        # Project the single token
        proj: torch.Tensor = self.in_proj(x)
        p_attn: torch.Tensor
        p_ssm: torch.Tensor
        h_in: torch.Tensor
        h_out: torch.Tensor
        delta: torch.Tensor
        p_attn, p_ssm, h_in, h_out, delta = torch.split(
            proj, [3 * D, 4 * D, 128, 128, 8], dim=-1
        )
        q_flat: torch.Tensor
        k_flat: torch.Tensor
        v_flat: torch.Tensor
        q_flat, k_flat, v_flat = torch.split(p_attn, [D, D, D], dim=-1)

        # Update K/V cache with the new token's projections
        k_single = k_flat.view(B, 1, 8, D // 8).transpose(1, 2)
        v_single = v_flat.view(B, 1, 8, D // 8).transpose(1, 2)
        attn_cache.key[:, :, pos] = k_single[:, :, 0]
        attn_cache.value[:, :, pos] = v_single[:, :, 0]
        if valid_mask is not None:
            attn_cache.valid_mask[:, pos] = valid_mask
        else:
            attn_cache.valid_mask[:, pos] = True

        # Attention: single query against all cached K/V
        q = q_flat.view(B, 1, 8, D // 8).transpose(1, 2)  # [B, 8, 1, d_head]
        active_kv_len = pos + 1
        k_active = attn_cache.key[:, :, :active_kv_len]  # [B, 8, active, d_head]
        v_active = attn_cache.value[:, :, :active_kv_len]
        active_mask = attn_cache.valid_mask[:, :active_kv_len]  # [B, active]
        scores: torch.Tensor = torch.matmul(q, k_active.transpose(-2, -1)) / (
            (D // 8) ** 0.5
        )
        scores = scores.masked_fill(~active_mask[:, None, None, :], float("-inf"))
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        y_attn = torch.matmul(attn_weights, v_active).transpose(1, 2).reshape(B, 1, D)
        y_attn = cast(torch.Tensor, self.attn_branch.out_proj(y_attn))

        # SSM step
        x_ssm: torch.Tensor
        g_ssm: torch.Tensor
        x_ssm, g_ssm = torch.split(p_ssm, [2 * D, 2 * D], dim=-1)
        y_ssm, next_ssm = self.ssm_branch.step(
            x_ssm,
            g_ssm,
            h_in,
            h_out,
            delta,
            cache.ssm,
            valid_mask=valid_mask,
        )

        hat_y_attn: torch.Tensor = self.norm_attn(y_attn)
        hat_y_ssm: torch.Tensor = self.norm_ssm(y_ssm)
        y_fused: torch.Tensor = (
            (hat_y_attn * self.beta_1) + (hat_y_ssm * self.beta_2)
        ) / 2
        if self.moe is not None:
            y_fused = self.moe(y_fused)
        x_out = cast(torch.Tensor, x[:, :1] + self.out_proj(y_fused))

        attn_cache.valid_length = active_kv_len
        return x_out, HybridLayerCache(attention=attn_cache, ssm=next_ssm)
