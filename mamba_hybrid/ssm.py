from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore

    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
    mamba_chunk_scan_combined = None


@dataclass
class SsmState:
    """Recurrent state for a pure-PyTorch Mamba-2 SSD scan."""

    state: torch.Tensor  # [B, num_heads, d_state, d_head]


class Mamba2SSDScan(nn.Module):
    """
    Mamba-2 Structured State Duality (SSD) scan in pure PyTorch.
    This serves as the standard sequential CPU/GPU compatible fallback.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        num_heads: int = 8,
        d_state: int = 16,
        use_cuda_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.expansion: int = expansion
        self.num_heads: int = num_heads
        self.d_state: int = d_state
        self.ssm_dim: int = d_model * expansion
        self.use_cuda_kernels: bool = use_cuda_kernels

        # Output projection to project ssm_dim back to d_model
        self.out_proj: nn.Linear = nn.Linear(self.ssm_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, ssm_dim]
        gate: torch.Tensor,  # [B, L, ssm_dim]
        h_in: torch.Tensor,  # [B, L, num_heads * d_state]
        h_out: torch.Tensor,  # [B, L, num_heads * d_state]
        delta: torch.Tensor,  # [B, L, num_heads]
        valid_mask: torch.Tensor | None = None,  # [B, L]
    ) -> torch.Tensor:  # [B, L, d_model]
        """
        Computes the sequential Structured State Duality (SSD) scan.

        Args:
            x: Input feature tensor of shape [B, L, ssm_dim]
            gate: Gate tensor of shape [B, L, ssm_dim]
            h_in: Input state projection tensor of shape [B, L, num_heads * d_state]
            h_out: Output state projection tensor of shape [B, L, num_heads * d_state]
            delta: Step sizes tensor of shape [B, L, num_heads]

        Returns:
            Output tensor of shape [B, L, d_model]
        """
        B, L, _ = x.shape
        if valid_mask is not None and valid_mask.shape != (B, L):
            raise ValueError("valid_mask must have shape [batch_size, seq_len]")

        # Positive continuous-time step sizes: [B, L, num_heads]
        delta_pos: torch.Tensor = torch.nn.functional.softplus(delta)

        # Split input and output state projections: [B, L, num_heads, d_state]
        h_in_split: torch.Tensor = h_in.view(B, L, self.num_heads, self.d_state)
        h_out_split: torch.Tensor = h_out.view(B, L, self.num_heads, self.d_state)

        # Head dimension: ssm_dim // num_heads
        d_head: int = self.ssm_dim // self.num_heads
        x_split: torch.Tensor = x.view(B, L, self.num_heads, d_head)

        can_use_cuda_kernels = self.use_cuda_kernels and (
            valid_mask is None or bool(valid_mask.all())
        )
        if can_use_cuda_kernels:
            if not HAS_CUDA_KERNELS:
                raise RuntimeError(
                    "CUDA kernels are not available. Please install mamba-ssm or disable use_cuda_kernels."
                )
            assert mamba_chunk_scan_combined is not None
            # A parameter set to a constant tensor of -1.0
            A: torch.Tensor = torch.full(
                (self.num_heads,), -1.0, device=x.device, dtype=x.dtype
            )
            # Call Triton kernel. Returns [B, L, num_heads, d_head]
            y: torch.Tensor = mamba_chunk_scan_combined(
                x_split,
                delta,
                A,
                h_in_split,
                h_out_split,
                chunk_size=256,
                dt_softplus=True,
            )
            # Reshape output and apply gate
            y_ssm: torch.Tensor = y.reshape(B, L, self.ssm_dim) * torch.sigmoid(gate)
        else:
            # Initialize the hidden state: [B, num_heads, d_state, d_head]
            h: torch.Tensor = torch.zeros(
                B, self.num_heads, self.d_state, d_head, device=x.device, dtype=x.dtype
            )
            # Output tensor: [B, L, ssm_dim]
            y_ssm = torch.zeros(B, L, self.ssm_dim, device=x.device, dtype=x.dtype)

            for t in range(L):
                # Step size for the current time step: [B, num_heads, 1]
                dt: torch.Tensor = delta_pos[:, t].unsqueeze(-1)
                dt_uns: torch.Tensor = dt.unsqueeze(-1)

                bt: torch.Tensor = h_in_split[:, t].unsqueeze(-1)
                xt: torch.Tensor = x_split[:, t].unsqueeze(-2)

                # Zero-order-hold SSD recurrence with A fixed to -1 per head.
                decay: torch.Tensor = torch.exp(-dt_uns)
                # Exact zero-order-hold discretization for A=-1.
                next_h = decay * h + (1.0 - decay) * (bt * xt)
                if valid_mask is None:
                    h = next_h
                else:
                    step_valid = valid_mask[:, t, None, None, None].to(torch.bool)
                    h = torch.where(step_valid, next_h, h)

                # Readout calculation
                ct: torch.Tensor = h_out_split[:, t].unsqueeze(-1)
                out_val: torch.Tensor = (h * ct).sum(dim=-2)

                # Reshape output and apply gate
                out_reshaped: torch.Tensor = out_val.reshape(B, self.ssm_dim)
                step_output = out_reshaped * torch.sigmoid(gate[:, t])
                if valid_mask is not None:
                    step_output = step_output * valid_mask[:, t, None].to(x.dtype)
                y_ssm[:, t] = step_output

        # Project back to d_model: [B, L, d_model]
        return cast(torch.Tensor, self.out_proj(y_ssm))

    def prefill(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        delta: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SsmState]:
        """Scan a full sequence and return contiguous output plus terminal state.

        Uses pure PyTorch recurrence regardless of use_cuda_kernels so that
        the returned state layout is defined.
        """
        B, L, _ = x.shape
        if valid_mask is not None and valid_mask.shape != (B, L):
            raise ValueError("valid_mask must have shape [batch_size, seq_len]")

        delta_pos: torch.Tensor = torch.nn.functional.softplus(delta)
        h_in_split: torch.Tensor = h_in.view(B, L, self.num_heads, self.d_state)
        h_out_split: torch.Tensor = h_out.view(B, L, self.num_heads, self.d_state)
        d_head: int = self.ssm_dim // self.num_heads
        x_split: torch.Tensor = x.view(B, L, self.num_heads, d_head)

        h: torch.Tensor = torch.zeros(
            B, self.num_heads, self.d_state, d_head, device=x.device, dtype=x.dtype
        )
        y_ssm = torch.zeros(B, L, self.ssm_dim, device=x.device, dtype=x.dtype)

        for t in range(L):
            dt: torch.Tensor = delta_pos[:, t].unsqueeze(-1)
            dt_uns: torch.Tensor = dt.unsqueeze(-1)
            bt: torch.Tensor = h_in_split[:, t].unsqueeze(-1)
            xt: torch.Tensor = x_split[:, t].unsqueeze(-2)
            decay: torch.Tensor = torch.exp(-dt_uns)
            next_h = decay * h + (1.0 - decay) * (bt * xt)
            if valid_mask is None:
                h = next_h
            else:
                step_valid = valid_mask[:, t, None, None, None].to(torch.bool)
                h = torch.where(step_valid, next_h, h)
            ct: torch.Tensor = h_out_split[:, t].unsqueeze(-1)
            out_val: torch.Tensor = (h * ct).sum(dim=-2)
            out_reshaped: torch.Tensor = out_val.reshape(B, self.ssm_dim)
            step_output = out_reshaped * torch.sigmoid(gate[:, t])
            if valid_mask is not None:
                step_output = step_output * valid_mask[:, t, None].to(x.dtype)
            y_ssm[:, t] = step_output

        return cast(torch.Tensor, self.out_proj(y_ssm)), SsmState(state=h)

    def step(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        delta: torch.Tensor,
        state: SsmState,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SsmState]:
        """Advance one token through the SSD recurrence using pure PyTorch.

        Args:
            x, gate, h_in, h_out, delta: Single-position tensors of shape
                [B, 1, ...].
            state: Recurrent state from a prior prefill or step.
            valid_mask: Optional [B] boolean indicating whether this token is
                valid (used during cached generation for finished rows).

        Returns:
            (output, next_state) where output is [B, 1, d_model] and state
            is the updated recurrence state.
        """
        B, one, _ = x.shape
        if one != 1:
            raise ValueError("step expects single-position tensors [B, 1, ...]")
        if valid_mask is not None and valid_mask.shape != (B,):
            raise ValueError("step valid_mask must have shape [batch_size]")

        d_head: int = self.ssm_dim // self.num_heads
        delta_pos: torch.Tensor = torch.nn.functional.softplus(delta)
        dt: torch.Tensor = delta_pos[:, 0].unsqueeze(-1)
        dt_uns: torch.Tensor = dt.unsqueeze(-1)
        h_in_split: torch.Tensor = h_in.view(B, self.num_heads, self.d_state)
        h_out_split: torch.Tensor = h_out.view(B, self.num_heads, self.d_state)
        x_split: torch.Tensor = x.view(B, self.num_heads, d_head)

        bt: torch.Tensor = h_in_split.unsqueeze(-1)
        xt: torch.Tensor = x_split.unsqueeze(-2)
        decay: torch.Tensor = torch.exp(-dt_uns)
        h = state.state
        next_h = decay * h + (1.0 - decay) * (bt * xt)
        if valid_mask is not None:
            step_valid = valid_mask[:, None, None, None].to(torch.bool)
            h = torch.where(step_valid, next_h, h)
        else:
            h = next_h
        ct: torch.Tensor = h_out_split.unsqueeze(-1)
        out_val: torch.Tensor = (h * ct).sum(dim=-2)
        out_reshaped: torch.Tensor = out_val.reshape(B, self.ssm_dim)
        step_output = out_reshaped * torch.sigmoid(gate[:, 0])
        if valid_mask is not None:
            step_output = step_output * valid_mask[:, None].to(x.dtype)

        return cast(torch.Tensor, self.out_proj(step_output.unsqueeze(1))), SsmState(
            state=h
        )
