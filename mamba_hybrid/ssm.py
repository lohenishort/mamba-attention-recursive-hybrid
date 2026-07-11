from typing import cast
import torch
import torch.nn as nn


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

        # delta step sizes: [B, L, num_heads]
        delta_sig: torch.Tensor = torch.sigmoid(delta)

        # Initialize the hidden state: [B, num_heads, d_state]
        h: torch.Tensor = torch.zeros(
            B, self.num_heads, self.d_state, device=x.device, dtype=x.dtype
        )
        # Output tensor: [B, L, ssm_dim]
        y_ssm: torch.Tensor = torch.zeros(
            B, L, self.ssm_dim, device=x.device, dtype=x.dtype
        )

        # Split input and output state projections: [B, L, num_heads, d_state]
        h_in_split: torch.Tensor = h_in.view(B, L, self.num_heads, self.d_state)
        h_out_split: torch.Tensor = h_out.view(B, L, self.num_heads, self.d_state)

        # Head dimension: ssm_dim // num_heads
        d_head: int = self.ssm_dim // self.num_heads

        for t in range(L):
            # Step size for the current time step: [B, num_heads, 1]
            dt: torch.Tensor = delta_sig[:, t].unsqueeze(-1)

            # State update: h_t = (1 - dt) * h_{t-1} + dt * B_t
            h = (1.0 - dt) * h + dt * h_in_split[:, t]

            # Readout calculation: out_val = sum_d(h_t * C_t) -> [B, num_heads]
            out_val: torch.Tensor = (h * h_out_split[:, t]).sum(dim=-1)

            # Broadcast the head outputs across the head dimension
            # out_expanded shape: [B, num_heads, d_head] -> reshaped to [B, ssm_dim]
            out_expanded: torch.Tensor = (
                out_val.unsqueeze(-1).expand(-1, -1, d_head).reshape(B, self.ssm_dim)
            )

            # Apply multiplicative gate: [B, ssm_dim]
            y_ssm[:, t] = out_expanded * torch.sigmoid(gate[:, t])

        # Project back to d_model: [B, L, d_model]
        return cast(torch.Tensor, self.out_proj(y_ssm))
