from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_hybrid.config import MambaHybridConfig


class PrefixCausalAttention(nn.Module):
    """
    Attention module that supports:
    - Bidirectional attention when causal=False
    - Prefix-causal masking when causal=True, where meta-tokens (prefix) can attend to
      each other bidirectionally, but subsequent generated tokens are causally masked and
      meta-tokens cannot attend to subsequent tokens.
    """

    def __init__(self, config: MambaHybridConfig) -> None:
        super().__init__()
        self.config: MambaHybridConfig = config
        self.d_model: int = config.d_model
        self.n_meta: int = config.n_meta
        self.num_heads: int = 8
        self.head_dim: int = self.d_model // self.num_heads

        # Projection layer for output
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,  # [B, num_heads, L, d_head]
        k: torch.Tensor,  # [B, num_heads, L, d_head]
        v: torch.Tensor,  # [B, num_heads, L, d_head]
        causal: bool = False,
    ) -> torch.Tensor:  # [B, L, d_model]
        """
        Forward pass for prefix-causal attention.

        Args:
            q: Query tensor of shape [B, num_heads, L, d_head]
            k: Key tensor of shape [B, num_heads, L, d_head]
            v: Value tensor of shape [B, num_heads, L, d_head]
            causal: Whether to apply prefix-causal masking.

        Returns:
            Output tensor of shape [B, L, d_model]
        """
        B, num_heads, L, d_head = q.shape

        # Compute scaled dot-product attention scores
        # scores shape: [B, num_heads, L, L]
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (d_head**0.5)

        if causal:
            # Initialize mask to zero (no attention allowed by default)
            mask: torch.Tensor = torch.zeros(L, L, device=q.device)

            # Effective meta-token count (in case sequence length is smaller than n_meta)
            eff_n_meta: int = min(self.n_meta, L)

            # 1. Meta-tokens can attend to all meta-tokens bidirectionally
            mask[:eff_n_meta, :eff_n_meta] = 1.0

            if L > self.n_meta:
                # 2. Subsequent tokens can attend to all meta-tokens
                mask[self.n_meta :, : self.n_meta] = 1.0
                # 3. Subsequent tokens can attend to themselves and prior subsequent tokens causally
                mask[self.n_meta :, self.n_meta :] = torch.tril(
                    torch.ones(L - self.n_meta, L - self.n_meta, device=q.device)
                )

            # Apply mask: fill 0s with -inf to prevent attention
            # mask shape: [1, 1, L, L]
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(1) == 0.0, float("-inf")
            )

        # Attention weights of shape [B, num_heads, L, L]
        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)

        # Weighted sum over values
        # y_attn shape: [B, L, d_model]
        y_attn: torch.Tensor = (
            torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, self.d_model)
        )

        # Final output projection
        return cast(torch.Tensor, self.out_proj(y_attn))
