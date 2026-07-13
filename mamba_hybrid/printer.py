"""Autoregressive answer printer conditioned on planner and raw-context memory."""

from typing import cast

import torch
import torch.nn as nn

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.operators import MambaAttentionHybridBlock, RMSNorm


class AutoregressivePrinter(nn.Module):
    """Print tokens causally from a bidirectional latent/context prefix."""

    def __init__(
        self,
        config: MambaHybridConfig,
        *,
        vocab_size: int,
        output_vocab_size: int | None = None,
        max_length: int,
        pad_token_id: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if max_length <= 0 or num_layers <= 0:
            raise ValueError("max_length and num_layers must be positive")
        self.input_vocab_size = vocab_size
        self.vocab_size = vocab_size if output_vocab_size is None else output_vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, config.d_model)
        self.position_embed = nn.Parameter(
            torch.randn(1, max_length, config.d_model) * 0.02
        )
        self.layers = nn.ModuleList(
            MambaAttentionHybridBlock(config) for _ in range(num_layers)
        )
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, self.vocab_size)

    def forward(
        self,
        prefix: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-token logits for teacher-forced inputs. [B,T,V]."""
        batch_size, prefix_length, _ = prefix.shape
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != batch_size:
            raise ValueError(
                "decoder_input_ids must have shape [batch_size, target_len]"
            )
        target_length = decoder_input_ids.shape[1]
        if target_length > self.max_length:
            raise ValueError("decoder input exceeds max_length")
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch_size,
                prefix_length,
                dtype=torch.bool,
                device=prefix.device,
            )
        if prefix_mask.shape != (batch_size, prefix_length):
            raise ValueError("prefix_mask must match the prefix sequence")

        generated = self.token_embed(decoder_input_ids)
        generated = generated + self.position_embed[:, :target_length]
        sequence = torch.cat([prefix, generated], dim=1)
        generated_mask = decoder_input_ids.ne(self.pad_token_id)
        valid_mask = torch.cat([prefix_mask, generated_mask], dim=1)
        for layer in self.layers:
            assert isinstance(layer, MambaAttentionHybridBlock)
            sequence = layer(
                sequence,
                causal=True,
                valid_mask=valid_mask,
                prefix_length=prefix_length,
            )
        hidden = self.norm(sequence[:, prefix_length:])
        return cast(torch.Tensor, self.output(hidden))

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        prefix_mask: torch.Tensor | None = None,
        allowed_tokens: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Greedily generate through EOS. Returned tokens exclude BOS."""
        limit = self.max_length if max_new_tokens is None else max_new_tokens
        if not 0 < limit <= self.max_length:
            raise ValueError("max_new_tokens must be within printer capacity")
        batch_size = prefix.shape[0]
        decoder_input = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=prefix.device,
        )
        outputs: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prefix.device)
        for _ in range(limit):
            logits = self(prefix, decoder_input, prefix_mask)[:, -1]
            if allowed_tokens is not None:
                if allowed_tokens.shape != (self.vocab_size,):
                    raise ValueError("allowed_tokens must have shape [vocab_size]")
                logits = logits.masked_fill(
                    ~allowed_tokens.to(device=logits.device, dtype=torch.bool),
                    torch.finfo(logits.dtype).min,
                )
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_token_id),
                next_token,
            )
            outputs.append(next_token)
            finished = finished | next_token.eq(eos_token_id)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break
        return torch.stack(outputs, dim=1)


__all__ = ["AutoregressivePrinter"]
