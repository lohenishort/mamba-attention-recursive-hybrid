"""Autoregressive answer printer conditioned on planner and raw-context memory."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.operators import HybridLayerCache, MambaAttentionHybridBlock, RMSNorm


@dataclass
class PrinterCache:
    """Complete printer state for cached autoregressive decoding."""

    layers: tuple[HybridLayerCache, ...]
    bos_token_id: int
    finished: torch.Tensor


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

    def prefill(
        self,
        prefix: torch.Tensor,
        prefix_mask: torch.Tensor | None = None,
        capacity: int = 0,
    ) -> PrinterCache:
        """Build per-layer caches from the complete prefix, layer by layer."""
        batch_size, prefix_length, _ = prefix.shape
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch_size,
                prefix_length,
                dtype=torch.bool,
                device=prefix.device,
            )
        if prefix_mask.shape != (batch_size, prefix_length):
            raise ValueError("prefix_mask must match the prefix sequence")
        effective_capacity = max(capacity, prefix_length + self.max_length)

        current = prefix
        current_mask = prefix_mask
        layer_caches: list[HybridLayerCache] = []
        for layer in self.layers:
            assert isinstance(layer, MambaAttentionHybridBlock)
            current, layer_cache = layer.prefill(
                current,
                valid_mask=current_mask,
                capacity=effective_capacity,
            )
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        batch_size,
                        current.shape[1] - current_mask.shape[1],
                        dtype=torch.bool,
                        device=current.device,
                    ),
                ],
                dim=1,
            )
            layer_caches.append(layer_cache)

        return PrinterCache(
            layers=tuple(layer_caches),
            bos_token_id=-1,  # Set during generation
            finished=torch.zeros(batch_size, dtype=torch.bool, device=prefix.device),
        )

    def decode_step(
        self,
        token_ids: torch.Tensor,
        cache: PrinterCache,
        *,
        position: int,
    ) -> tuple[torch.Tensor, PrinterCache]:
        """Advance one token through all layers using cached state."""
        batch_size = token_ids.shape[0]
        if token_ids.ndim != 1 or token_ids.shape[0] != batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        if position >= self.max_length:
            raise ValueError("position exceeds max_length")

        token_embeddings = (
            self.token_embed(token_ids) + self.position_embed[:, position]
        )
        hidden = token_embeddings.unsqueeze(1)

        token_valid = token_ids.ne(self.pad_token_id)
        new_layers: list[HybridLayerCache] = []
        for layer, layer_cache in zip(self.layers, cache.layers):
            assert isinstance(layer, MambaAttentionHybridBlock)
            hidden, updated_cache = layer.step(
                hidden, valid_mask=token_valid, cache=layer_cache
            )
            new_layers.append(updated_cache)

        hidden = self.norm(hidden)
        logits: torch.Tensor = self.output(hidden).squeeze(1)
        return logits, PrinterCache(
            layers=tuple(new_layers),
            bos_token_id=cache.bos_token_id,
            finished=cache.finished.clone()
            if cache.finished is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=logits.device),
        )

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
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Greedily generate through EOS. Returned tokens exclude BOS."""
        if use_cache:
            return self._generate_cached(
                prefix,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                prefix_mask=prefix_mask,
                allowed_tokens=allowed_tokens,
                max_new_tokens=max_new_tokens,
            )
        return self._generate_uncached(
            prefix,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            prefix_mask=prefix_mask,
            allowed_tokens=allowed_tokens,
            max_new_tokens=max_new_tokens,
        )

    def _generate_uncached(
        self,
        prefix: torch.Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        prefix_mask: torch.Tensor | None = None,
        allowed_tokens: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Legacy uncached generation (matches original behavior exactly)."""
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

    def _generate_cached(
        self,
        prefix: torch.Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        prefix_mask: torch.Tensor | None = None,
        allowed_tokens: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Incremental cached generation using prefill then one-token steps."""
        limit = self.max_length if max_new_tokens is None else max_new_tokens
        if not 0 < limit <= self.max_length:
            raise ValueError("max_new_tokens must be within printer capacity")
        batch_size = prefix.shape[0]
        cache = self.prefill(prefix, prefix_mask, capacity=limit)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=prefix.device)
        outputs: list[torch.Tensor] = []
        current = torch.full(
            (batch_size,),
            bos_token_id,
            dtype=torch.long,
            device=prefix.device,
        )

        for position in range(limit):
            current = torch.where(finished, self.pad_token_id, current)
            logits, cache = self.decode_step(current, cache, position=position)
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
            current = next_token
            if bool(finished.all()):
                break

        return torch.stack(outputs, dim=1)


__all__ = ["AutoregressivePrinter"]
