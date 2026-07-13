import torch

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.printer import AutoregressivePrinter


def test_printer_is_causal_over_generated_tokens() -> None:
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1, M_max=1)
    printer = AutoregressivePrinter(
        config, vocab_size=12, max_length=4, pad_token_id=0, num_layers=1
    ).eval()
    prefix = torch.randn(1, 3, 16)
    first = torch.tensor([[1, 2, 3]])
    changed_future = torch.tensor([[1, 2, 9]])

    first_logits = printer(prefix, first)
    changed_logits = printer(prefix, changed_future)

    assert torch.allclose(first_logits[:, :2], changed_logits[:, :2], atol=1e-6)


def test_printer_generation_stops_at_eos() -> None:
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1, M_max=1)
    printer = AutoregressivePrinter(
        config, vocab_size=8, max_length=4, pad_token_id=0, num_layers=1
    ).eval()
    with torch.no_grad():
        printer.output.weight.zero_()
        printer.output.bias.zero_()
        printer.output.bias[2] = 10.0

    generated = printer.generate(torch.randn(2, 3, 16), bos_token_id=1, eos_token_id=2)

    assert torch.equal(generated, torch.full((2, 1), 2))


def test_printer_prefill_and_steps_match_full_logits_with_masks() -> None:
    torch.manual_seed(0)
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1, M_max=1)
    printer = AutoregressivePrinter(
        config,
        vocab_size=13,
        output_vocab_size=11,
        max_length=5,
        pad_token_id=12,
        num_layers=2,
    ).eval()
    prefix = torch.randn(2, 4, 16)
    prefix_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    decoder_input = torch.tensor([[11, 3, 4, 5], [11, 7, 12, 12]])

    full_logits = printer(prefix, decoder_input, prefix_mask)
    cache = printer.prefill(prefix, prefix_mask)
    incremental_logits: list[torch.Tensor] = []
    for position in range(decoder_input.shape[1]):
        logits, cache = printer.decode_step(
            decoder_input[:, position], cache, position=position
        )
        incremental_logits.append(logits)

    assert len(cache.layers) == 2
    assert cache.layers[0] is not cache.layers[1]
    assert (
        cache.layers[0].attention.key.data_ptr()
        != cache.layers[1].attention.key.data_ptr()
    )
    assert all(layer.attention.valid_length == 8 for layer in cache.layers)
    assert torch.equal(
        cache.layers[0].attention.valid_mask[:, :8],
        torch.cat([prefix_mask, decoder_input.ne(12)], dim=1),
    )
    assert torch.allclose(
        torch.stack(incremental_logits, dim=1), full_logits, atol=1e-5
    )


def test_cached_generation_matches_reference_with_staggered_eos_and_padding() -> None:
    torch.manual_seed(2)
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1, M_max=1)
    printer = AutoregressivePrinter(
        config, vocab_size=9, max_length=5, pad_token_id=0, num_layers=2
    ).eval()
    prefix = torch.randn(3, 4, 16)
    prefix_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    allowed = torch.tensor([False, True, True, True, True, False, False, False, False])

    cached = printer.generate(
        prefix,
        bos_token_id=8,
        eos_token_id=2,
        prefix_mask=prefix_mask,
        allowed_tokens=allowed,
        use_cache=True,
    )
    reference = printer.generate(
        prefix,
        bos_token_id=8,
        eos_token_id=2,
        prefix_mask=prefix_mask,
        allowed_tokens=allowed,
        use_cache=False,
    )

    assert torch.equal(cached, reference)


def test_cached_generation_keeps_finished_rows_padded() -> None:
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1, M_max=1)
    printer = AutoregressivePrinter(
        config, vocab_size=6, max_length=4, pad_token_id=0, num_layers=1
    ).eval()
    calls = 0

    def stagger_logits(hidden: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        logits = torch.full((hidden.shape[0], 1, 6), -10.0, device=hidden.device)
        logits[:, :, 1] = 1.0
        if calls == 0:
            logits[0, :, 2] = 10.0
        else:
            logits[:, :, 2] = 10.0
        calls += 1
        return logits

    printer.output.forward = stagger_logits  # type: ignore[assignment,method-assign]
    generated = printer.generate(torch.randn(2, 3, 16), bos_token_id=5, eos_token_id=2)

    assert torch.equal(generated, torch.tensor([[2, 0], [1, 2]]))
