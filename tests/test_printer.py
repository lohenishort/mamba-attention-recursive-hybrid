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
