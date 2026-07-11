import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.halting import ACTHaltingModule


def test_bce_halting() -> None:
    config = MambaHybridConfig(d_model=64)
    module = ACTHaltingModule(config)
    z = torch.randn(2, 16, 64)
    y = torch.randn(2, 8, 64)
    prob = module(z, y)
    assert prob.shape == (2,)
    assert (prob < 0.1).all()  # due to bias init -5.0


def test_halting_gradients() -> None:
    config = MambaHybridConfig(d_model=64)
    module = ACTHaltingModule(config)
    z = torch.randn(2, 16, 64, requires_grad=True)
    y = torch.randn(2, 8, 64, requires_grad=True)
    prob = module(z, y)
    loss = prob.sum()
    loss.backward()

    # Gradients on inputs should be None due to detach
    assert z.grad is None
    assert y.grad is None

    # Gradients on module parameters should be present
    for p in module.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
