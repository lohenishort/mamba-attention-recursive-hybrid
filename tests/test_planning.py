import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.planning import PlanningLoop


def test_planning_loop() -> None:
    # Setup config with small dimensions
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, n_steps=3
    )
    loop: PlanningLoop = PlanningLoop(config)

    # Inputs
    x_raw: torch.Tensor = torch.randn(2, 10, 64)  # [batch_size, seq_len, d_model]
    z_init: torch.Tensor = torch.randn(2, 16, 64)  # [batch_size, n_meta, d_model]
    y_init: torch.Tensor = torch.randn(2, 8, 64)  # [batch_size, l_ans, d_model]

    z_final, y_final = loop(x_raw, z_init, y_init, warmup=True)
    assert z_final.shape == (2, 16, 64)
    assert y_final.shape == (2, 8, 64)


def test_planning_loop_gradients() -> None:
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, n_steps=3
    )
    loop: PlanningLoop = PlanningLoop(config)

    # Test with warmup=True (no gradients stored)
    x_raw: torch.Tensor = torch.randn(2, 10, 64, requires_grad=True)
    z_init: torch.Tensor = torch.randn(2, 16, 64, requires_grad=True)
    y_init: torch.Tensor = torch.randn(2, 8, 64, requires_grad=True)

    z_final, y_final = loop(x_raw, z_init, y_init, warmup=True)
    assert z_final.requires_grad is False
    assert y_final.requires_grad is False

    # Test with warmup=False (gradients stored)
    z_final_grad, y_final_grad = loop(x_raw, z_init, y_init, warmup=False)
    assert z_final_grad.requires_grad is True
    assert y_final_grad.requires_grad is True

    loss: torch.Tensor = z_final_grad.sum() + y_final_grad.sum()
    loss.backward()  # type: ignore[no-untyped-call]

    assert x_raw.grad is not None
    assert z_init.grad is not None
    assert y_init.grad is not None

    assert not torch.isnan(x_raw.grad).any()
    assert not torch.isnan(z_init.grad).any()
    assert not torch.isnan(y_init.grad).any()
