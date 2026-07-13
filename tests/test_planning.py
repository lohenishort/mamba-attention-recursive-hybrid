import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.answer_update import AnswerUpdateBlock
from mamba_hybrid.planning import PlanningLoop


def test_planning_loop() -> None:
    # Setup config with small dimensions
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, n_steps=3, M_max=3
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
        d_model=64, n_meta=16, l_ans=8, n_steps=3, M_max=3
    )
    loop: PlanningLoop = PlanningLoop(config)

    # The compatibility flag must not cut the full-recursion graph.
    x_raw: torch.Tensor = torch.randn(2, 10, 64, requires_grad=True)
    z_init: torch.Tensor = torch.randn(2, 16, 64, requires_grad=True)
    y_init: torch.Tensor = torch.randn(2, 8, 64, requires_grad=True)

    z_final, y_final = loop(x_raw, z_init, y_init, warmup=True)
    assert z_final.requires_grad is True
    assert y_final.requires_grad is True

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


def test_answer_update_routes_homogeneous_batch_in_one_call() -> None:
    config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=3, n_steps=1, M_max=1, use_moe=True
    )
    loop = PlanningLoop(config)
    assert loop.answer_update_blocks is not None
    block = loop.answer_update_blocks["MAZE"]
    assert isinstance(block, AnswerUpdateBlock)
    batch_sizes: list[int] = []

    def record_batch(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        del module
        batch_sizes.append(inputs[0].shape[0])

    handle = block.register_forward_pre_hook(record_batch)
    try:
        output = loop.update_answer(
            torch.randn(4, 2, 8),
            torch.randn(4, 3, 8),
            task_names=["MAZE"] * 4,
        )
    finally:
        handle.remove()

    assert output.shape == (4, 3, 8)
    assert batch_sizes == [4]
