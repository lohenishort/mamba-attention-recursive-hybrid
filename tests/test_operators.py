import torch
import torch.nn as nn
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.operators import MambaAttentionHybridBlock


def test_hybrid_block_shapes() -> None:
    # Setup config with d_model=64 and n_meta=16
    config: MambaHybridConfig = MambaHybridConfig(d_model=64, n_meta=16)
    block: MambaAttentionHybridBlock = MambaAttentionHybridBlock(config)

    # Input tensor x: [batch_size, seq_len, d_model]
    x: torch.Tensor = torch.randn(2, 32, 64)
    # Output tensor out: [batch_size, seq_len, d_model]
    out: torch.Tensor = block(x, causal=False)
    assert out.shape == (2, 32, 64)


def test_hybrid_block_gradient_flow() -> None:
    config: MambaHybridConfig = MambaHybridConfig(d_model=64, n_meta=16)
    block: MambaAttentionHybridBlock = MambaAttentionHybridBlock(config)

    # Input tensor x: [batch_size, seq_len, d_model]
    x: torch.Tensor = torch.randn(2, 32, 64, requires_grad=True)
    out: torch.Tensor = block(x, causal=False)

    loss: torch.Tensor = out.sum()
    loss.backward()  # type: ignore[no-untyped-call]

    # Verify gradients flow to input
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Verify gradients flow to projection layer parameters
    assert block.in_proj.weight.grad is not None
    assert block.out_proj.weight.grad is not None

    # Verify gradients flow to attention branch parameters
    assert block.attn_branch.out_proj.weight.grad is not None

    # Verify gradients flow to SSM branch parameters
    assert block.ssm_branch.out_proj.weight.grad is not None

    # Verify gradients flow to learned beta scaling parameters
    assert block.beta_1.grad is not None
    assert block.beta_2.grad is not None


def test_rms_norm() -> None:
    from mamba_hybrid.operators import RMSNorm

    d_model: int = 64
    x: torch.Tensor = torch.randn(2, 32, d_model)
    norm: RMSNorm = RMSNorm(d_model)
    y: torch.Tensor = norm(x)
    assert y.shape == x.shape
    # Check that mean of y^2 along the last dimension is approximately 1
    variance: torch.Tensor = y.pow(2).mean(-1)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-4)


def test_hybrid_block_causal() -> None:
    config: MambaHybridConfig = MambaHybridConfig(d_model=64, n_meta=16)
    block: MambaAttentionHybridBlock = MambaAttentionHybridBlock(config)

    # Input tensor x: [batch_size, seq_len, d_model]
    x: torch.Tensor = torch.randn(2, 32, 64, requires_grad=True)
    out: torch.Tensor = block(x, causal=True)

    # Assert shape
    assert out.shape == (2, 32, 64)

    # Assert gradient calculation
    loss: torch.Tensor = out.sum()
    loss.backward()  # type: ignore[no-untyped-call]

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Verify gradients flow to other parameters
    assert block.in_proj.weight.grad is not None
    assert block.out_proj.weight.grad is not None
    assert block.attn_branch.out_proj.weight.grad is not None
    assert block.ssm_branch.out_proj.weight.grad is not None
    assert block.beta_1.grad is not None
    assert block.beta_2.grad is not None


def test_moe_layer_and_block() -> None:
    from mamba_hybrid.operators import TaskPrefixedMoeLayer

    # 1. Test TaskPrefixedMoeLayer shape & gradient flow
    moe = TaskPrefixedMoeLayer(d_model=64)
    x = torch.randn(2, 32, 64, requires_grad=True)
    out = moe(x, task_names=["MAZE", "SUDOKU"])
    assert out.shape == (2, 32, 64)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # 2. Test MambaAttentionHybridBlock with MoE enabled
    config = MambaHybridConfig(d_model=64, n_meta=16, use_moe=True)
    block = MambaAttentionHybridBlock(config)
    assert block.moe is not None

    x_block = torch.randn(2, 32, 64, requires_grad=True)
    out_block = block(x_block, causal=False, task_names=["MAZE", "SUDOKU"])
    assert out_block.shape == (2, 32, 64)

    loss_block = out_block.sum()
    loss_block.backward()
    assert x_block.grad is not None
    # Verify gradient flows to expert parameters
    expert_seq = block.moe.experts["MAZE"]
    assert isinstance(expert_seq, nn.Sequential)
    assert expert_seq[0].weight.grad is not None


def test_non_causal_hybrid_block_is_sequence_reversal_equivariant() -> None:
    config = MambaHybridConfig(d_model=16, n_meta=2, l_ans=2, n_steps=1)
    block = MambaAttentionHybridBlock(config).eval()
    inputs = torch.randn(2, 7, 16)

    forwards = block(inputs, causal=False)
    backwards = block(inputs.flip(1), causal=False).flip(1)

    assert torch.allclose(forwards, backwards, atol=1e-5)
