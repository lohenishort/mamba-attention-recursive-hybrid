import torch
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
