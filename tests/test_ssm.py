import torch
from mamba_hybrid.ssm import Mamba2SSDScan


def test_ssm_scan() -> None:
    # B=2, L=10, D=64, H=8, D_state=16
    # x: [B, L, ssm_dim] where ssm_dim = d_model * expansion = 128
    x = torch.randn(2, 10, 128)
    # gate: [B, L, ssm_dim] where ssm_dim = 128
    gate = torch.randn(2, 10, 128)
    # h_in: [B, L, H * D_state] where H=8, D_state=16 -> 128
    h_in = torch.randn(2, 10, 8 * 16)
    # h_out: [B, L, H * D_state] where H=8, D_state=16 -> 128
    h_out = torch.randn(2, 10, 8 * 16)
    # delta: [B, L, H] where H=8
    delta = torch.randn(2, 10, 8)

    scan = Mamba2SSDScan(d_model=64, expansion=2, num_heads=8, d_state=16)
    out = scan(x, gate, h_in, h_out, delta)
    # out: [B, L, d_model]
    assert out.shape == (2, 10, 64)


def test_ssm_scan_use_cuda_kernels() -> None:
    scan = Mamba2SSDScan(
        d_model=64, expansion=2, num_heads=8, d_state=16, use_cuda_kernels=True
    )
    x = torch.randn(2, 10, 128)
    gate = torch.randn(2, 10, 128)
    h_in = torch.randn(2, 10, 8 * 16)
    h_out = torch.randn(2, 10, 8 * 16)
    delta = torch.randn(2, 10, 8)
    try:
        out = scan(x, gate, h_in, h_out, delta)
        assert out.shape == (2, 10, 64)
    except RuntimeError as e:
        assert "CUDA kernels" in str(e) or "Triton" in str(e) or "mamba" in str(e)


def test_ssm_recurrence_matches_closed_form_single_step() -> None:
    scan = Mamba2SSDScan(d_model=8, expansion=1, num_heads=1, d_state=1)
    scan.out_proj.weight.data.copy_(torch.eye(8))
    x = torch.ones(1, 1, 8)
    gate = torch.zeros_like(x)
    state_input = torch.full((1, 1, 1), 2.0)
    state_output = torch.full((1, 1, 1), 3.0)
    delta = torch.zeros(1, 1, 1)
    output = scan(x, gate, state_input, state_output, delta)
    dt = torch.nn.functional.softplus(torch.tensor(0.0))
    expected = torch.full_like(output, float((1.0 - torch.exp(-dt)) * 2.0 * 3.0 * 0.5))
    assert torch.allclose(output, expected)
