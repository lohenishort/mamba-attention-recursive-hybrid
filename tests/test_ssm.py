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
