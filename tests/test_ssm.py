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


def test_ssm_prefill_and_steps_match_full_scan_with_masked_rows() -> None:
    torch.manual_seed(0)
    scan = Mamba2SSDScan(d_model=16, expansion=2, num_heads=8, d_state=3).eval()
    x = torch.randn(2, 6, 32)
    gate = torch.randn_like(x)
    h_in = torch.randn(2, 6, 24)
    h_out = torch.randn(2, 6, 24)
    delta = torch.randn(2, 6, 8)
    valid_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, False, True, True, False, True]]
    )

    full_output, full_state = scan.prefill(
        x, gate, h_in, h_out, delta, valid_mask=valid_mask
    )
    prefix_output, state = scan.prefill(
        x[:, :3],
        gate[:, :3],
        h_in[:, :3],
        h_out[:, :3],
        delta[:, :3],
        valid_mask=valid_mask[:, :3],
    )
    step_outputs: list[torch.Tensor] = []
    for position in range(3, x.shape[1]):
        output, state = scan.step(
            x[:, position : position + 1],
            gate[:, position : position + 1],
            h_in[:, position : position + 1],
            h_out[:, position : position + 1],
            delta[:, position : position + 1],
            state,
            valid_mask=valid_mask[:, position],
        )
        step_outputs.append(output)

    incremental_output = torch.cat([prefix_output, *step_outputs], dim=1)
    assert full_state.state.shape == (2, 8, 3, 4)
    assert torch.allclose(incremental_output, full_output, atol=1e-6)
    assert torch.allclose(state.state, full_state.state, atol=1e-6)
    assert torch.allclose(
        scan(x, gate, h_in, h_out, delta, valid_mask=valid_mask),
        full_output,
        atol=1e-6,
    )


def test_ssm_incremental_path_remains_pure_pytorch_with_cuda_configured() -> None:
    scan = Mamba2SSDScan(
        d_model=8,
        expansion=1,
        num_heads=1,
        d_state=1,
        use_cuda_kernels=True,
    )
    values = torch.ones(1, 1, 8)
    projections = torch.ones(1, 1, 1)

    output, state = scan.prefill(
        values,
        torch.zeros_like(values),
        projections,
        projections,
        torch.zeros(1, 1, 1),
    )
    step_output, _ = scan.step(
        values,
        torch.zeros_like(values),
        projections,
        projections,
        torch.zeros(1, 1, 1),
        state,
    )

    assert output.shape == step_output.shape == (1, 1, 8)
