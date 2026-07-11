import torch
from unittest.mock import patch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid


def test_model_e2e_forward() -> None:
    """Tests the end-to-end forward pass shapes of MambaAttentionHybrid."""
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
    model = MambaAttentionHybrid(config)
    x_raw = torch.randn(2, 32, 64)
    y_final, bce_probs = model(x_raw)
    assert y_final.shape == (2, 8, 64)
    assert len(bce_probs) == config.n_steps
    for prob in bce_probs:
        assert prob.shape == (2,)


def test_init_answer() -> None:
    """Tests the projection and initialization of the answer state."""
    config = MambaHybridConfig(d_model=32, n_meta=8, l_ans=12)
    model = MambaAttentionHybrid(config)
    x_raw = torch.randn(3, 16, 32)
    ans_init = model.init_answer(x_raw)

    # Output shape should be [B, l_ans, d_model]
    assert ans_init.shape == (3, 12, 32)

    # Verify that the value is correct: average pooled x_raw passed through linear projection
    pooled = x_raw.mean(dim=1)
    expected_proj = model.ans_init_proj(pooled)
    # Check that each position along the sequence dim matches expected_proj
    for i in range(12):
        assert torch.allclose(ans_init[:, i, :], expected_proj, atol=1e-5)


def test_model_determinism() -> None:
    """Tests that the model is deterministic in eval mode and is stochastic/adds noise in train mode."""
    config = MambaHybridConfig(d_model=32, n_meta=8, l_ans=4, n_steps=3, t_cycles=2)
    model = MambaAttentionHybrid(config)
    x_raw = torch.randn(2, 10, 32)

    # 1. Eval Mode: must be deterministic
    model.eval()
    y_final_eval1, bce_probs_eval1 = model(x_raw)
    y_final_eval2, bce_probs_eval2 = model(x_raw)

    assert torch.allclose(y_final_eval1, y_final_eval2, atol=1e-6)
    for p1, p2 in zip(bce_probs_eval1, bce_probs_eval2):
        assert torch.allclose(p1, p2, atol=1e-6)

    # 2. Train Mode: with noise always enabled (via patch of torch.rand to return 0.1)
    model.train()
    # Mocking torch.rand to return 0.1 ensures the noise branch (rand < 0.15) is executed,
    # and the noise scale is non-zero.
    with patch("torch.rand", return_value=torch.tensor([0.1])):
        # Running twice with different seeds/stochasticity should produce different results
        # since noise itself is random (randn_like).
        torch.manual_seed(42)
        y_final_train1, _ = model(x_raw)

        torch.manual_seed(43)
        y_final_train2, _ = model(x_raw)

        # Output should be different due to different noise samples
        assert not torch.allclose(y_final_train1, y_final_train2, atol=1e-6)


def test_model_gradients() -> None:
    """Tests gradient flow through the MambaAttentionHybrid model during the supervision cycle."""
    config = MambaHybridConfig(d_model=32, n_meta=8, l_ans=4, n_steps=3, t_cycles=3)
    model = MambaAttentionHybrid(config)
    model.train()

    x_raw = torch.randn(2, 10, 32)
    y_final, bce_probs = model(x_raw)

    # Compute a dummy loss
    loss = y_final.sum() + sum(p.sum() for p in bce_probs)
    loss.backward()

    # Check that gradients do NOT flow to model initialization parameters due to detach() at boundary
    assert model.M_meta.grad is None
    assert model.ans_init_proj.weight.grad is None
    assert model.ans_init_proj.bias.grad is None

    # Check gradient flow to planning block parameters (within supervision cycle)
    assert model.planning_loop.planning_block.beta_1.grad is not None
    assert model.planning_loop.planning_block.beta_2.grad is not None

    # Check gradient flow to ACT halting head parameters
    for name, param in model.q_head.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."
