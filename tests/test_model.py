import torch
from unittest.mock import patch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid


def test_model_e2e_forward() -> None:
    """Tests the end-to-end forward pass shapes of MambaAttentionHybrid."""
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, vocab_size=73
    )
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    x_raw: torch.Tensor = torch.randn(2, 32, 64)  # [batch_size, seq_len, d_model]
    y_final: torch.Tensor
    bce_probs: list[torch.Tensor]
    y_final, bce_probs = model(x_raw)
    # y_final: [batch_size, l_ans, d_model]
    assert y_final.shape == (2, 8, 73)
    assert len(bce_probs) == config.n_steps
    prob: torch.Tensor
    for prob in bce_probs:
        # prob: [batch_size]
        assert prob.shape == (2,)


def test_init_answer_pools_unaligned_input() -> None:
    """Tests pooled initialization when input and answer lengths differ."""
    config: MambaHybridConfig = MambaHybridConfig(d_model=32, n_meta=8, l_ans=12)
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    x_raw: torch.Tensor = torch.randn(3, 16, 32)  # [batch_size, seq_len, d_model]
    ans_init: torch.Tensor = model.init_answer(x_raw)  # [batch_size, l_ans, d_model]

    # Output shape should be [B, l_ans, d_model]
    assert ans_init.shape == (3, 12, 32)

    # Verify that the value is correct: average pooled x_raw passed through linear projection
    pooled: torch.Tensor = x_raw.mean(dim=1)  # [batch_size, d_model]
    expected_proj: torch.Tensor = model.ans_init_proj(pooled)  # [batch_size, d_model]
    # Check that each position along the sequence dim matches expected_proj + y_pos_embed
    i: int
    for i in range(12):
        assert torch.allclose(
            ans_init[:, i, :], expected_proj + model.y_pos_embed[:, i, :], atol=1e-5
        )
        # Verify that positional embedding broke the symmetry (not equal to expected_proj alone)
        assert not torch.allclose(ans_init[:, i, :], expected_proj, atol=1e-5)


def test_init_answer_preserves_aligned_input_positions() -> None:
    """Tests that aligned answer slots retain their corresponding input states."""
    config = MambaHybridConfig(d_model=32, n_meta=8, l_ans=4)
    model = MambaAttentionHybrid(config)
    x_raw = torch.arange(4 * 32, dtype=torch.float32).view(1, 4, 32)
    with torch.no_grad():
        model.ans_init_proj.weight.copy_(torch.eye(32))
        model.ans_init_proj.bias.zero_()
        model.y_pos_embed.zero_()

    answer = model.init_answer(x_raw)

    assert torch.equal(answer, x_raw)


def test_masked_padding_does_not_change_model_output() -> None:
    config = MambaHybridConfig(
        d_model=32,
        n_meta=4,
        l_ans=2,
        n_steps=1,
        t_cycles=1,
        M_min=1,
        M_max=1,
        vocab_size=11,
    )
    model = MambaAttentionHybrid(config).eval()
    x_raw = torch.randn(1, 3, 32)
    padded = torch.cat([x_raw, torch.randn(1, 2, 32)], dim=1)

    base_logits, _ = model(x_raw)
    padded_logits, _ = model(
        padded, x_mask=torch.tensor([[True, True, True, False, False]])
    )

    assert torch.allclose(base_logits, padded_logits, atol=1e-5)


def test_model_rejects_malformed_input_mask() -> None:
    config = MambaHybridConfig(d_model=32, n_meta=4, l_ans=2, n_steps=1, t_cycles=1)
    model = MambaAttentionHybrid(config)

    with torch.no_grad():
        try:
            model(torch.randn(1, 3, 32), x_mask=torch.ones(1, 2, dtype=torch.bool))
        except ValueError as error:
            assert "x_mask" in str(error)
        else:
            raise AssertionError("malformed x_mask was accepted")


def test_act_emits_one_decision_per_completed_cycle() -> None:
    config = MambaHybridConfig(
        d_model=32,
        n_meta=4,
        l_ans=2,
        n_steps=2,
        M_min=1,
        M_max=3,
        vocab_size=7,
    )
    model = MambaAttentionHybrid(config).train()

    states, probabilities = model.forward_state_trajectory(torch.randn(1, 4, 32))

    assert len(states) == config.M_max
    assert len(probabilities) == config.M_max


def test_model_determinism() -> None:
    """Tests that the model is deterministic in eval mode and is stochastic/adds noise in train mode."""
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=32, n_meta=8, l_ans=4, n_steps=3, t_cycles=2, M_max=3
    )
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    x_raw: torch.Tensor = torch.randn(2, 10, 32)  # [batch_size, seq_len, d_model]

    # 1. Eval Mode: must be deterministic
    model.eval()
    y_final_eval1: torch.Tensor
    bce_probs_eval1: list[torch.Tensor]
    y_final_eval1, bce_probs_eval1 = model(x_raw)
    # y_final_eval1: [batch_size, l_ans, d_model]

    y_final_eval2: torch.Tensor
    bce_probs_eval2: list[torch.Tensor]
    y_final_eval2, bce_probs_eval2 = model(x_raw)
    # y_final_eval2: [batch_size, l_ans, d_model]

    assert torch.allclose(y_final_eval1, y_final_eval2, atol=1e-6)
    p1: torch.Tensor
    p2: torch.Tensor
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
        y_final_train1: torch.Tensor
        y_final_train1, _ = model(x_raw)
        # y_final_train1: [batch_size, l_ans, d_model]

        torch.manual_seed(43)
        y_final_train2: torch.Tensor
        y_final_train2, _ = model(x_raw)
        # y_final_train2: [batch_size, l_ans, d_model]

        # Output should be different due to different noise samples
        assert not torch.allclose(y_final_train1, y_final_train2, atol=1e-6)


def test_model_gradients() -> None:
    """Tests gradient flow through the MambaAttentionHybrid model during the supervision cycle."""
    # Full-recursion BPTT must reach learned initialization through every cycle.
    config_warmup: MambaHybridConfig = MambaHybridConfig(
        d_model=32, n_meta=8, l_ans=4, n_steps=3, t_cycles=3, M_max=3
    )
    model_warmup: MambaAttentionHybrid = MambaAttentionHybrid(config_warmup)
    model_warmup.train()

    x_raw_warmup: torch.Tensor = torch.randn(
        2, 10, 32
    )  # [batch_size, seq_len, d_model]
    y_final_warmup: torch.Tensor
    bce_probs_warmup: list[torch.Tensor]
    y_final_warmup, bce_probs_warmup = model_warmup(x_raw_warmup)
    # y_final_warmup: [batch_size, l_ans, d_model]

    loss_warmup: torch.Tensor = y_final_warmup.sum() + sum(
        p.sum() for p in bce_probs_warmup
    )
    loss_warmup.backward()  # type: ignore[no-untyped-call]

    assert model_warmup.M_meta.grad is not None
    assert model_warmup.y_pos_embed.grad is not None
    assert model_warmup.ans_init_proj.weight.grad is not None
    assert model_warmup.ans_init_proj.bias.grad is not None

    # Check gradient flow to planning block parameters
    assert model_warmup.planning_loop.planning_block.beta_1.grad is not None
    assert model_warmup.planning_loop.planning_block.beta_2.grad is not None

    # Check gradient flow to ACT halting head parameters
    name: str
    param: torch.Tensor
    for name, param in model_warmup.q_head.bce_mlp.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."
    assert all(param.grad is None for param in model_warmup.q_head.q_mlp.parameters())

    # Case 2: t_cycles = 1, gradients MUST flow to M_meta / ans_init_proj
    config_no_warmup: MambaHybridConfig = MambaHybridConfig(
        d_model=32, n_meta=8, l_ans=4, n_steps=3, t_cycles=1, M_max=3
    )
    model_no_warmup: MambaAttentionHybrid = MambaAttentionHybrid(config_no_warmup)
    model_no_warmup.train()

    x_raw_no_warmup: torch.Tensor = torch.randn(
        2, 10, 32
    )  # [batch_size, seq_len, d_model]
    y_final_no_warmup: torch.Tensor
    bce_probs_no_warmup: list[torch.Tensor]
    y_final_no_warmup, bce_probs_no_warmup = model_no_warmup(x_raw_no_warmup)
    # y_final_no_warmup: [batch_size, l_ans, d_model]

    loss_no_warmup: torch.Tensor = y_final_no_warmup.sum() + sum(
        p.sum() for p in bce_probs_no_warmup
    )
    loss_no_warmup.backward()  # type: ignore[no-untyped-call]

    # With t_cycles = 1, gradients must flow back to initialization parameters
    assert model_no_warmup.M_meta.grad is not None
    assert model_no_warmup.y_pos_embed.grad is not None
    assert model_no_warmup.ans_init_proj.weight.grad is not None
    assert model_no_warmup.ans_init_proj.bias.grad is not None

    # Check gradient flow to planning block parameters
    assert model_no_warmup.planning_loop.planning_block.beta_1.grad is not None
    assert model_no_warmup.planning_loop.planning_block.beta_2.grad is not None

    # Check gradient flow to ACT halting head parameters
    for name, param in model_no_warmup.q_head.bce_mlp.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."
    assert all(
        param.grad is None for param in model_no_warmup.q_head.q_mlp.parameters()
    )


def test_act_halts_at_minimum_and_validates_tasks() -> None:
    config = MambaHybridConfig(
        d_model=32, n_meta=4, l_ans=2, n_steps=4, M_min=2, M_max=4
    )
    model = MambaAttentionHybrid(config).eval()
    x_raw = torch.randn(1, 3, 32)
    with patch.object(model.q_head, "forward", return_value=torch.ones(1)):
        logits, probabilities = model(x_raw)
    assert logits.shape == (1, 2, config.vocab_size)
    assert len(probabilities) == config.M_min
    with torch.no_grad():
        try:
            model(x_raw, task_names=["UNKNOWN"])
        except ValueError as error:
            assert "unknown task_names" in str(error)
        else:
            raise AssertionError("invalid task name was accepted")
