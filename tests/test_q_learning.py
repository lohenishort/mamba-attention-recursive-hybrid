import torch
import pytest
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_q_joint_loss
from mamba_hybrid.halting import polyak_update


def test_q_learning_flow() -> None:
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
    model = MambaAttentionHybrid(config)
    target_model = MambaAttentionHybrid(config)

    x = torch.randn(2, 32, 64)
    targets = torch.randint(0, 64, (2, 8))

    # Mock running Q forward
    y_final, states, q_preds = model.forward_q(x)
    correct_mask = torch.tensor([1.0, 0.0])

    loss = compute_q_joint_loss(
        y_final, targets, q_preds, correct_mask, target_model, states=states
    )
    assert loss > 0


def test_q_learning_with_states_and_gradients() -> None:
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
    model = MambaAttentionHybrid(config)
    target_model = MambaAttentionHybrid(config)

    x = torch.randn(2, 32, 64)
    targets = torch.randint(0, 64, (2, 8))

    # Mock running Q forward
    y_final, states, q_preds = model.forward_q(x)
    correct_mask = torch.tensor([1.0, 0.0])

    # Test compute_q_joint_loss with states passed explicitly
    loss = compute_q_joint_loss(
        y_final, targets, q_preds, correct_mask, target_model, states=states
    )
    assert loss > 0

    # Test gradients flow back to the model parameters
    loss.backward()  # type: ignore[no-untyped-call]

    # Verify we got gradients on q_mlp weights
    grad_found = False
    for name, param in model.named_parameters():
        if "q_head.q_mlp" in name:
            assert param.grad is not None
            assert torch.norm(param.grad) >= 0.0
            grad_found = True
    assert grad_found, "Could not find gradient for q_mlp parameters"


def test_q_learning_gradients_initial_params() -> None:
    # 1. Gradients flow to M_meta / ans_init_proj when t_cycles=1
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8, t_cycles=1)
    model = MambaAttentionHybrid(config)
    model.train()
    x = torch.randn(2, 32, 64)
    y_final, states, q_preds = model.forward_q(x)
    loss = y_final.sum() + sum(q.sum() for q in q_preds)
    loss.backward()  # type: ignore[no-untyped-call]

    assert model.M_meta.grad is not None
    assert model.ans_init_proj.weight.grad is not None

    # 2. Number of steps is randomized in training mode but matches n_steps in eval mode
    config_bounds = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, M_min=2, M_max=5, n_steps=6
    )
    model_bounds = MambaAttentionHybrid(config_bounds)

    # Eval mode
    model_bounds.eval()
    _, states_eval, q_preds_eval = model_bounds.forward_q(x)
    assert len(states_eval) == 5
    assert len(q_preds_eval) == 5

    # Train mode
    model_bounds.train()
    steps_observed = set()
    for _ in range(50):
        _, states_train, q_preds_train = model_bounds.forward_q(x)
        num_steps = len(states_train)
        assert 2 <= num_steps <= 5
        assert len(q_preds_train) == num_steps
        steps_observed.add(num_steps)
    # Check that we observed at least some variety to confirm randomization
    assert len(steps_observed) > 1


def test_q_loss_rejects_missing_trajectory_states() -> None:
    config = MambaHybridConfig(d_model=64, n_meta=4, l_ans=2, n_steps=1, M_max=1)
    model = MambaAttentionHybrid(config)
    logits, _, q_preds = model.forward_q(torch.randn(1, 2, 64))
    with pytest.raises(ValueError, match="states"):
        compute_q_joint_loss(
            logits,
            torch.zeros(1, 2, dtype=torch.long),
            q_preds,
            torch.ones(1),
            model,
        )


def test_polyak_update_interpolates_parameters() -> None:
    source = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    source.weight.data.fill_(1.0)
    target.weight.data.zero_()
    polyak_update(target, source, tau=0.25)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.25))
