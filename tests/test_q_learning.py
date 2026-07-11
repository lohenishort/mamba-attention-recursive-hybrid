import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_q_joint_loss


def test_q_learning_flow() -> None:
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
    model = MambaAttentionHybrid(config)
    target_model = MambaAttentionHybrid(config)

    x = torch.randn(2, 32, 64)
    targets = torch.randint(0, 64, (2, 8))

    # Mock running Q forward
    y_final, _, q_preds = model.forward_q(x)
    correct_mask = torch.tensor([1.0, 0.0])

    loss = compute_q_joint_loss(y_final, targets, q_preds, correct_mask, target_model)
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
