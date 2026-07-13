import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.inference import ptrm_inference


def test_integration_flow() -> None:
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8, vocab_size=64)
    model = MambaAttentionHybrid(config)
    x = torch.randn(2, 32, 64)
    # [batch_size, seq_len, d_model] = [2, 32, 64]
    targets = torch.randint(0, 64, (2, 8))
    # [batch_size, l_ans] = [2, 8]

    y_final, bce_probs = model(x)
    # y_final: [batch_size, l_ans, d_model] = [2, 8, 64]
    # bce_probs: [batch_size, M_max] = [2, M_max]

    correct = torch.tensor([1.0, 0.0])
    # [batch_size] = [2]

    loss = compute_bce_joint_loss(y_final, targets, bce_probs, correct)
    assert loss > 0

    out = ptrm_inference(x, model, K=2, sigma_base=0.01)
    assert out.shape == (2, 8, 64)
