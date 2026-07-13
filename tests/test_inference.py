import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.inference import ptrm_inference, select_consensus


def test_ptrm_runs() -> None:
    """Verifies that the ptrm_inference function runs and returns the expected shape."""
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, vocab_size=71
    )
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    model.eval()
    x: torch.Tensor = torch.randn(2, 32, 64)  # [batch_size, seq_len, d_model]
    out: torch.Tensor = ptrm_inference(x, model, K=3, sigma_base=0.01)
    # Expected shape: [batch_size, l_ans, d_model]
    assert out.shape == (2, 8, 71)


def test_ptrm_k_equals_1() -> None:
    """Verifies that when K = 1, ptrm_inference falls back to the default forward pass."""
    config: MambaHybridConfig = MambaHybridConfig(d_model=32, n_meta=8, l_ans=4)
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    model.eval()
    x: torch.Tensor = torch.randn(2, 10, 32)  # [batch_size, seq_len, d_model]
    out1: torch.Tensor = ptrm_inference(x, model, K=1)

    with torch.no_grad():
        out2: torch.Tensor
        out2, _ = model(x)

    assert torch.allclose(out1, out2)


def test_ptrm_consensus_selection() -> None:
    """Verifies consensus majority selection and Q-score routing across candidates."""
    # Set up config
    config: MambaHybridConfig = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=8, t_cycles=1, n_steps=2, M_max=2
    )
    model: MambaAttentionHybrid = MambaAttentionHybrid(config)
    model.eval()

    # We want K = 3 rollouts
    # Candidates for Batch 0:
    # k = 0: Candidate A (sum = 10.0), score = 0.8
    # k = 1: Candidate B (sum = 20.0), score = 0.9
    # k = 2: Candidate A (sum = 10.0), score = 0.7
    # Expected output: Candidate A with score 0.8 (since A is the majority group, and 0.8 > 0.7)

    # Candidates for Batch 1:
    # k = 0: Candidate C (sum = 30.0), score = 0.4
    # k = 1: Candidate D (sum = 40.0), score = 0.6
    # k = 2: Candidate D (sum = 40.0), score = 0.5
    # Expected output: Candidate D with score 0.6 (majority group D, and 0.6 > 0.5)

    B: int = 2
    L_ans: int = 8
    D: int = 64

    cand_A: torch.Tensor = torch.zeros(L_ans, D)
    cand_A[0, 0] = 10.0

    cand_B: torch.Tensor = torch.zeros(L_ans, D)
    cand_B[0, 0] = 20.0

    cand_C: torch.Tensor = torch.zeros(L_ans, D)
    cand_C[0, 0] = 30.0

    cand_D: torch.Tensor = torch.zeros(L_ans, D)
    cand_D[0, 0] = 40.0

    # Stack for batch size 2:
    # rollout 0: [cand_A, cand_C]
    # rollout 1: [cand_B, cand_D]
    # rollout 2: [cand_A, cand_D]
    rollout_cands: list[torch.Tensor] = [
        torch.stack([cand_A, cand_C], dim=0),
        torch.stack([cand_B, cand_D], dim=0),
        torch.stack([cand_A, cand_D], dim=0),
    ]

    # scores:
    # rollout 0: [0.8, 0.4]
    # rollout 1: [0.9, 0.6]
    # rollout 2: [0.7, 0.5]
    rollout_scores: list[torch.Tensor] = [
        torch.tensor([0.8, 0.4]),
        torch.tensor([0.9, 0.6]),
        torch.tensor([0.7, 0.5]),
    ]

    call_count: int = 0

    def mock_answer_update(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        res: torch.Tensor = rollout_cands[call_count]
        call_count += 1
        return res

    def mock_q_head(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        idx: int = call_count - 1
        return rollout_scores[idx]

    logits = torch.zeros(3, B, L_ans, 5)
    token_sequences = torch.tensor(
        [
            [[1] * L_ans, [3] * L_ans],
            [[2] * L_ans, [4] * L_ans],
            [[1] * L_ans, [4] * L_ans],
        ]
    )
    logits.scatter_(-1, token_sequences.unsqueeze(-1), 1.0)
    out = select_consensus(logits, torch.stack(rollout_scores))

    assert out.shape == (2, L_ans, 5)
    # Check Batch 0 is cand_A (sum 10.0)
    assert torch.equal(out[0].argmax(dim=-1), logits[0, 0].argmax(dim=-1))
    # Check Batch 1 is cand_D (sum 40.0)
    assert torch.equal(out[1].argmax(dim=-1), logits[1, 1].argmax(dim=-1))


def test_all_unique_consensus_uses_highest_score_and_restores_mode() -> None:
    logits = torch.zeros(3, 1, 1, 3)
    logits[0, 0, 0, 0] = 1.0
    logits[1, 0, 0, 1] = 1.0
    logits[2, 0, 0, 2] = 1.0
    selected = select_consensus(logits, torch.tensor([[0.1], [0.9], [0.2]]))
    assert selected.argmax(dim=-1).item() == 1

    config = MambaHybridConfig(d_model=32, n_meta=4, l_ans=2, n_steps=1, M_max=1)
    model = MambaAttentionHybrid(config).train()
    first = ptrm_inference(torch.randn(1, 2, 32), model, K=1)
    assert first.shape == (1, 2, config.vocab_size)
    assert model.training
