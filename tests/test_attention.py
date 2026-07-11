import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.attention import PrefixCausalAttention


def test_attention_shapes() -> None:
    # Set up config with d_model=64 and n_meta=16
    config: MambaHybridConfig = MambaHybridConfig(d_model=64, n_meta=16)
    attn: PrefixCausalAttention = PrefixCausalAttention(config)

    # [batch_size, num_heads, seq_len, d_head]
    q: torch.Tensor = torch.randn(2, 8, 32, 8)
    k: torch.Tensor = torch.randn(2, 8, 32, 8)
    v: torch.Tensor = torch.randn(2, 8, 32, 8)

    out: torch.Tensor = attn(q, k, v, causal=True)
    # Expected output shape: [batch_size, seq_len, d_model]
    assert out.shape == (2, 32, 64)


def test_prefix_causal_mask_gradients() -> None:
    # Verifies that gradients do not flow from future tokens to past tokens
    # based on the prefix-causal attention masking requirements.
    n_meta: int = 4
    seq_len: int = 10
    config: MambaHybridConfig = MambaHybridConfig(d_model=16, n_meta=n_meta)
    attn: PrefixCausalAttention = PrefixCausalAttention(config)

    # [batch_size, num_heads, seq_len, d_head]
    q: torch.Tensor = torch.randn(1, 8, seq_len, 2, requires_grad=True)
    k: torch.Tensor = torch.randn(1, 8, seq_len, 2, requires_grad=True)
    v: torch.Tensor = torch.randn(1, 8, seq_len, 2, requires_grad=True)

    out: torch.Tensor = attn(q, k, v, causal=True)

    # 1. Meta-tokens (0 to n_meta-1) should not attend to subsequent generated tokens (n_meta to seq_len-1).
    # Let's verify that out[0, i, :] does not depend on q, k, v at position j where j >= n_meta and i < n_meta.
    for i in range(n_meta):
        loss: torch.Tensor = out[0, i].sum()
        loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        # Gradients with respect to q, k, v at index j >= n_meta must be zero.
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        for j in range(n_meta, seq_len):
            assert torch.all(q.grad[0, :, j, :] == 0.0), (
                f"Meta-token at {i} attended to future token at {j}"
            )
            assert torch.all(k.grad[0, :, j, :] == 0.0), (
                f"Meta-token at {i} attended to future token at {j}"
            )
            assert torch.all(v.grad[0, :, j, :] == 0.0), (
                f"Meta-token at {i} attended to future token at {j}"
            )

        # Reset gradients
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()

    # 2. Subsequent tokens (i >= n_meta) should attend to:
    #    - all meta-tokens (j < n_meta)
    #    - prior subsequent tokens (j <= i)
    #    But NOT future subsequent tokens (j > i).
    for i in range(n_meta, seq_len):
        loss = out[0, i].sum()
        loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # Verify no gradient flow from future tokens (j > i)
        for j in range(i + 1, seq_len):
            assert torch.all(q.grad[0, :, j, :] == 0.0), (
                f"Token at {i} attended to future token at {j}"
            )
            assert torch.all(k.grad[0, :, j, :] == 0.0), (
                f"Token at {i} attended to future token at {j}"
            )
            assert torch.all(v.grad[0, :, j, :] == 0.0), (
                f"Token at {i} attended to future token at {j}"
            )

        # Reset gradients
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
