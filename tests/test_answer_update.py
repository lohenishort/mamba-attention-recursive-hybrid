import torch
from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.answer_update import AnswerUpdateBlock


def test_answer_update() -> None:
    config: MambaHybridConfig = MambaHybridConfig(d_model=64)
    block: AnswerUpdateBlock = AnswerUpdateBlock(config)
    z: torch.Tensor = torch.randn(2, 16, 64)
    y: torch.Tensor = torch.randn(2, 8, 64)
    out: torch.Tensor = block(z, y)
    assert out.shape == (2, 8, 64)
