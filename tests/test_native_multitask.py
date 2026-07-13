import torch
from torch.utils.data import DataLoader, TensorDataset

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_multitask import NativeMultiTaskModel, round_robin_batches


def test_native_multitask_shares_core_and_routes_gradients() -> None:
    config = MambaHybridConfig(
        d_model=8,
        n_meta=2,
        l_ans=81,
        n_steps=1,
        M_min=1,
        M_max=1,
        vocab_size=259,
        use_moe=True,
    )
    model = NativeMultiTaskModel(config, grid_size=3)
    decoder = torch.full((1, 81), model.sudoku.pad_token, dtype=torch.long)
    decoder[:, 0] = model.sudoku.bos_token

    logits, _ = model.forward_task(
        "SUDOKU",
        {
            "input_ids": torch.zeros(1, 81, dtype=torch.long),
            "decoder_input_ids": decoder,
        },
    )
    logits.sum().backward()  # type: ignore[no-untyped-call]

    assert model.sudoku.reasoning_encoder is model.reasoning_encoder
    assert model.dijkstra.reasoning_encoder is model.reasoning_encoder
    assert model.reasoning_encoder.M_meta.grad is not None
    assert model.sudoku.printer.output.weight.grad is not None
    assert model.dijkstra.printer.output.weight.grad is None


def test_round_robin_batches_remain_homogeneous() -> None:
    loaders = {
        "SUDOKU": DataLoader(TensorDataset(torch.tensor([1, 2])), batch_size=1),
        "MAZE": DataLoader(TensorDataset(torch.tensor([3])), batch_size=1),
    }

    names = [name for name, _ in round_robin_batches(loaders)]  # type: ignore[arg-type]

    assert names == ["SUDOKU", "MAZE", "SUDOKU"]
