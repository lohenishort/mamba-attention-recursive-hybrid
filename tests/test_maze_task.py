import pytest
import torch

from mamba_hybrid.tasks.maze import (
    DOWN,
    EOS,
    RIGHT,
    decode_moves,
    maze_correct_mask,
    pad_moves,
    path_to_moves,
)


def test_maze_path_round_trip_uses_moves_and_eos() -> None:
    grid = [[0, 0], [1, 0]]
    path = [(0, 0), (0, 1), (1, 1)]

    moves = path_to_moves(path)
    decoded = decode_moves(moves, grid)

    assert moves == [RIGHT, DOWN, EOS]
    assert decoded.path == path
    assert decoded.legal


def test_maze_decoder_rejects_wall_and_missing_eos() -> None:
    grid = [[0, 1], [0, 0]]

    assert not decode_moves([RIGHT, EOS], grid).legal
    assert not decode_moves([DOWN, RIGHT], grid).legal


def test_maze_padding_never_truncates() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        pad_moves([RIGHT, DOWN, EOS], length=2)


def test_maze_correct_mask_scores_complete_paths() -> None:
    grids = torch.tensor([[[0, 0], [1, 0]], [[0, 1], [0, 0]]])
    predictions = torch.tensor(
        [[RIGHT, DOWN, EOS], [RIGHT, DOWN, EOS]], dtype=torch.long
    )

    assert torch.equal(
        maze_correct_mask(predictions, grids), torch.tensor([True, False])
    )
