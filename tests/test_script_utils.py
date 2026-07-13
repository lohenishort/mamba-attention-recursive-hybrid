import random

import pytest
import torch

from scripts.generate_massive_sudoku import count_solutions, generate_sudoku_board
from mamba_hybrid.tasks.maze import path_to_moves
from scripts.train_dijkstra import augment_dijkstra_example
from scripts.utils import deterministic_split_indices, exact_match


def test_exact_match_ignores_padding() -> None:
    predictions = torch.tensor([[1, 2, 9], [1, 4, 9]])
    targets = torch.tensor([[1, 2, 0], [1, 3, 0]])
    assert exact_match(predictions, targets, 0).tolist() == [True, False]


def test_split_is_deterministic_and_disjoint() -> None:
    first = deterministic_split_indices(20, 42)
    second = deterministic_split_indices(20, 42)
    assert first == second
    assert not set(first[0]) & set(first[1])
    assert sorted(first[0] + first[1]) == list(range(20))


def test_dijkstra_augmentation_preserves_source() -> None:
    random.seed(3)
    adjacency = [[0.0] * 4 for _ in range(4)]
    sample = {
        "adjacency": adjacency,
        "parents": [2, 2, -1, 1],
        "distances": [1.0, 1.0, 0.0, 2.0],
        "source": 2,
    }
    augmented = augment_dijkstra_example(sample)
    assert augmented["parents"][augmented["source"]] == -1


def test_maze_path_validation_rejects_jumps() -> None:
    with pytest.raises(ValueError, match="non-adjacent"):
        path_to_moves([(0, 0), (1, 1)])


def test_generated_sudoku_has_one_solution() -> None:
    random.seed(1)
    puzzle, solution = generate_sudoku_board()
    assert count_solutions([row[:] for row in puzzle]) == 1
    assert all(sorted(row) == list(range(1, 10)) for row in solution)
