import math

import pytest

from mamba_hybrid.evaluation import (
    select_consensus,
    validate_maze_path,
    validate_sudoku_board,
)


SOLUTION = [
    5,
    3,
    4,
    6,
    7,
    8,
    9,
    1,
    2,
    6,
    7,
    2,
    1,
    9,
    5,
    3,
    4,
    8,
    1,
    9,
    8,
    3,
    4,
    2,
    5,
    6,
    7,
    8,
    5,
    9,
    7,
    6,
    1,
    4,
    2,
    3,
    4,
    2,
    6,
    8,
    5,
    3,
    7,
    9,
    1,
    7,
    1,
    3,
    9,
    2,
    4,
    8,
    5,
    6,
    9,
    6,
    1,
    5,
    3,
    7,
    2,
    8,
    4,
    2,
    8,
    7,
    4,
    1,
    9,
    6,
    3,
    5,
    3,
    4,
    5,
    2,
    8,
    6,
    1,
    7,
    9,
]


def test_consensus_uses_exact_tokens_before_confidence() -> None:
    assert select_consensus([[1, 2], [9], [1, 2]], [0.2, 0.99, 0.8]) == 2


def test_consensus_all_unique_uses_global_confidence() -> None:
    assert select_consensus([[1], [2], [3]], [0.4, 0.9, 0.8]) == 1


@pytest.mark.parametrize("scores", [[], [0.0, math.nan]])
def test_consensus_rejects_invalid_input(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        select_consensus([[1], [2]], scores)


def test_maze_path_validation() -> None:
    grid = [[0, 0, 1], [1, 0, 1], [0, 0, 0]]
    assert validate_maze_path(grid, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)])
    assert not validate_maze_path(grid, [(0, 0), (0, 1), (2, 1), (2, 2)])
    assert not validate_maze_path(grid, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])


def test_maze_supports_explicit_endpoints_and_wall_value() -> None:
    assert validate_maze_path(
        [[7, 0], [7, 0]], [(0, 1), (1, 1)], start=(0, 1), goal=(1, 1), wall_value=7
    )


def test_sudoku_board_validation_and_clues() -> None:
    puzzle = [0] * 81
    puzzle[0] = 5
    assert validate_sudoku_board(SOLUTION, puzzle)
    bad_clue = puzzle.copy()
    bad_clue[0] = 4
    assert not validate_sudoku_board(SOLUTION, bad_clue)
    duplicate = SOLUTION.copy()
    duplicate[0] = duplicate[1]
    assert not validate_sudoku_board(duplicate)
