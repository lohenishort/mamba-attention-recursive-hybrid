"""Fast evaluator helpers with a source-tree-safe Python fallback."""

from collections import Counter
from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from typing import Callable, TypeAlias, cast

import numpy as np
import numpy.typing as npt

Coordinate: TypeAlias = tuple[int, int]

try:
    _native = import_module("mamba_hybrid._native")

    _native_select_consensus = cast(
        Callable[[Sequence[Sequence[int]], Sequence[float]], int],
        _native.select_consensus,
    )
    _native_select_consensus_array = cast(
        Callable[
            [npt.NDArray[np.int64], npt.NDArray[np.float32]],
            list[int],
        ],
        _native.select_consensus_array,
    )
    _native_validate_maze_path = cast(
        Callable[
            [
                Sequence[Sequence[int]],
                Sequence[Coordinate],
                Coordinate,
                Coordinate | None,
                int,
            ],
            bool,
        ],
        _native.validate_maze_path,
    )
    _native_validate_maze_moves_array = cast(
        Callable[
            [
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
            ],
            list[bool],
        ],
        _native.validate_maze_moves_array,
    )
    _native_validate_sudoku_board = cast(
        Callable[[Sequence[int], Sequence[int] | None], bool],
        _native.validate_sudoku_board,
    )

    NATIVE_AVAILABLE = True
except (ImportError, AttributeError):
    NATIVE_AVAILABLE = False


def _py_select_consensus(
    candidates: Sequence[Sequence[int]], confidences: Sequence[float]
) -> int:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) != len(confidences):
        raise ValueError("candidates and confidences must have equal length")
    if not all(isfinite(score) for score in confidences):
        raise ValueError("confidences must be finite")
    normalized = [tuple(candidate) for candidate in candidates]
    counts = Counter(normalized)
    max_count = max(counts.values())
    return max(
        (
            index
            for index, candidate in enumerate(normalized)
            if counts[candidate] == max_count
        ),
        key=lambda index: (confidences[index], -index),
    )


def _py_select_consensus_batch(
    candidates: Sequence[Sequence[Sequence[int]]],
    confidences: Sequence[Sequence[float]],
) -> list[int]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) != len(confidences):
        raise ValueError("candidates and confidences must have equal rollout counts")
    batch_size = len(candidates[0])
    if batch_size == 0:
        raise ValueError("candidate batches must not be empty")
    if any(len(rollout) != batch_size for rollout in candidates) or any(
        len(rollout) != batch_size for rollout in confidences
    ):
        raise ValueError("all rollouts must have equal batch size")
    return [
        _py_select_consensus(
            [rollout[batch_index] for rollout in candidates],
            [rollout[batch_index] for rollout in confidences],
        )
        for batch_index in range(batch_size)
    ]


def _py_validate_maze_path(
    grid: Sequence[Sequence[int]],
    path: Sequence[Coordinate],
    start: Coordinate = (0, 0),
    goal: Coordinate | None = None,
    wall_value: int = 1,
) -> bool:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols or any(len(row) != cols for row in grid) or not path:
        return False
    expected_goal = goal if goal is not None else (rows - 1, cols - 1)
    if path[0] != start or path[-1] != expected_goal:
        return False
    for row, col in path:
        if not (0 <= row < rows and 0 <= col < cols) or grid[row][col] == wall_value:
            return False
    return all(
        abs(row_a - row_b) + abs(col_a - col_b) == 1
        for (row_a, col_a), (row_b, col_b) in zip(path, path[1:])
    )


def _py_validate_maze_moves(
    tokens: Sequence[int],
    grid: Sequence[Sequence[int]],
    start: Coordinate,
    goal: Coordinate,
) -> bool:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    row, col = start
    if (
        not rows
        or not cols
        or any(len(grid_row) != cols for grid_row in grid)
        or not (0 <= row < rows and 0 <= col < cols)
        or grid[row][col] != 0
    ):
        return False
    deltas = {2: (-1, 0), 3: (1, 0), 4: (0, -1), 5: (0, 1)}
    for token in tokens:
        if token == 1:
            return (row, col) == goal
        if token not in deltas:
            return False
        row_delta, col_delta = deltas[token]
        row += row_delta
        col += col_delta
        if not (0 <= row < rows and 0 <= col < cols) or grid[row][col] != 0:
            return False
    return False


def _py_validate_maze_moves_batch(
    predictions: Sequence[Sequence[int]],
    grids: Sequence[Sequence[Sequence[int]]],
    starts: Sequence[Coordinate] | None = None,
    goals: Sequence[Coordinate] | None = None,
) -> list[bool]:
    batch_size = len(predictions)
    if (
        len(grids) != batch_size
        or (starts is not None and len(starts) != batch_size)
        or (goals is not None and len(goals) != batch_size)
    ):
        raise ValueError(
            "predictions, grids, starts, and goals must have equal batch size"
        )
    return [
        _py_validate_maze_moves(
            predictions[index],
            grids[index],
            (0, 0) if starts is None else starts[index],
            (
                (len(grids[index]) - 1, len(grids[index][0]) - 1)
                if goals is None and grids[index] and grids[index][0]
                else ((-1, -1) if goals is None else goals[index])
            ),
        )
        for index in range(batch_size)
    ]


def _py_validate_sudoku_board(
    board: Sequence[int], puzzle: Sequence[int] | None = None
) -> bool:
    expected = set(range(1, 10))
    if len(board) != 81 or any(value not in expected for value in board):
        return False
    if puzzle is not None and (
        len(puzzle) != 81
        or any(value not in range(10) for value in puzzle)
        or any(clue and clue != value for clue, value in zip(puzzle, board))
    ):
        return False
    rows = [board[offset : offset + 9] for offset in range(0, 81, 9)]
    columns = [[board[row * 9 + col] for row in range(9)] for col in range(9)]
    boxes = [
        [
            board[(box_row + row) * 9 + box_col + col]
            for row in range(3)
            for col in range(3)
        ]
        for box_row in (0, 3, 6)
        for box_col in (0, 3, 6)
    ]
    return all(set(unit) == expected for unit in rows + columns + boxes)


def select_consensus(
    candidates: Sequence[Sequence[int]], confidences: Sequence[float]
) -> int:
    """Return the best candidate index by exact-token consensus and confidence."""
    if NATIVE_AVAILABLE:
        return _native_select_consensus(candidates, confidences)
    return _py_select_consensus(candidates, confidences)


def select_consensus_array(
    candidates: npt.NDArray[np.int64],
    confidences: npt.NDArray[np.float32],
) -> list[int]:
    """Select exact-token consensus from contiguous rollout arrays."""
    candidate_values = np.ascontiguousarray(candidates, dtype=np.int64)
    confidence_values = np.ascontiguousarray(confidences, dtype=np.float32)
    if NATIVE_AVAILABLE:
        return _native_select_consensus_array(candidate_values, confidence_values)
    return _py_select_consensus_batch(
        cast(list[list[list[int]]], candidate_values.tolist()),
        cast(list[list[float]], confidence_values.tolist()),
    )


def validate_maze_path(
    grid: Sequence[Sequence[int]],
    path: Sequence[Coordinate],
    start: Coordinate = (0, 0),
    goal: Coordinate | None = None,
    wall_value: int = 1,
) -> bool:
    """Return whether ``path`` is a complete legal path through ``grid``."""
    if NATIVE_AVAILABLE:
        return _native_validate_maze_path(grid, path, start, goal, wall_value)
    return _py_validate_maze_path(grid, path, start, goal, wall_value)


def validate_maze_moves_array(
    predictions: npt.NDArray[np.int64],
    grids: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    goals: npt.NDArray[np.int64],
) -> list[bool]:
    """Validate contiguous batched maze arrays in Rust when available."""
    prediction_values = np.ascontiguousarray(predictions, dtype=np.int64)
    grid_values = np.ascontiguousarray(grids, dtype=np.int64)
    start_values = np.ascontiguousarray(starts, dtype=np.int64)
    goal_values = np.ascontiguousarray(goals, dtype=np.int64)
    if NATIVE_AVAILABLE:
        return _native_validate_maze_moves_array(
            prediction_values, grid_values, start_values, goal_values
        )
    start_coordinates = [tuple(map(int, value)) for value in start_values.tolist()]
    goal_coordinates = [tuple(map(int, value)) for value in goal_values.tolist()]
    return _py_validate_maze_moves_batch(
        cast(list[list[int]], prediction_values.tolist()),
        cast(list[list[list[int]]], grid_values.tolist()),
        cast(list[Coordinate], start_coordinates),
        cast(list[Coordinate], goal_coordinates),
    )


def validate_sudoku_board(
    board: Sequence[int], puzzle: Sequence[int] | None = None
) -> bool:
    """Return whether a flat board is a valid solution respecting optional clues."""
    if NATIVE_AVAILABLE:
        return _native_validate_sudoku_board(board, puzzle)
    return _py_validate_sudoku_board(board, puzzle)


__all__ = [
    "Coordinate",
    "NATIVE_AVAILABLE",
    "select_consensus",
    "select_consensus_array",
    "validate_maze_moves_array",
    "validate_maze_path",
    "validate_sudoku_board",
]
