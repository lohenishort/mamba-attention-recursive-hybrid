"""Fast evaluator helpers with a source-tree-safe Python fallback."""

from collections import Counter
from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from typing import Callable, TypeAlias, cast

Coordinate: TypeAlias = tuple[int, int]

try:
    _native = import_module("mamba_hybrid._native")

    _native_select_consensus = cast(
        Callable[[Sequence[Sequence[int]], Sequence[float]], int],
        _native.select_consensus,
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
    _native_validate_sudoku_board = cast(
        Callable[[Sequence[int], Sequence[int] | None], bool],
        _native.validate_sudoku_board,
    )

    NATIVE_AVAILABLE = True
except ImportError:
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
            if (
                not (0 <= row < rows and 0 <= col < cols)
                or grid[row][col] == wall_value
            ):
                return False
        return all(
            abs(row_a - row_b) + abs(col_a - col_b) == 1
            for (row_a, col_a), (row_b, col_b) in zip(path, path[1:])
        )

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
    "validate_maze_path",
    "validate_sudoku_board",
]
