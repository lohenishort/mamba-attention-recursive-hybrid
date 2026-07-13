"""Maze move encoding and path-level validation."""

from dataclasses import dataclass

import torch

from mamba_hybrid.evaluation import validate_maze_path

PAD = 0
EOS = 1
UP = 2
DOWN = 3
LEFT = 4
RIGHT = 5
VOCAB_SIZE = 6

_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
_MOVES = {delta: token for token, delta in _DELTAS.items()}


@dataclass(frozen=True)
class DecodedMazePath:
    path: list[tuple[int, int]]
    saw_eos: bool
    legal: bool
    reached_goal: bool


def path_to_moves(path: list[tuple[int, int]]) -> list[int]:
    """Encode an adjacent coordinate path as directional moves plus EOS."""
    if not path:
        raise ValueError("path must not be empty")
    moves: list[int] = []
    for current, following in zip(path, path[1:]):
        delta = (following[0] - current[0], following[1] - current[1])
        if delta not in _MOVES:
            raise ValueError("path contains non-adjacent coordinates")
        moves.append(_MOVES[delta])
    return [*moves, EOS]


def pad_moves(moves: list[int], length: int) -> torch.Tensor:
    """Pad a complete move sequence without truncation."""
    if len(moves) > length:
        raise ValueError(f"move sequence length {len(moves)} exceeds l_ans={length}")
    return torch.tensor([*moves, *([PAD] * (length - len(moves)))], dtype=torch.long)


def decode_moves(
    tokens: list[int],
    grid: list[list[int]],
    *,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
) -> DecodedMazePath:
    """Decode moves and reject boundaries, walls, missing EOS, and early EOS."""
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    expected_goal = goal if goal is not None else (rows - 1, columns - 1)
    path = [start]
    legal = bool(rows and columns and grid[start[0]][start[1]] == 0)
    saw_eos = False
    for token in tokens:
        if token == EOS:
            saw_eos = True
            legal = legal and path[-1] == expected_goal
            break
        if token == PAD or token not in _DELTAS:
            legal = False
            break
        row_delta, column_delta = _DELTAS[token]
        row = path[-1][0] + row_delta
        column = path[-1][1] + column_delta
        if not (0 <= row < rows and 0 <= column < columns) or grid[row][column] != 0:
            legal = False
            break
        path.append((row, column))
    reached_goal = path[-1] == expected_goal
    legal = (
        legal
        and saw_eos
        and reached_goal
        and validate_maze_path(grid, path, start, expected_goal)
    )
    return DecodedMazePath(path, saw_eos, legal, reached_goal)


def maze_correct_mask(
    predictions: torch.Tensor,
    grids: torch.Tensor,
    *,
    starts: list[tuple[int, int]] | None = None,
    goals: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return whether each predicted move sequence is a legal goal-reaching path."""
    batch_size = predictions.shape[0]
    start_values = starts if starts is not None else [(0, 0)] * batch_size
    size = grids.shape[1]
    goal_values = goals if goals is not None else [(size - 1, size - 1)] * batch_size
    results = [
        decode_moves(
            predictions[index].tolist(),
            grids[index].tolist(),
            start=start_values[index],
            goal=goal_values[index],
        ).legal
        for index in range(batch_size)
    ]
    return torch.tensor(results, dtype=torch.bool, device=predictions.device)


__all__ = [
    "DOWN",
    "DecodedMazePath",
    "EOS",
    "LEFT",
    "PAD",
    "RIGHT",
    "UP",
    "VOCAB_SIZE",
    "decode_moves",
    "maze_correct_mask",
    "pad_moves",
    "path_to_moves",
]
