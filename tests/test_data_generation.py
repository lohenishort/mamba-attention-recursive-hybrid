import random
from collections import deque

from scripts.download_all_datasets import generate_dijkstra_graph, generate_sudoku_board
from scripts.generate_data import generate_maze
from scripts.generate_massive_sudoku import count_solutions


def test_generated_maze_path_is_legal_and_shortest() -> None:
    state = random.getstate()
    random.seed(0)
    try:
        grid, path = generate_maze(8)
    finally:
        random.setstate(state)

    assert path[0] == (0, 0)
    assert path[-1] == (7, 7)
    assert all(grid[row][column] == 0 for row, column in path)
    assert all(
        abs(row - next_row) + abs(column - next_column) == 1
        for (row, column), (next_row, next_column) in zip(path, path[1:])
    )

    queue = deque([((0, 0), 0)])
    visited = {(0, 0)}
    shortest_distance = -1
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) == (7, 7):
            shortest_distance = distance
            break
        for row_delta, column_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = row + row_delta, column + column_delta
            neighbor_row, neighbor_column = neighbor
            if (
                0 <= neighbor_row < 8
                and 0 <= neighbor_column < 8
                and grid[neighbor_row][neighbor_column] == 0
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    assert len(path) - 1 == shortest_distance


def test_generated_sudoku_boards_have_unique_solutions() -> None:
    state = random.getstate()
    random.seed(0)
    try:
        samples = [generate_sudoku_board() for _ in range(5)]
    finally:
        random.setstate(state)

    for puzzle, solution in samples:
        assert count_solutions([row[:] for row in puzzle]) == 1
        assert all(
            clue == 0 or clue == solution[row][column]
            for row, puzzle_row in enumerate(puzzle)
            for column, clue in enumerate(puzzle_row)
        )


def test_generated_dijkstra_graph_persists_explicit_source() -> None:
    state = random.getstate()
    random.seed(4)
    try:
        sample = generate_dijkstra_graph(num_nodes=5, source=3)
    finally:
        random.setstate(state)

    assert sample["schema_version"] == 2
    assert sample["source"] == 3
    assert sample["distances"][3] == 0.0
    assert sample["parents"][3] == -1
