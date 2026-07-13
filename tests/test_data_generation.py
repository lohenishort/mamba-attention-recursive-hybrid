import random

from scripts.download_all_datasets import generate_dijkstra_graph, generate_sudoku_board
from scripts.generate_massive_sudoku import count_solutions


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
