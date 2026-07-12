import os
import json
import random
from typing import List, Tuple


def pattern(r: int, c: int) -> int:
    return (3 * (r % 3) + r // 3 + c) % 9


def shuffle(s: List[int]) -> List[int]:
    return random.sample(s, len(s))


def generate_sudoku_board() -> Tuple[List[List[int]], List[List[int]]]:
    r_base = range(3)
    rows = [g * 3 + r for g in shuffle(list(r_base)) for r in shuffle(list(r_base))]
    cols = [g * 3 + c for g in shuffle(list(r_base)) for c in shuffle(list(r_base))]
    nums = shuffle(list(range(1, 10)))

    board = [[nums[pattern(r, c)] for c in cols] for r in rows]
    puzzle = [row[:] for row in board]

    # Randomly remove between 40% and 70% of numbers to create different difficulty levels
    remove_prob = random.uniform(0.4, 0.7)
    for r in range(9):
        for c in range(9):
            if random.random() < remove_prob:
                puzzle[r][c] = 0
    return puzzle, board


def main() -> None:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "sudoku.jsonl")

    num_samples = 100000
    print(
        f"Generating massive Sudoku dataset ({num_samples} samples) with mixed difficulty levels..."
    )

    with open(dest, "w") as f:
        for i in range(num_samples):
            puzzle, solution = generate_sudoku_board()
            sample = {"puzzle": puzzle, "solution": solution}
            f.write(json.dumps(sample) + "\n")
            if (i + 1) % 20000 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")

    print(f"Successfully generated and saved {num_samples} Sudoku puzzles to {dest}")


if __name__ == "__main__":
    main()
