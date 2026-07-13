import os
import json
import random
from typing import List, Tuple


def pattern(r: int, c: int) -> int:
    return (3 * (r % 3) + r // 3 + c) % 9


def shuffle(s: List[int]) -> List[int]:
    return random.sample(s, len(s))


def count_solutions(board: List[List[int]], limit: int = 2) -> int:
    empty = next(((r, c) for r in range(9) for c in range(9) if board[r][c] == 0), None)
    if empty is None:
        return 1
    r, c = empty
    used = (
        set(board[r])
        | {board[i][c] for i in range(9)}
        | {
            board[i][j]
            for i in range(r // 3 * 3, r // 3 * 3 + 3)
            for j in range(c // 3 * 3, c // 3 * 3 + 3)
        }
    )
    total = 0
    for value in range(1, 10):
        if value in used:
            continue
        board[r][c] = value
        total += count_solutions(board, limit - total)
        board[r][c] = 0
        if total >= limit:
            break
    return total


def generate_sudoku_board() -> Tuple[List[List[int]], List[List[int]]]:
    r_base = range(3)
    rows = [g * 3 + r for g in shuffle(list(r_base)) for r in shuffle(list(r_base))]
    cols = [g * 3 + c for g in shuffle(list(r_base)) for c in shuffle(list(r_base))]
    nums = shuffle(list(range(1, 10)))

    board = [[nums[pattern(r, c)] for c in cols] for r in rows]
    puzzle = [row[:] for row in board]

    target_removals = random.randint(32, 50)
    removed = 0
    for index in random.sample(range(81), 81):
        if removed >= target_removals:
            break
        r, c = divmod(index, 9)
        previous = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions([row[:] for row in puzzle]) != 1:
            puzzle[r][c] = previous
        else:
            removed += 1
    return puzzle, board


def main() -> None:
    random.seed(42)
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
