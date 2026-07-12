import os
import torch
import multiprocessing
from typing import Dict, Any
from scripts.generate_data import generate_maze


def generate_single_maze(size: int) -> Dict[str, Any]:
    grid, path = generate_maze(size)
    return {"grid": grid, "path": path}


def main() -> None:
    num_samples = 20000
    size = 30
    print(
        f"Generating massive Maze dataset ({num_samples} samples of {size}x{size}) using multiprocessing..."
    )

    num_cores = multiprocessing.cpu_count()
    print(f"Utilizing {num_cores} CPU cores for parallel generation...")
    with multiprocessing.Pool(num_cores) as pool:
        # Use imap_unordered for progress reporting
        samples = []
        for i, sample in enumerate(
            pool.imap_unordered(generate_single_maze, [size] * num_samples), 1
        ):
            samples.append(sample)
            if i % 4000 == 0:
                print(f"  Generated {i}/{num_samples} mazes...")

    os.makedirs("data", exist_ok=True)
    torch.save(samples, "data/maze_hard.pt")
    print(
        f"Successfully generated and saved {num_samples} Maze-Hard samples to data/maze_hard.pt"
    )


if __name__ == "__main__":
    main()
