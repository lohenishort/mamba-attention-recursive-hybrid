import argparse
import random
from typing import List, Tuple
import torch


def generate_maze(size: int) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    while True:
        grid = [[1] * size for _ in range(size)]
        stack = [(0, 0)]
        grid[0][0] = 0
        visited = {(0, 0)}
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                    path_neighbors = 0
                    for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if 0 <= nnx < size and 0 <= nny < size and grid[nnx][nny] == 0:
                            path_neighbors += 1
                    if path_neighbors <= 2:
                        neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                grid[nx][ny] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
                
        grid[0][0] = 0
        grid[size - 1][size - 1] = 0
        
        queue = [[(0, 0)]]
        bfs_visited = {(0, 0)}
        path = []
        while queue:
            curr_path = queue.pop(0)
            cx, cy = curr_path[-1]
            if cx == size - 1 and cy == size - 1:
                path = curr_path
                break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and grid[nx][ny] == 0 and (nx, ny) not in bfs_visited:
                    bfs_visited.add((nx, ny))
                    queue.append(curr_path + [(nx, ny)])
                    
        if path:
            return grid, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic data generator for Mamba-Attention Hybrid reasoning tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["maze"],
        help="Task type to generate data for",
    )
    parser.add_argument("--size", type=int, default=30, help="Grid size of the maze")
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to generate"
    )
    args = parser.parse_args()

    print(f"Generating synthetic {args.task} dataset...")
    print(f"Size: {args.size}x{args.size}, Samples: {args.num_samples}")

    samples = []
    for _ in range(args.num_samples):
        grid, path = generate_maze(args.size)
        samples.append({"grid": grid, "path": path})

    print(f"Successfully generated {len(samples)} samples!")
    # Save a dry-run confirmation message or tensor file if needed
    import os

    os.makedirs("data", exist_ok=True)
    torch.save(samples, "data/maze_dryrun.pt")
    print("Saved dataset to data/maze_dryrun.pt")


if __name__ == "__main__":
    main()
