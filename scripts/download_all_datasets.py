import os
import urllib.request
import json
import random
import heapq
from typing import List, Tuple, Dict, Any
import torch
from scripts.generate_data import generate_maze


def download_file(url: str, dest: str) -> None:
    print(f"Downloading {url} to {dest}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


# --- 1. Maze Dataset Generator ---


def build_maze_dataset(dest: str, size: int = 30, num_samples: int = 1000) -> None:
    print(f"Generating Maze-Hard dataset ({num_samples} samples of {size}x{size})...")
    samples = []
    for _ in range(num_samples):
        grid, path = generate_maze(size)
        samples.append({"grid": grid, "path": path})
    torch.save(samples, dest)
    print(f"Saved Maze-Hard dataset to {dest}")


# --- 2. Sudoku Generator ---
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
    for r in range(9):
        for c in range(9):
            if random.random() < 0.5:  # remove 50% of the numbers
                puzzle[r][c] = 0
    return puzzle, board


def build_sudoku_dataset(dest: str, num_samples: int = 1000) -> None:
    print(f"Generating Sudoku dataset ({num_samples} samples)...")
    samples = []
    for _ in range(num_samples):
        puzzle, solution = generate_sudoku_board()
        samples.append({"puzzle": puzzle, "solution": solution})

    with open(dest, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved Sudoku dataset to {dest}")


# --- 3. Dijkstra Graph Generator (CLRS proxy) ---
def generate_dijkstra_graph(
    num_nodes: int = 20, edge_prob: float = 0.3
) -> Dict[str, Any]:
    adj = [[0.0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                weight = random.uniform(1.0, 10.0)
                adj[i][j] = weight
                adj[j][i] = weight

    dist = [float("inf")] * num_nodes
    dist[0] = 0.0
    pq = [(0.0, 0)]
    parent = [-1] * num_nodes

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(num_nodes):
            w = adj[u][v]
            if w > 0.0:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

    return {"adjacency": adj, "distances": dist, "parents": parent}


def build_dijkstra_dataset(dest: str, num_samples: int = 1000) -> None:
    print(f"Generating Dijkstra Graph Routing dataset ({num_samples} samples)...")
    samples = [generate_dijkstra_graph() for _ in range(num_samples)]
    torch.save(samples, dest)
    print(f"Saved Dijkstra dataset to {dest}")


# --- Main Driver ---
def main() -> None:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate Maze
    build_maze_dataset(os.path.join(data_dir, "maze_hard.pt"))

    # 2. Generate Sudoku
    build_sudoku_dataset(os.path.join(data_dir, "sudoku.jsonl"))

    # 3. Generate Dijkstra (CLRS proxy)
    build_dijkstra_dataset(os.path.join(data_dir, "dijkstra.pt"))

    # 4. Download GSM8K train/test sets
    download_file(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        os.path.join(data_dir, "gsm8k_train.jsonl"),
    )
    download_file(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        os.path.join(data_dir, "gsm8k_test.jsonl"),
    )

    # 5. Clone/Download ARC-AGI dataset
    # We can clone the ARC repo to data/ARC-AGI
    arc_path = os.path.join(data_dir, "ARC-AGI")
    if not os.path.exists(arc_path):
        print("Cloning ARC-AGI repository...")
        import subprocess

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/fchollet/ARC-AGI.git",
                    arc_path,
                ],
                check=True,
            )
            print("Successfully cloned ARC-AGI dataset repository.")
        except Exception as e:
            print(f"Failed to clone ARC-AGI: {e}")
    else:
        print("ARC-AGI repository already exists in data/.")


if __name__ == "__main__":
    main()
