"""Smoke-evaluate every native head in a shared multitask checkpoint."""

import json

import torch

from mamba_hybrid.tasks.dijkstra import compute_dijkstra_metrics
from mamba_hybrid.tasks.gsm8k import decode_bytes
from mamba_hybrid.tasks.maze import decode_moves
from scripts.train_dijkstra import DijkstraDataset
from scripts.train_gsm8k import GSM8KDataset, collate_gsm8k
from scripts.train_maze import MazeDataset
from scripts.train_multitask import NativeMultiTaskModel
from scripts.train_sudoku import SudokuDataset
from scripts.utils import config_from_dict, require_file


def main() -> None:
    checkpoint_path = "data/unified_model.pt"
    require_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("task") != "multitask_native"
    ):
        raise ValueError(
            "legacy ASCII multitask checkpoint is incompatible; retrain v2"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_samples = torch.load("data/maze_dryrun.pt")
    grid_size = len(maze_samples[0]["grid"])
    model = NativeMultiTaskModel(
        config_from_dict(checkpoint["config"]), grid_size=grid_size
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open("data/sudoku.jsonl") as dataset_file:
        sudoku_sample = json.loads(next(dataset_file))
    sudoku_input, sudoku_target = SudokuDataset([sudoku_sample])[0]
    sudoku_prediction = (
        model.sudoku(sudoku_input.unsqueeze(0).to(device))[0]
        .argmax(dim=-1)
        .squeeze(0)
        .cpu()
    )
    sudoku_blanks = sudoku_input.eq(0)
    print(
        "SUDOKU blank accuracy: "
        f"{sudoku_prediction[sudoku_blanks].eq(sudoku_target[sudoku_blanks]).float().mean().item():.4f}"
    )

    dijkstra_samples = torch.load("data/dijkstra.pt")
    adjacency, source, target, distances = DijkstraDataset(dijkstra_samples)[0]
    dijkstra_prediction, _ = model.dijkstra.generate(
        adjacency.unsqueeze(0).to(device), source.unsqueeze(0).to(device)
    )
    dijkstra_metrics = compute_dijkstra_metrics(
        dijkstra_prediction.cpu(),
        target.unsqueeze(0),
        adjacency.unsqueeze(0),
        distances.unsqueeze(0),
        source.unsqueeze(0),
    )
    print(f"DIJKSTRA optimal parent rate: {dijkstra_metrics.optimal_parent_rate:.4f}")

    maze_dataset = MazeDataset(
        "data/maze_dryrun.pt",
        size=grid_size,
        max_path_len=model.maze.config.l_ans,
    )
    maze_grid, _ = maze_dataset[0]
    maze_tokens, _ = model.maze.generate(maze_grid.unsqueeze(0).to(device))
    maze_decoded = decode_moves(maze_tokens[0].cpu().tolist(), maze_samples[0]["grid"])
    print(f"MAZE solved: {maze_decoded.legal}")

    gsm_dataset = GSM8KDataset("data/gsm8k_test.jsonl")
    questions, mask, _, targets = collate_gsm8k([gsm_dataset[0]])
    gsm_tokens = model.gsm8k.generate(questions.to(device), mask.to(device))[0].cpu()
    expected = decode_bytes([token for token in targets[0].tolist() if token != -100])
    print(
        f"GSM8K predicted: {decode_bytes(gsm_tokens.tolist())!r}; expected: {expected!r}"
    )


if __name__ == "__main__":
    main()
