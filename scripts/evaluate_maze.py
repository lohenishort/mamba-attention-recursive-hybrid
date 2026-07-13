"""Evaluate Maze checkpoints by legal goal-reaching paths."""

import torch
from torch.utils.data import DataLoader, Subset

from mamba_hybrid.tasks.maze import decode_moves, path_to_moves
from scripts.train_maze import MazeDataset, MazeReasoningModel
from scripts.utils import config_from_dict, load_validation_indices, require_file


def main() -> None:
    checkpoint_path = "data/maze_model.pt"
    require_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "maze":
        raise ValueError("legacy Maze checkpoint is not compatible; retrain schema v2")
    data_path = str(checkpoint["dataset"])
    require_file(data_path)
    config = config_from_dict(checkpoint["config"])
    grid_size = int(checkpoint["task_config"]["grid_size"])
    dataset = MazeDataset(data_path, size=grid_size, max_path_len=config.l_ans)
    indices = load_validation_indices(checkpoint)
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False)
    raw_samples = torch.load(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MazeReasoningModel(config, grid_size=grid_size).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    solved = 0
    optimal = 0
    saw_eos = 0
    total = 0
    offset = 0
    for grids, _ in loader:
        generated, _ = model.generate(grids.to(device))
        for batch_index, tokens in enumerate(generated.cpu().tolist()):
            sample = raw_samples[indices[offset + batch_index]]
            decoded = decode_moves(tokens, sample["grid"])
            solved += int(decoded.legal)
            saw_eos += int(decoded.saw_eos)
            reference_moves = path_to_moves(
                [tuple(coordinate) for coordinate in sample["path"]]
            )
            optimal += int(decoded.legal and len(decoded.path) == len(reference_moves))
            total += 1
        offset += grids.shape[0]
    print(f"EOS rate: {saw_eos / total:.4f}")
    print(f"Legal goal-reaching solve rate: {solved / total:.4f}")
    print(f"Shortest-path solve rate: {optimal / total:.4f}")


if __name__ == "__main__":
    main()
