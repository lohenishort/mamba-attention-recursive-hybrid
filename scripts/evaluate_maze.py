import os
import torch
from typing import List, Dict, Any, Tuple

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.inference import ptrm_inference
from scripts.train_maze import MazeReasoningModel, MazeDataset


def print_maze(
    grid: List[List[float]],
    path: List[Tuple[int, int]],
    pred_path: List[Tuple[int, int]] | None = None,
) -> None:
    """Prints the maze in ASCII format.
    █ = Wall
    . = Open path
    S = Start, E = End
    * = Ground truth path
    X = Predicted path
    """
    size = len(grid)
    grid_chars = [
        ["█" if grid[r][c] == 1 else " " for c in range(size)] for r in range(size)
    ]

    # Draw ground truth path
    for r, c in path:
        if 0 <= r < size and 0 <= c < size:
            grid_chars[r][c] = "·"

    # Draw predicted path if provided
    if pred_path:
        for r, c in pred_path:
            if 0 <= r < size and 0 <= c < size:
                if grid_chars[r][c] == "·":
                    grid_chars[r][c] = "*"  # Match/overlap
                else:
                    grid_chars[r][c] = "x"  # Misaligned prediction

    grid_chars[0][0] = "S"
    grid_chars[size - 1][size - 1] = "E"

    # Print with borders
    print("+" + "-" * size + "+")
    for r in range(size):
        print("|" + "".join(grid_chars[r]) + "|")
    print("+" + "-" * size + "+")


def main() -> None:
    model_path = "data/maze_model.pt"
    data_path = "data/maze_dryrun.pt"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Trained model or dataset not found. Run train_maze first.")
        return

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading test environment on {device}...")

    # Initialize config (must match training parameters)
    l_ans = 16
    config = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=l_ans, n_steps=2, t_cycles=2
    )

    # Load dataset
    raw_data: List[Dict[str, Any]] = torch.load(data_path)
    dataset = MazeDataset(data_path, size=10, max_path_len=l_ans)

    # Load model
    model = MazeReasoningModel(config, grid_size=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\n--- Running Evaluation with PTRM Consensus Selection ---")

    # Pick a random sample
    idx = 0
    grid_flat, target_ids = dataset[idx]

    # Prepare input tensor for ptrm_inference
    # We embed the input using model's state_embed and 2D positional embeddings
    with torch.no_grad():
        grid_flat_batch = grid_flat.unsqueeze(0).to(device)  # [1, 100]
        B = grid_flat_batch.shape[0]

        # Build X_raw just like in forward pass
        x_state = model.state_embed(grid_flat_batch.long())
        row_pos = model.row_embed.unsqueeze(2).expand(
            B, model.grid_size, model.grid_size, -1
        )
        col_pos = model.col_embed.unsqueeze(1).expand(
            B, model.grid_size, model.grid_size, -1
        )
        pos_2d = torch.cat([row_pos, col_pos], dim=-1).view(
            B, model.grid_size * model.grid_size, -1
        )
        X_raw = x_state + pos_2d

        # Run PTRM consensus voting inference with K=5 rollouts
        # This calls ptrm_inference which stochastically samples trajectories and filters them
        y_final = ptrm_inference(
            X_raw, model.reasoning_encoder, K=5, sigma_base=0.01
        )  # [1, l_ans, d_model]

        # Project representation to token vocabulary
        logits = model.token_generator(y_final)  # [1, l_ans, vocab_size]
        preds = logits.argmax(dim=-1).squeeze(0)  # [l_ans]

    # Decode predicted tokens back to coordinates
    pred_path: List[Tuple[int, int]] = []
    for token in preds.tolist():
        if token < 100:  # Valid cell token
            r = token // 10
            c = token % 10
            pred_path.append((r, c))

    # Remove padding from target path
    true_path_tokens = target_ids.tolist()
    true_path = []
    for token in true_path_tokens:
        if token < 100:
            true_path.append((token // 10, token % 10))

    print("\n[Ground Truth Path]:", true_path)
    print("[Predicted Path]:   ", pred_path)

    print("\nVisualizing Maze Solving:")
    print("S = Start, E = End, · = Truth, * = Predicted Path (Match)")
    print_maze(raw_data[idx]["grid"], true_path, pred_path)


if __name__ == "__main__":
    main()
