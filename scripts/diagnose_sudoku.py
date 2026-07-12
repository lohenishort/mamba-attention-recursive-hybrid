import os
import json
import torch
from typing import List, Dict, Any

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_sudoku import SudokuDataset, SudokuReasoningModel


def main() -> None:
    checkpoint_path = "data/sudoku_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Recreate config
    config_dict = checkpoint["config"]
    # Handle optional config parameters to avoid constructor issues
    clean_config_dict = {
        k: v
        for k, v in config_dict.items()
        if k
        in [
            "d_model",
            "n_meta",
            "l_ans",
            "n_steps",
            "t_cycles",
            "M_min",
            "M_max",
            "use_cuda_kernels",
            "use_moe",
        ]
    }
    config = MambaHybridConfig(**clean_config_dict)

    # Initialize model
    model = SudokuReasoningModel(config, vocab_size=10)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load some samples
    data_path = "data/sudoku.jsonl"
    all_samples: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            all_samples.append(json.loads(line))
            if len(all_samples) >= 100:
                break

    dataset = SudokuDataset(all_samples, augment=False)

    total_cells = 0
    correct_cells = 0
    total_boards = 0
    correct_boards = 0

    with torch.no_grad():
        for i in range(min(20, len(dataset))):
            puzzle, solution = dataset[i]
            # Add batch dim
            puzzle_batch = puzzle.unsqueeze(0)
            logits, _ = model(puzzle_batch)
            preds = logits.argmax(dim=-1).squeeze(0)

            cell_matches = (preds == solution).sum().item()
            correct_cells += cell_matches
            total_cells += 81
            total_boards += 1
            if cell_matches == 81:
                correct_boards += 1

            print(f"Board {i + 1:02d} | Correct cells: {cell_matches}/81")
            if i < 3:
                print("Puzzle:   ", puzzle.tolist()[:27])
                print("Predict:  ", preds.tolist()[:27])
                print("Solution: ", solution.tolist()[:27])
                print("-" * 50)

    print("\nOverall Summary:")
    print(
        f"Per-cell Accuracy: {correct_cells / total_cells * 100:.2f}% ({correct_cells}/{total_cells})"
    )
    print(
        f"Per-board Accuracy: {correct_boards / total_boards * 100:.2f}% ({correct_boards}/{total_boards})"
    )


if __name__ == "__main__":
    main()
