import os
import torch
from typing import List

from scripts.train_sudoku import SudokuReasoningModel, SudokuDataset
from scripts.utils import config_from_dict, load_validation_indices


def print_sudoku_board(board: List[int]) -> None:
    """Prints a flat 81-element Sudoku board in a beautiful 9x9 layout."""
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("------+-------+------")
        row_str = []
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str.append("|")
            val = board[r * 9 + c]
            row_str.append(str(val) if val != 0 else ".")
        print(" ".join(row_str))


def main() -> None:
    model_path = "data/sudoku_model.pt"
    data_path = "data/sudoku.jsonl"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Trained model or dataset not found. Run train_sudoku first.")
        return

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading Sudoku Evaluation Environment on {device}...")

    # Load checkpoint and config
    checkpoint = torch.load(model_path, map_location=device)
    config = config_from_dict(checkpoint["config"])

    # Initialize model
    model = SudokuReasoningModel(config, vocab_size=10).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load dataset
    import json

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    validation_indices = load_validation_indices(checkpoint)
    dataset = SudokuDataset(samples, augment=False)

    # Run evaluation on the first sample
    input_ids, target_ids = dataset[validation_indices[0]]

    with torch.no_grad():
        inp_batch = input_ids.unsqueeze(0).to(device)
        logits, _ = model(inp_batch)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()

    print("\n=== INPUT PUZZLE ===")
    print_sudoku_board(input_ids.tolist())

    print("\n=== PREDICTED SOLUTION ===")
    print_sudoku_board(preds)

    print("\n=== GROUND TRUTH SOLUTION ===")
    print_sudoku_board(target_ids.tolist())

    # Check accuracy
    correct_cells = sum(1 for p, t in zip(preds, target_ids.tolist()) if p == t)
    print(f"\nCell Accuracy: {correct_cells}/81 ({correct_cells / 81 * 100:.1f}%)")
    total_cells = 0
    exact_boards = 0
    with torch.no_grad():
        for index in validation_indices:
            inputs, targets = dataset[index]
            predictions = (
                model(inputs.unsqueeze(0).to(device))[0].argmax(-1).squeeze(0).cpu()
            )
            total_cells += int(predictions.eq(targets).sum().item())
            exact_boards += int(predictions.eq(targets).all().item())
    print(f"Held-out cell accuracy: {total_cells / (81 * len(validation_indices)):.4f}")
    print(
        f"Held-out exact-board accuracy: {exact_boards / len(validation_indices):.4f}"
    )


if __name__ == "__main__":
    main()
