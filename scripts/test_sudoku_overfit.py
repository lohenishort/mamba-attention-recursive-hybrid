import os
import json
import torch
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.train_sudoku import SudokuDataset, SudokuReasoningModel


def main() -> None:
    data_path = "data/sudoku.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load first 100 samples
    samples: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 100:
                break

    # We use a smaller model for fast overfit testing
    config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=81, n_steps=2, t_cycles=2)

    dataset = SudokuDataset(samples, augment=False)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Overfit test device: {device}")

    model = SudokuReasoningModel(config, vocab_size=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    print("Starting overfit training for 60 epochs...")
    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        correct_cells = 0
        total_cells = 0

        for input_ids, target_ids in loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(input_ids)
            preds = logits.argmax(dim=-1)

            # Target cells matched
            correct_cells += (preds == target_ids).sum().item()
            total_cells += target_ids.numel()

            is_correct = (preds == target_ids).all(dim=-1)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(dataset)
        acc = (correct_cells / total_cells) * 100
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Cell Acc: {acc:.2f}%")

    # Evaluate final predictions on first sample
    model.eval()
    with torch.no_grad():
        puzzle, solution = dataset[0]
        logits, _ = model(puzzle.unsqueeze(0).to(device))
        pred = logits.argmax(dim=-1).squeeze(0).cpu()
        print("\nVerification on Sample 0:")
        print("Puzzle:  ", puzzle.tolist()[:27])
        print("Predict: ", pred.tolist()[:27])
        print("Solution:", solution.tolist()[:27])
        match_count = (pred == solution).sum().item()
        print(f"Match: {match_count}/81 cells")


if __name__ == "__main__":
    main()
