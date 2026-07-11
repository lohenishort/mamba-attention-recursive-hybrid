import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss


# --- 1. Custom Dataset for Sudoku ---
class SudokuDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, jsonl_path: str, max_samples: int = 1000) -> None:
        self.samples: List[Dict[str, Any]] = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
                    if len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        puzzle = sample["puzzle"]  # 9x9 list of ints
        solution = sample["solution"]  # 9x9 list of ints

        # Flatten puzzle and solution to 81-element 1D tensors
        puzzle_flat = [val for row in puzzle for val in row]
        solution_flat = [val for row in solution for val in row]

        return torch.tensor(puzzle_flat, dtype=torch.long), torch.tensor(
            solution_flat, dtype=torch.long
        )


# --- 2. Sudoku Reasoning Model ---
class SudokuReasoningModel(nn.Module):
    def __init__(self, config: MambaHybridConfig, vocab_size: int = 10) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.reasoning_encoder = MambaAttentionHybrid(config)
        self.token_generator = nn.Linear(config.d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # input_ids shape: [B, 81]
        X_raw = self.embed(input_ids)  # [B, 81, D]
        y_final, bce_probs = self.reasoning_encoder(X_raw)  # [B, 81, D]
        logits = self.token_generator(y_final)  # [B, 81, vocab_size]
        return logits, bce_probs


# --- 3. Main Training Driver ---
def main() -> None:
    data_path = "data/sudoku.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run download_all_datasets first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure model for exactly 81 answer slots
    l_ans = 81
    config = MambaHybridConfig(
        d_model=128, n_meta=32, l_ans=l_ans, n_steps=4, t_cycles=3
    )

    # Initialize dataset & loader
    dataset = SudokuDataset(data_path, max_samples=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model & optimizer
    model = SudokuReasoningModel(config, vocab_size=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    epochs = 20
    print(f"Starting training on {len(train_set)} Sudoku puzzles...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for input_ids, target_ids in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(input_ids)

            # Check if predicted board matches solution exactly
            preds = logits.argmax(dim=-1)
            is_correct = (preds == target_ids).all(dim=-1)
            correct_mask = is_correct.float()

            # Since there is no padding in 81-length Sudoku targets, we don't need ignore_index
            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            correct_count += is_correct.sum().item()
            total_samples += input_ids.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_count / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, bce_probs = model(input_ids)
                preds = logits.argmax(dim=-1)
                is_correct = (preds == target_ids).all(dim=-1)
                loss = compute_bce_joint_loss(
                    logits, target_ids, bce_probs, is_correct.float(), alpha=1.0
                )

                val_loss += loss.item() * input_ids.size(0)
                val_correct += is_correct.sum().item()
                val_samples += input_ids.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save the model
    os.makedirs("data", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": vars(config)},
        "data/sudoku_model.pt",
    )
    print("Successfully saved trained model state dict to data/sudoku_model.pt")


if __name__ == "__main__":
    main()
