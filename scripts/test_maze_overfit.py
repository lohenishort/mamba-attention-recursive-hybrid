import os
import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.train_maze import MazeDataset, MazeReasoningModel


def main() -> None:
    data_path = "data/maze_hard.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Configuration for 30x30 Maze Solver overfit test
    l_ans = 64
    config = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=l_ans, n_steps=2, t_cycles=2
    )

    # Load dataset and subset to first 5 samples
    full_dataset = MazeDataset(data_path, size=30, max_path_len=l_ans)

    # Custom subset to keep it standard
    samples = full_dataset.samples[:5]

    class SmallMazeDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self, samples: List[Dict[str, Any]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            sample = self.samples[idx]
            grid = torch.tensor(sample["grid"], dtype=torch.float32)
            path = sample["path"]
            grid_flat = grid.flatten()
            path_tokens = [r * 30 + c for r, c in path]
            if len(path_tokens) < 64:
                path_tokens += [900] * (64 - len(path_tokens))
            else:
                path_tokens = path_tokens[:64]
            return grid_flat, torch.tensor(path_tokens, dtype=torch.long)

    dataset = SmallMazeDataset(samples)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Maze overfit test device: {device}")

    model = MazeReasoningModel(config, grid_size=30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    print("Starting overfit training for 150 epochs...")
    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0
        correct_cells = 0
        total_cells = 0

        for grid_flat, target_ids in loader:
            grid_flat, target_ids = grid_flat.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(grid_flat)
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

            total_loss += loss.item() * grid_flat.size(0)

        avg_loss = total_loss / len(dataset)
        acc = (correct_cells / total_cells) * 100
        if epoch % 15 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Path Token Acc: {acc:.2f}%"
            )

    # Evaluate final predictions on first sample
    model.eval()
    with torch.no_grad():
        grid_flat, target_ids = dataset[0]
        logits, _ = model(grid_flat.unsqueeze(0).to(device))
        pred = logits.argmax(dim=-1).squeeze(0).cpu()
        print("\nVerification on Sample 0:")
        print("Predict:  ", pred.tolist()[:20])
        print("Solution: ", target_ids.tolist()[:20])
        match_count = (pred == target_ids).sum().item()
        print(f"Match: {match_count}/64 tokens")


if __name__ == "__main__":
    main()
