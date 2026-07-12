import os
import torch
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.train_maze import MazeDataset, MazeReasoningModel


def main() -> None:
    data_path = "data/maze_hard.pt"
    if not os.path.exists(data_path):
        print(
            f"Error: {data_path} not found. Please run scripts/generate_massive_maze.py first."
        )
        return

    # Intel XPU / CUDA / CPU Device selection
    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: Training on CPU will be slow. For Intel GPU acceleration, make sure Intel Extension for PyTorch (IPEX) is installed and active."
        )

    # Configuration for 30x30 Maze Solver
    l_ans = 64
    config = MambaHybridConfig(
        d_model=128, n_meta=32, l_ans=l_ans, n_steps=4, t_cycles=3
    )

    # Initialize dataset & dataloader (30x30 grid)
    dataset = MazeDataset(data_path, size=30, max_path_len=l_ans)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model & optimizer
    model = MazeReasoningModel(config, grid_size=30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    epochs = 30
    print(
        f"Starting training of Mamba-Attention Hybrid 30x30 Maze Solver on {len(train_set)} samples..."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for grid_flat, target_ids in train_loader:
            grid_flat, target_ids = grid_flat.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(grid_flat)
            preds = logits.argmax(dim=-1)
            is_correct = (preds == target_ids).all(dim=-1)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * grid_flat.size(0)
            correct_count += is_correct.sum().item()
            total_samples += grid_flat.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_count / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for grid_flat, target_ids in val_loader:
                grid_flat, target_ids = grid_flat.to(device), target_ids.to(device)
                logits, bce_probs = model(grid_flat)
                preds = logits.argmax(dim=-1)
                is_correct = (preds == target_ids).all(dim=-1)
                loss = compute_bce_joint_loss(
                    logits, target_ids, bce_probs, is_correct.float(), alpha=1.0
                )

                val_loss += loss.item() * grid_flat.size(0)
                val_correct += is_correct.sum().item()
                val_samples += grid_flat.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
            flush=True,
        )

        os.makedirs("data", exist_ok=True)
        torch.save(model.state_dict(), "data/maze_model.pt")


if __name__ == "__main__":
    main()
