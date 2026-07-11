import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss


# --- 1. Custom Dataset for Maze Pathfinding ---
class MazeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_path: str, size: int = 30, max_path_len: int = 64) -> None:
        self.samples: List[Dict[str, Any]] = torch.load(data_path)
        self.size = size
        self.max_path_len = max_path_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        grid = torch.tensor(sample["grid"], dtype=torch.float32)  # [size, size]
        path = sample["path"]

        # Flatten grid: [L_raw] = [900]
        grid_flat = grid.flatten()

        # Convert path coordinates (r, c) to flat token indices: r * size + c
        path_tokens = [r * self.size + c for r, c in path]

        # Pad or truncate path tokens to max_path_len
        if len(path_tokens) < self.max_path_len:
            # Pad with a dummy padding token (e.g. size * size)
            pad_val = self.size * self.size
            path_tokens += [pad_val] * (self.max_path_len - len(path_tokens))
        else:
            path_tokens = path_tokens[: self.max_path_len]

        return grid_flat, torch.tensor(path_tokens, dtype=torch.long)


# --- 2. Maze reasoning model wrapper ---
class MazeReasoningModel(nn.Module):
    def __init__(self, config: MambaHybridConfig, grid_size: int = 30) -> None:
        super().__init__()
        self.config = config
        self.grid_size = grid_size
        self.vocab_size = grid_size * grid_size + 1  # +1 for padding token

        # State embeddings (0 = path, 1 = wall)
        self.state_embed = nn.Embedding(2, config.d_model)

        # 2D Positional Embeddings
        self.row_embed = nn.Parameter(torch.randn(1, grid_size, config.d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(1, grid_size, config.d_model // 2))

        self.reasoning_encoder = MambaAttentionHybrid(config)
        self.token_generator = nn.Linear(config.d_model, self.vocab_size)

    def forward(
        self, grid_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # grid_flat shape: [B, grid_size * grid_size]
        B = grid_flat.shape[0]

        # State embeddings: [B, grid_size * grid_size, d_model]
        x_state = self.state_embed(grid_flat.long())

        # Row & Col positional embeddings
        # We broadcast row and col features to shape [B, grid_size, grid_size, d_model // 2]
        # and concatenate them, then flatten back to [B, grid_size * grid_size, d_model]
        row_pos = self.row_embed.unsqueeze(2).expand(
            B, self.grid_size, self.grid_size, -1
        )
        col_pos = self.col_embed.unsqueeze(1).expand(
            B, self.grid_size, self.grid_size, -1
        )
        pos_2d = torch.cat([row_pos, col_pos], dim=-1).view(
            B, self.grid_size * self.grid_size, -1
        )

        X_raw = x_state + pos_2d  # [B, L_raw, D]

        # Run Mamba-Attention hybrid reasoning loops
        y_final, bce_probs = self.reasoning_encoder(X_raw)  # [B, l_ans, D]

        # Project representation to path token space
        logits = self.token_generator(y_final)  # [B, l_ans, vocab_size]
        return logits, bce_probs


# --- 3. Main Training Driver ---
def main() -> None:
    data_path = "data/maze_dryrun.pt"
    if not os.path.exists(data_path):
        print(
            f"Error: {data_path} not found. Please run the download/generation script first."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    l_ans = 16
    config = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=l_ans, n_steps=2, t_cycles=2
    )

    # Initialize dataset & dataloader
    dataset = MazeDataset(data_path, size=10, max_path_len=l_ans)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model & optimizer
    model = MazeReasoningModel(config, grid_size=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    epochs = 5
    print("Starting training of Mamba-Attention Hybrid Maze Solver...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for grid_flat, target_ids in train_loader:
            grid_flat, target_ids = grid_flat.to(device), target_ids.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits, bce_probs = model(grid_flat)  # logits: [B, l_ans, vocab_size]

            # Calculate accuracy dynamically to get correct_mask for halting supervision
            preds = logits.argmax(dim=-1)  # [B, l_ans]
            # Check if all tokens match target sequence
            is_correct = (preds == target_ids).all(dim=-1)  # [B]
            correct_mask = is_correct.float()

            # Compute joint loss
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

        # Validation epoch
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
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save the model
    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/maze_model.pt")
    print("Successfully saved trained model state dict to data/maze_model.pt")


if __name__ == "__main__":
    main()
