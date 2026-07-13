import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.utils import deterministic_split_indices, exact_match, seed_everything


# --- 1. Custom Dataset for Maze Pathfinding ---
class MazeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, data_path: str, size: int | None = None, max_path_len: int = 64
    ) -> None:
        self.samples: List[Dict[str, Any]] = torch.load(data_path)
        if not self.samples:
            raise ValueError(f"Maze dataset is empty: {data_path}")
        inferred_size = len(self.samples[0]["grid"])
        self.size = inferred_size if size is None else size
        if self.size != inferred_size:
            raise ValueError(
                f"Configured maze size {self.size} does not match dataset size {inferred_size}"
            )
        self.max_path_len = max_path_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        grid = torch.tensor(sample["grid"], dtype=torch.float32)  # [size, size]
        path = sample["path"]
        validate_maze_path(sample["grid"], path)

        # Flatten grid: [L_raw] = [900]
        grid_flat = grid.flatten()

        # Convert path coordinates (r, c) to flat token indices: r * size + c
        path_tokens = [r * self.size + c for r, c in path]

        # Pad or truncate path tokens to max_path_len
        if len(path_tokens) < self.max_path_len:
            # Pad with a dummy padding token (e.g. size * size)
            pad_val = self.size * self.size
            path_tokens += [pad_val] * (self.max_path_len - len(path_tokens))
        elif len(path_tokens) > self.max_path_len:
            raise ValueError(
                f"Path length {len(path_tokens)} exceeds l_ans={self.max_path_len}; "
                "increase l_ans instead of silently truncating the path"
            )

        return grid_flat, torch.tensor(path_tokens, dtype=torch.long)


def validate_maze_path(grid: List[List[int]], path: List[Tuple[int, int]]) -> None:
    size = len(grid)
    if not path or path[0] != (0, 0) or path[-1] != (size - 1, size - 1):
        raise ValueError("Maze path must connect the top-left and bottom-right cells")
    for (r, c), (next_r, next_c) in zip(path, path[1:]):
        if not (0 <= r < size and 0 <= c < size) or grid[r][c] != 0:
            raise ValueError(f"Maze path visits an invalid or blocked cell: {(r, c)}")
        if abs(r - next_r) + abs(c - next_c) != 1:
            raise ValueError("Maze path contains non-adjacent steps")
    end_r, end_c = path[-1]
    if grid[end_r][end_c] != 0:
        raise ValueError("Maze path ends on a blocked cell")


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

        if config.vocab_size != self.vocab_size:
            raise ValueError("config.vocab_size must match the maze vocabulary")
        self.reasoning_encoder = MambaAttentionHybrid(config)

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

        output: Tuple[torch.Tensor, List[torch.Tensor]] = self.reasoning_encoder(X_raw)
        return output


# --- 3. Main Training Driver ---
def main() -> None:
    data_path = "data/maze_dryrun.pt"
    if not os.path.exists(data_path):
        print(
            f"Error: {data_path} not found. Please run the download/generation script first."
        )
        return

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Configuration
    raw_samples: List[Dict[str, Any]] = torch.load(data_path)
    grid_size = len(raw_samples[0]["grid"])
    l_ans = max(len(sample["path"]) for sample in raw_samples)
    config = MambaHybridConfig(
        d_model=64,
        n_meta=16,
        l_ans=l_ans,
        n_steps=2,
        t_cycles=2,
        vocab_size=grid_size * grid_size + 1,
    )

    # Initialize dataset & dataloader
    dataset = MazeDataset(data_path, size=grid_size, max_path_len=l_ans)
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(len(dataset), seed)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, validation_indices)

    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model & optimizer
    model = MazeReasoningModel(config, grid_size=grid_size).to(device)
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
            is_correct = exact_match(preds, target_ids, grid_size * grid_size)  # [B]
            correct_mask = is_correct.float()

            # Compute joint loss
            loss = compute_bce_joint_loss(
                logits,
                target_ids,
                bce_probs,
                correct_mask,
                alpha=1.0,
                ignore_index=grid_size * grid_size,
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * grid_flat.size(0)
            correct_count += int(is_correct.sum().item())
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
                is_correct = exact_match(preds, target_ids, grid_size * grid_size)
                loss = compute_bce_joint_loss(
                    logits,
                    target_ids,
                    bce_probs,
                    is_correct.float(),
                    alpha=1.0,
                    ignore_index=grid_size * grid_size,
                )

                val_loss += loss.item() * grid_flat.size(0)
                val_correct += int(is_correct.sum().item())
                val_samples += grid_flat.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save the model
    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(config),
            "grid_size": grid_size,
            "max_path_len": l_ans,
            "padding_index": grid_size * grid_size,
            "dataset": data_path,
            "seed": seed,
            "validation_indices": validation_indices,
        },
        "data/maze_model.pt",
    )
    print("Successfully saved trained model state dict to data/maze_model.pt")


if __name__ == "__main__":
    main()
