import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.utils import (
    deterministic_split_indices,
    exact_match,
    seed_everything,
)


# --- 1. Custom Dataset for Dijkstra ---
import random


def augment_dijkstra(
    adj: List[List[float]], parents: List[int]
) -> Tuple[List[List[float]], List[int]]:
    N = len(adj)
    perm = list(range(N))
    source = next((i for i, parent in enumerate(parents) if parent == -1), 0)
    movable = [i for i in range(N) if i != source]
    shuffled = movable.copy()
    random.shuffle(shuffled)
    for old, new in zip(movable, shuffled):
        perm[old] = new
    perm[source] = source

    new_adj = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            new_adj[perm[i]][perm[j]] = adj[i][j]

    new_parents = [-1] * N
    for i in range(N):
        p = parents[i]
        if p != -1:
            new_parents[perm[i]] = perm[p]

    return new_adj, new_parents


class DijkstraDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        augment: bool = False,
        num_nodes: int = 20,
        d_model: int = 128,
    ) -> None:
        self.samples = samples
        self.augment = augment
        self.num_nodes = num_nodes
        self.d_model = d_model

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        adj = sample["adjacency"]  # [num_nodes, num_nodes]
        parents = sample["parents"]  # [num_nodes]

        if self.augment:
            adj, parents = augment_dijkstra(adj, parents)

        # Convert adjacency to continuous features: [num_nodes, d_model]
        features = torch.zeros(self.num_nodes, self.d_model)
        for i in range(self.num_nodes):
            row = adj[i]
            features[i, : len(row)] = torch.tensor(row, dtype=torch.float32)

        # Target: parents representation [num_nodes]
        target = []
        for i, p in enumerate(parents):
            if p == -1:
                target.append(i)  # Self-loop for source/unconnected
            else:
                target.append(p)

        return features, torch.tensor(target, dtype=torch.long)


# --- 2. Dijkstra Reasoning Model ---
class DijkstraReasoningModel(nn.Module):
    def __init__(self, config: MambaHybridConfig, vocab_size: int = 20) -> None:
        super().__init__()
        self.config = config
        self.pos_embed = nn.Parameter(torch.randn(1, 20, config.d_model))
        self.reasoning_encoder = MambaAttentionHybrid(config)
        if config.vocab_size != vocab_size:
            raise ValueError("config.vocab_size must match the Dijkstra vocabulary")

    def forward(self, X_raw: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # X_raw shape: [B, 20, D] (continuous graph adjacency rows)
        X_raw = X_raw + self.pos_embed
        output: Tuple[torch.Tensor, List[torch.Tensor]] = self.reasoning_encoder(X_raw)
        return output


# --- 3. Main Training Driver ---
def main() -> None:
    data_path = "data/dijkstra.pt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found; run `uv run python -m scripts.download_all_datasets`"
        )

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Configure model for exactly 20 answer slots (parents of the 20 nodes)
    l_ans = 20
    config = MambaHybridConfig(
        d_model=128, n_meta=32, l_ans=l_ans, n_steps=4, t_cycles=3, vocab_size=20
    )

    # Load all Dijkstra samples
    all_samples: List[Dict[str, Any]] = []
    if os.path.exists(data_path):
        all_samples = torch.load(data_path)[:1000]

    # Shuffle and split
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(
        len(all_samples), seed
    )
    train_samples = [all_samples[index] for index in train_indices]
    val_samples_list = [all_samples[index] for index in validation_indices]

    train_set = DijkstraDataset(train_samples, augment=True, num_nodes=20, d_model=128)
    val_set = DijkstraDataset(
        val_samples_list, augment=False, num_nodes=20, d_model=128
    )

    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Initialize model & optimizer
    model = DijkstraReasoningModel(config, vocab_size=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    epochs = 20
    print(f"Starting training on {len(train_set)} Dijkstra graph routing samples...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for input_feats, target_ids in train_loader:
            input_feats, target_ids = input_feats.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(input_feats)

            # Check if predicted parent tree matches solution exactly
            preds = logits.argmax(dim=-1)
            is_correct = exact_match(preds, target_ids)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_feats.size(0)
            correct_count += int(is_correct.sum().item())
            total_samples += input_feats.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_count / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for input_feats, target_ids in val_loader:
                input_feats, target_ids = input_feats.to(device), target_ids.to(device)
                logits, bce_probs = model(input_feats)
                preds = logits.argmax(dim=-1)
                is_correct = exact_match(preds, target_ids)
                loss = compute_bce_joint_loss(
                    logits, target_ids, bce_probs, is_correct.float(), alpha=1.0
                )

                val_loss += loss.item() * input_feats.size(0)
                val_correct += int(is_correct.sum().item())
                val_samples += input_feats.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save the model
    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(config),
            "seed": seed,
            "validation_indices": validation_indices,
            "dataset": data_path,
            "num_nodes": 20,
            "vocab_size": 20,
        },
        "data/dijkstra_model.pt",
    )
    print("Successfully saved trained model state dict to data/dijkstra_model.pt")


if __name__ == "__main__":
    main()
