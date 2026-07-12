import os
import torch
from typing import Tuple
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.train_dijkstra import DijkstraDataset, DijkstraReasoningModel


def main() -> None:
    data_path = "data/dijkstra.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load all Dijkstra samples
    all_samples = torch.load(data_path)
    
    # We use a smaller model for fast overfit testing
    config = MambaHybridConfig(
        d_model=64, n_meta=16, l_ans=20, n_steps=2, t_cycles=2
    )

    full_dataset = DijkstraDataset(all_samples, augment=False, num_nodes=20, d_model=64)

    class SmallDijkstraDataset(
        torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]
    ):
        def __init__(self, dataset: DijkstraDataset) -> None:
            self.dataset = dataset

        def __len__(self) -> int:
            return 5

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.dataset[idx]

    dataset = SmallDijkstraDataset(full_dataset)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dijkstra overfit test device: {device}")

    model = DijkstraReasoningModel(config, vocab_size=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    print("Starting overfit training for 300 epochs...")
    for epoch in range(1, 301):
        model.train()
        total_loss = 0.0
        correct_nodes = 0
        total_nodes = 0

        for X_raw, target_ids in loader:
            X_raw, target_ids = X_raw.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(X_raw)
            preds = logits.argmax(dim=-1)

            # Target nodes matched
            correct_nodes += (preds == target_ids).sum().item()
            total_nodes += target_ids.numel()

            is_correct = (preds == target_ids).all(dim=-1)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            total_loss += loss.item() * X_raw.size(0)

        avg_loss = total_loss / len(dataset)
        acc = (correct_nodes / total_nodes) * 100
        if epoch % 30 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Parent Match Acc: {acc:.2f}%"
            )

    # Evaluate final predictions on first sample
    model.eval()
    with torch.no_grad():
        X_raw, target_ids = dataset[0]
        logits, _ = model(X_raw.unsqueeze(0).to(device))
        pred = logits.argmax(dim=-1).squeeze(0).cpu()
        print("\nVerification on Sample 0:")
        print("Predict:  ", pred.tolist())
        print("Solution: ", target_ids.tolist())
        match_count = (pred == target_ids).sum().item()
        print(f"Match: {match_count}/20 parent nodes")


if __name__ == "__main__":
    main()
