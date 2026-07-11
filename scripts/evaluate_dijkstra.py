import os
import torch
from typing import List

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_dijkstra import DijkstraReasoningModel, DijkstraDataset


def print_routing_tree(parents: List[int], distances: List[float]) -> None:
    """Prints the shortest path parent routing tree in a clean format."""
    print("Node | Parent | Distance from Node 0")
    print("-----+--------+---------------------")
    for i in range(len(parents)):
        parent_str = "Source" if parents[i] == i else f"Node {parents[i]}"
        dist_str = (
            f"{distances[i]:.2f}" if distances[i] != float("inf") else "Unreachable"
        )
        print(f"{i:4d} | {parent_str:6s} | {dist_str}")


def main() -> None:
    model_path = "data/dijkstra_model.pt"
    data_path = "data/dijkstra.pt"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Trained model or dataset not found. Run train_dijkstra first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Dijkstra Evaluation Environment on {device}...")

    # Load checkpoint and config
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint["config"]
    config = MambaHybridConfig(
        d_model=config_dict.get("d_model", 128),
        n_meta=config_dict.get("n_meta", 32),
        l_ans=config_dict.get("l_ans", 20),
        n_steps=config_dict.get("n_steps", 4),
        t_cycles=config_dict.get("t_cycles", 3),
    )

    # Initialize model
    model = DijkstraReasoningModel(config, vocab_size=20).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load dataset
    dataset = DijkstraDataset(data_path, max_samples=10, num_nodes=20, d_model=128)

    # Run evaluation on the first sample
    input_feats, target_ids = dataset[0]

    with torch.no_grad():
        inp_batch = input_feats.unsqueeze(0).to(device)
        logits, _ = model(inp_batch)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()

    # Get raw sample to read the actual distances
    raw_data = torch.load(data_path)[0]
    distances = raw_data["distances"]

    print("\n=== PREDICTED ROUTING TREE ===")
    print_routing_tree(preds, distances)

    print("\n=== GROUND TRUTH ROUTING TREE ===")
    print_routing_tree(target_ids.tolist(), distances)

    # Check accuracy
    correct_nodes = sum(1 for p, t in zip(preds, target_ids.tolist()) if p == t)
    print(
        f"\nParent Node Prediction Accuracy: {correct_nodes}/20 ({correct_nodes / 20 * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
