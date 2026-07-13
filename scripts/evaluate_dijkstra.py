import os
import torch
from typing import List

from scripts.train_dijkstra import DijkstraReasoningModel, DijkstraDataset
from scripts.utils import config_from_dict, load_validation_indices


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
        raise FileNotFoundError(
            "Dijkstra checkpoint or dataset missing; train it first"
        )

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading Dijkstra Evaluation Environment on {device}...")

    # Load checkpoint and config
    checkpoint = torch.load(model_path, map_location=device)
    config = config_from_dict(checkpoint["config"])
    num_nodes = int(checkpoint["num_nodes"])

    # Initialize model
    model = DijkstraReasoningModel(config, vocab_size=int(checkpoint["vocab_size"])).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load dataset
    all_samples = torch.load(data_path)
    validation_indices = load_validation_indices(checkpoint)
    dataset = DijkstraDataset(
        all_samples, augment=False, num_nodes=num_nodes, d_model=config.d_model
    )

    # Run evaluation on the first sample
    input_feats, target_ids = dataset[validation_indices[0]]

    with torch.no_grad():
        inp_batch = input_feats.unsqueeze(0).to(device)
        logits, _ = model(inp_batch)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()

    # Get raw sample to read the actual distances
    raw_data = all_samples[validation_indices[0]]
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
    node_correct = 0
    graph_correct = 0
    with torch.no_grad():
        for index in validation_indices:
            features, target = dataset[index]
            prediction = (
                model(features.unsqueeze(0).to(device))[0].argmax(-1).squeeze(0).cpu()
            )
            node_correct += int(prediction.eq(target).sum().item())
            graph_correct += int(prediction.eq(target).all().item())
    print(
        f"Held-out node accuracy: {node_correct / (len(validation_indices) * num_nodes):.4f}"
    )
    print(
        f"Held-out exact graph accuracy: {graph_correct / len(validation_indices):.4f}"
    )


if __name__ == "__main__":
    main()
