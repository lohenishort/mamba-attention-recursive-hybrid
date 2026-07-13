"""Evaluate Dijkstra checkpoints with tie-tolerant shortest-path metrics."""

import torch
from torch.utils.data import DataLoader, Subset

from mamba_hybrid.tasks.dijkstra import compute_dijkstra_metrics
from scripts.train_dijkstra import DijkstraDataset, DijkstraReasoningModel
from scripts.utils import config_from_dict, load_validation_indices, require_file


def main() -> None:
    checkpoint_path = "data/dijkstra_model.pt"
    require_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "dijkstra":
        raise ValueError(
            "legacy Dijkstra checkpoint is not compatible; retrain schema v2"
        )
    data_path = str(checkpoint["dataset"])
    require_file(data_path)
    samples = torch.load(data_path)
    indices = load_validation_indices(checkpoint)
    num_nodes = int(checkpoint["task_config"]["num_nodes"])
    dataset = DijkstraDataset(samples, num_nodes=num_nodes)
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DijkstraReasoningModel(
        config_from_dict(checkpoint["config"]), num_nodes=num_nodes
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    adjacencies: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    for adjacency, source, target, distance in loader:
        generated, _ = model.generate(adjacency.to(device), source.to(device))
        predictions.append(generated.cpu())
        targets.append(target)
        adjacencies.append(adjacency)
        distances.append(distance)
        sources.append(source)
    metrics = compute_dijkstra_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(adjacencies),
        torch.cat(distances),
        torch.cat(sources),
    )
    print(f"Node accuracy: {metrics.node_accuracy:.4f}")
    print(f"Legal-parent rate: {metrics.legal_parent_rate:.4f}")
    print(f"Optimal-parent rate: {metrics.optimal_parent_rate:.4f}")
    print(f"Exact optimal-tree rate: {metrics.exact_tree_rate:.4f}")


if __name__ == "__main__":
    main()
