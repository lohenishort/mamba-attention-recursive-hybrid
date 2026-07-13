"""Dijkstra task semantics and structural validation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DijkstraMetrics:
    node_accuracy: float
    legal_parent_rate: float
    optimal_parent_rate: float
    exact_tree_rate: float


def encode_parent_targets(parents: list[int], source: int) -> torch.Tensor:
    """Encode source as self and unreachable vertices as the extra class."""
    num_nodes = len(parents)
    targets = [
        source if index == source else (num_nodes if parent == -1 else parent)
        for index, parent in enumerate(parents)
    ]
    return torch.tensor(targets, dtype=torch.long)


def valid_parent_mask(adjacency: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Return legal parent classes. Shapes: adjacency [B,N,N], source [B], mask [B,N,N+1]."""
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape [batch_size, num_nodes, num_nodes]")
    batch_size, num_nodes, _ = adjacency.shape
    if source.shape != (batch_size,):
        raise ValueError("source must have shape [batch_size]")
    if bool(((source < 0) | (source >= num_nodes)).any()):
        raise ValueError("source contains an invalid node index")

    mask = torch.zeros(
        batch_size,
        num_nodes,
        num_nodes + 1,
        dtype=torch.bool,
        device=adjacency.device,
    )
    mask[:, :, :num_nodes] = adjacency.gt(0)
    mask[:, :, num_nodes] = True
    batch_indices = torch.arange(batch_size, device=adjacency.device)
    mask[batch_indices, source] = False
    mask[batch_indices, source, source] = True
    return mask


def constrain_parent_logits(
    logits: torch.Tensor, adjacency: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Mask non-edge parent classes while retaining the unreachable class."""
    mask = valid_parent_mask(adjacency, source)
    if logits.shape != mask.shape:
        raise ValueError(
            "logits must have shape [batch_size, num_nodes, num_nodes + 1]"
        )
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def optimal_parent_mask(
    adjacency: torch.Tensor,
    distances: torch.Tensor,
    source: torch.Tensor,
    *,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Return every parent class compatible with shortest-path distances."""
    batch_size, num_nodes, _ = adjacency.shape
    if distances.shape != (batch_size, num_nodes):
        raise ValueError("distances must have shape [batch_size, num_nodes]")
    parent_distances = distances[:, None, :]
    node_distances = distances[:, :, None]
    candidate_costs = parent_distances + adjacency
    finite_nodes = torch.isfinite(node_distances)
    optimal_edges = (
        adjacency.gt(0)
        & finite_nodes
        & torch.isclose(candidate_costs, node_distances, atol=tolerance, rtol=tolerance)
    )
    mask = torch.zeros(
        batch_size,
        num_nodes,
        num_nodes + 1,
        dtype=torch.bool,
        device=adjacency.device,
    )
    mask[:, :, :num_nodes] = optimal_edges
    mask[:, :, num_nodes] = ~torch.isfinite(distances)
    batch_indices = torch.arange(batch_size, device=adjacency.device)
    mask[batch_indices, source] = False
    mask[batch_indices, source, source] = True
    return mask


def dijkstra_correct_mask(
    predictions: torch.Tensor,
    adjacency: torch.Tensor,
    distances: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Return whether every predicted parent is shortest-path optimal. Shape: [B]."""
    optimal = optimal_parent_mask(adjacency, distances, source)
    selected = optimal.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    return selected.all(dim=-1)


def compute_dijkstra_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    adjacency: torch.Tensor,
    distances: torch.Tensor,
    source: torch.Tensor,
) -> DijkstraMetrics:
    """Compute canonical-label diagnostics and tie-tolerant task metrics."""
    legal = (
        valid_parent_mask(adjacency, source)
        .gather(-1, predictions.unsqueeze(-1))
        .squeeze(-1)
    )
    optimal = (
        optimal_parent_mask(adjacency, distances, source)
        .gather(-1, predictions.unsqueeze(-1))
        .squeeze(-1)
    )
    exact = optimal.all(dim=-1)
    return DijkstraMetrics(
        node_accuracy=float(predictions.eq(targets).float().mean().item()),
        legal_parent_rate=float(legal.float().mean().item()),
        optimal_parent_rate=float(optimal.float().mean().item()),
        exact_tree_rate=float(exact.float().mean().item()),
    )


def distances_from_sample(values: list[float]) -> torch.Tensor:
    """Convert serialized distances while preserving infinity."""
    return torch.tensor(
        [value if math.isfinite(value) else float("inf") for value in values],
        dtype=torch.float32,
    )


__all__ = [
    "DijkstraMetrics",
    "compute_dijkstra_metrics",
    "constrain_parent_logits",
    "dijkstra_correct_mask",
    "distances_from_sample",
    "encode_parent_targets",
    "optimal_parent_mask",
    "valid_parent_mask",
]
