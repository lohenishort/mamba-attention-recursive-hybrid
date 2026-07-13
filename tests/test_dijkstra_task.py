import torch

from mamba_hybrid.tasks.dijkstra import (
    compute_dijkstra_metrics,
    constrain_parent_logits,
    dijkstra_correct_mask,
    encode_parent_targets,
)


def test_dijkstra_targets_distinguish_source_and_unreachable() -> None:
    targets = encode_parent_targets([-1, 0, -1], source=0)

    assert torch.equal(targets, torch.tensor([0, 0, 3]))


def test_dijkstra_constraints_reject_non_edge_parents() -> None:
    adjacency = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    logits = torch.zeros(1, 3, 4)

    constrained = constrain_parent_logits(logits, adjacency, torch.tensor([0]))

    assert constrained[0, 0].argmax().item() == 0
    assert constrained[0, 2, 0] == torch.finfo(logits.dtype).min
    assert constrained[0, 2, 1] == 0
    assert constrained[0, 2, 3] == 0


def test_dijkstra_metrics_accept_tied_optimal_trees() -> None:
    adjacency = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    distances = torch.tensor([[0.0, 1.0, 1.0, 2.0]])
    targets = torch.tensor([[0, 0, 0, 1]])
    alternative = torch.tensor([[0, 0, 0, 2]])
    source = torch.tensor([0])

    assert dijkstra_correct_mask(alternative, adjacency, distances, source).item()
    metrics = compute_dijkstra_metrics(
        alternative, targets, adjacency, distances, source
    )
    assert metrics.node_accuracy == 0.75
    assert metrics.optimal_parent_rate == 1.0
    assert metrics.exact_tree_rate == 1.0
