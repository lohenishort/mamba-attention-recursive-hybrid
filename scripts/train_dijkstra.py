"""Train the task-native Dijkstra shortest-path-tree model."""

import os
import random
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.printer import AutoregressivePrinter
from mamba_hybrid.tasks.common import shift_targets_right
from mamba_hybrid.tasks.dijkstra import (
    compute_dijkstra_metrics,
    constrain_parent_logits,
    dijkstra_correct_mask,
    distances_from_sample,
    encode_parent_targets,
    valid_parent_mask,
)
from scripts.utils import deterministic_split_indices, seed_everything

DijkstraBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def augment_dijkstra_example(sample: dict[str, Any]) -> dict[str, Any]:
    """Relabel every graph field consistently, including source and distances."""
    adjacency = sample["adjacency"]
    parents = sample["parents"]
    distances = sample["distances"]
    num_nodes = len(adjacency)
    source = int(sample.get("source", 0))
    permutation = list(range(num_nodes))
    random.shuffle(permutation)

    new_adjacency = [[0.0] * num_nodes for _ in range(num_nodes)]
    new_parents = [-1] * num_nodes
    new_distances = [float("inf")] * num_nodes
    for old_node in range(num_nodes):
        new_node = permutation[old_node]
        new_distances[new_node] = distances[old_node]
        old_parent = parents[old_node]
        if old_parent != -1:
            new_parents[new_node] = permutation[old_parent]
        for old_neighbor in range(num_nodes):
            new_adjacency[new_node][permutation[old_neighbor]] = adjacency[old_node][
                old_neighbor
            ]

    return {
        "schema_version": 2,
        "adjacency": new_adjacency,
        "distances": new_distances,
        "parents": new_parents,
        "source": permutation[source],
    }


class DijkstraDataset(Dataset[DijkstraBatch]):
    """Return raw graphs, explicit sources, parent targets, and shortest distances."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        augment: bool = False,
        num_nodes: int = 20,
    ) -> None:
        self.samples = samples
        self.augment = augment
        self.num_nodes = num_nodes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DijkstraBatch:
        sample = self.samples[index]
        if self.augment:
            sample = augment_dijkstra_example(sample)
        adjacency = torch.tensor(sample["adjacency"], dtype=torch.float32)
        if adjacency.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("Dijkstra adjacency size does not match num_nodes")
        source = int(sample.get("source", 0))
        targets = encode_parent_targets(sample["parents"], source)
        distances = distances_from_sample(sample["distances"])
        return adjacency, torch.tensor(source), targets, distances


class DijkstraReasoningModel(nn.Module):
    """Adapt weighted adjacency rows to the shared recursive reasoning core."""

    def __init__(
        self,
        config: MambaHybridConfig,
        num_nodes: int = 20,
        *,
        reasoning_encoder: MambaAttentionHybrid | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.edge_proj = nn.Linear(2 * num_nodes, config.d_model)
        self.node_embed = nn.Parameter(torch.randn(1, num_nodes, config.d_model))
        self.source_embed = nn.Embedding(2, config.d_model)
        if reasoning_encoder is None:
            self.reasoning_encoder = MambaAttentionHybrid(config)
        else:
            object.__setattr__(self, "reasoning_encoder", reasoning_encoder)
        self.bos_token = num_nodes + 1
        self.printer = AutoregressivePrinter(
            config,
            vocab_size=num_nodes + 2,
            output_vocab_size=num_nodes + 1,
            max_length=num_nodes,
            pad_token_id=num_nodes,
        )

    def encode_inputs(
        self, adjacency: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Encode weights, edge presence, node identity, and source identity. [B,N,D]."""
        if adjacency.ndim != 3 or adjacency.shape[1:] != (
            self.num_nodes,
            self.num_nodes,
        ):
            raise ValueError(
                "adjacency must have shape [batch_size, num_nodes, num_nodes]"
            )
        scale = adjacency.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        features = torch.cat(
            [adjacency / scale, adjacency.gt(0).to(adjacency.dtype)], dim=-1
        )
        source_flags = torch.zeros(
            adjacency.shape[0],
            self.num_nodes,
            dtype=torch.long,
            device=adjacency.device,
        )
        source_flags.scatter_(1, source.unsqueeze(1), 1)
        return cast(
            torch.Tensor,
            self.edge_proj(features)
            + self.node_embed
            + self.source_embed(source_flags),
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        source: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_raw = self.encode_inputs(adjacency, source)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["DIJKSTRA"] * x_raw.shape[0]
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        logits = self.printer(prefix, decoder_input_ids, prefix_mask)
        return constrain_parent_logits(logits, adjacency, source), probabilities

    def forward_cycle_logits(
        self,
        adjacency: torch.Tensor,
        source: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Decode every completed cycle for ACT correctness targets."""
        x_raw = self.encode_inputs(adjacency, source)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["DIJKSTRA"] * x_raw.shape[0]
        )
        cycle_logits: list[torch.Tensor] = []
        for state in states:
            prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
                x_raw, state
            )
            logits = self.printer(prefix, decoder_input_ids, prefix_mask)
            cycle_logits.append(constrain_parent_logits(logits, adjacency, source))
        return cycle_logits, probabilities

    @torch.no_grad()
    def generate(
        self, adjacency: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate one structurally legal parent token per node."""
        x_raw = self.encode_inputs(adjacency, source)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["DIJKSTRA"] * x_raw.shape[0]
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        decoder_input = torch.full(
            (adjacency.shape[0], 1),
            self.bos_token,
            dtype=torch.long,
            device=adjacency.device,
        )
        legal = valid_parent_mask(adjacency, source)
        outputs: list[torch.Tensor] = []
        for node in range(self.num_nodes):
            logits = self.printer(prefix, decoder_input, prefix_mask)[:, -1]
            logits = logits.masked_fill(~legal[:, node], torch.finfo(logits.dtype).min)
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1), probabilities


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

    num_nodes = 20
    config = MambaHybridConfig(
        d_model=128,
        n_meta=32,
        l_ans=num_nodes,
        n_steps=4,
        t_cycles=3,
        M_min=1,
        M_max=3,
        vocab_size=num_nodes + 1,
    )
    samples: list[dict[str, Any]] = torch.load(data_path)[:1000]
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(len(samples), seed)
    train_set = DijkstraDataset(
        [samples[index] for index in train_indices], augment=True, num_nodes=num_nodes
    )
    validation_set = DijkstraDataset(
        [samples[index] for index in validation_indices], num_nodes=num_nodes
    )
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)

    model = DijkstraReasoningModel(config, num_nodes=num_nodes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        train_nodes = 0
        train_correct_nodes = 0
        train_graphs = 0
        train_correct_graphs = 0
        for adjacency, source, targets, distances in train_loader:
            adjacency = adjacency.to(device)
            source = source.to(device)
            targets = targets.to(device)
            distances = distances.to(device)
            optimizer.zero_grad()
            decoder_inputs = shift_targets_right(
                targets,
                bos_token_id=model.bos_token,
                pad_token_id=num_nodes,
            )
            cycle_logits, probabilities = model.forward_cycle_logits(
                adjacency, source, decoder_inputs
            )
            logits = cycle_logits[-1]
            predictions = logits.argmax(dim=-1)
            cycle_correct = torch.stack(
                [
                    dijkstra_correct_mask(
                        cycle_prediction.argmax(dim=-1),
                        adjacency,
                        distances,
                        source,
                    )
                    for cycle_prediction in cycle_logits
                ]
            )
            loss = compute_bce_joint_loss(
                logits,
                targets,
                probabilities,
                cycle_correct.float(),
                alpha=1.0,
                min_cycles=config.M_min,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * adjacency.shape[0]
            train_correct_nodes += int(predictions.eq(targets).sum().item())
            train_nodes += targets.numel()
            train_correct_graphs += int(cycle_correct[-1].sum().item())
            train_graphs += adjacency.shape[0]

        model.eval()
        validation_predictions: list[torch.Tensor] = []
        validation_targets: list[torch.Tensor] = []
        validation_adjacency: list[torch.Tensor] = []
        validation_distances: list[torch.Tensor] = []
        validation_sources: list[torch.Tensor] = []
        with torch.no_grad():
            for adjacency, source, targets, distances in validation_loader:
                adjacency = adjacency.to(device)
                source = source.to(device)
                targets = targets.to(device)
                decoder_inputs = shift_targets_right(
                    targets,
                    bos_token_id=model.bos_token,
                    pad_token_id=num_nodes,
                )
                validation_predictions.append(
                    model(adjacency, source, decoder_inputs)[0].argmax(dim=-1).cpu()
                )
                validation_targets.append(targets.cpu())
                validation_adjacency.append(adjacency.cpu())
                validation_distances.append(distances)
                validation_sources.append(source.cpu())
        metrics = compute_dijkstra_metrics(
            torch.cat(validation_predictions),
            torch.cat(validation_targets),
            torch.cat(validation_adjacency),
            torch.cat(validation_distances),
            torch.cat(validation_sources),
        )
        print(
            f"Epoch {epoch:02d}/20 | Loss: {train_loss / len(train_set):.4f} | "
            f"Train Node Acc: {train_correct_nodes / train_nodes:.4f} | "
            f"Train Optimal Trees: {train_correct_graphs / train_graphs:.4f} | "
            f"Val Node Acc: {metrics.node_accuracy:.4f} | "
            f"Val Optimal Parent: {metrics.optimal_parent_rate:.4f} | "
            f"Val Optimal Trees: {metrics.exact_tree_rate:.4f}",
            flush=True,
        )

    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "task": "dijkstra",
            "state_dict": model.state_dict(),
            "config": vars(config),
            "task_config": {"num_nodes": num_nodes, "unreachable_token": num_nodes},
            "seed": seed,
            "dataset": data_path,
            "validation_indices": validation_indices,
        },
        "data/dijkstra_model.pt",
    )


if __name__ == "__main__":
    main()
