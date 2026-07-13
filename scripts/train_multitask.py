"""Task-native multitask training with homogeneous batches and one shared planner."""

import json
import os
from collections.abc import Iterator, Mapping
from itertools import zip_longest
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.tasks.common import shift_targets_right
from mamba_hybrid.tasks.dijkstra import dijkstra_correct_mask
from mamba_hybrid.tasks.gsm8k import VOCAB_SIZE as BYTE_VOCAB_SIZE
from mamba_hybrid.tasks.maze import PAD as MAZE_PAD
from mamba_hybrid.tasks.maze import maze_correct_mask
from scripts.train_dijkstra import DijkstraDataset, DijkstraReasoningModel
from scripts.train_gsm8k import (
    GSM8KDataset,
    GSM8KReasoningModel,
    collate_gsm8k,
)
from scripts.train_maze import MazeDataset, MazeReasoningModel
from scripts.train_sudoku import (
    SudokuDataset,
    SudokuReasoningModel,
    sudoku_completion_targets,
)
from scripts.utils import seed_everything

TaskName = Literal["SUDOKU", "DIJKSTRA", "MAZE", "GSM8K"]
TaskBatch = Any


class NativeMultiTaskModel(nn.Module):
    """Route task-native adapters and printers through one MoE planning core."""

    def __init__(
        self,
        config: MambaHybridConfig,
        *,
        grid_size: int = 10,
        num_nodes: int = 20,
        max_question_bytes: int = 1024,
        max_gsm_answer_length: int = 16,
    ) -> None:
        super().__init__()
        if not config.use_moe:
            raise ValueError("native multitask training requires use_moe=True")
        self.reasoning_encoder = MambaAttentionHybrid(config)
        self.sudoku = SudokuReasoningModel(
            config, reasoning_encoder=self.reasoning_encoder
        )
        self.dijkstra = DijkstraReasoningModel(
            config, num_nodes=num_nodes, reasoning_encoder=self.reasoning_encoder
        )
        self.maze = MazeReasoningModel(
            config, grid_size=grid_size, reasoning_encoder=self.reasoning_encoder
        )
        self.gsm8k = GSM8KReasoningModel(
            config,
            max_question_bytes=max_question_bytes,
            max_answer_length=max_gsm_answer_length,
            reasoning_encoder=self.reasoning_encoder,
        )

    def forward_task(
        self, task_name: TaskName, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run one homogeneous task batch through its native adapter and printer."""
        if task_name == "SUDOKU":
            return cast(
                tuple[torch.Tensor, list[torch.Tensor]],
                self.sudoku(batch["input_ids"], batch["decoder_input_ids"]),
            )
        if task_name == "DIJKSTRA":
            return cast(
                tuple[torch.Tensor, list[torch.Tensor]],
                self.dijkstra(
                    batch["adjacency"], batch["source"], batch["decoder_input_ids"]
                ),
            )
        if task_name == "MAZE":
            return cast(
                tuple[torch.Tensor, list[torch.Tensor]],
                self.maze(batch["grid"], batch["decoder_input_ids"]),
            )
        if task_name == "GSM8K":
            return cast(
                tuple[torch.Tensor, list[torch.Tensor]],
                self.gsm8k(
                    batch["question_ids"],
                    batch["question_mask"],
                    batch["decoder_input_ids"],
                ),
            )
        raise ValueError(f"unsupported task: {task_name}")


def round_robin_batches(
    loaders: Mapping[TaskName, DataLoader[Any]],
) -> Iterator[tuple[TaskName, TaskBatch]]:
    """Yield one homogeneous batch per task in deterministic round-robin order."""
    names = list(loaders)
    iterators = [iter(loaders[name]) for name in names]
    for row in zip_longest(*iterators):
        for name, batch in zip(names, row):
            if batch is not None:
                yield name, batch


def _sequence_correct(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = targets.ne(-100)
    return (logits.argmax(dim=-1).eq(targets) | ~mask).all(dim=-1)


def native_task_loss(
    model: NativeMultiTaskModel,
    task_name: TaskName,
    batch: TaskBatch,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Compute final-step task CE plus per-cycle binary ACT supervision."""
    if task_name == "SUDOKU":
        input_ids, targets = (tensor.to(device) for tensor in batch)
        decoder_inputs = shift_targets_right(
            targets,
            bos_token_id=model.sudoku.bos_token,
            pad_token_id=model.sudoku.pad_token,
        )
        cycle_logits, probabilities = model.sudoku.forward_cycle_logits(
            input_ids, decoder_inputs
        )
        cycle_correct = torch.stack(
            [logits.argmax(dim=-1).eq(targets).all(dim=-1) for logits in cycle_logits]
        )
        loss_targets = sudoku_completion_targets(input_ids, targets)
        ignore_index = -100
    elif task_name == "DIJKSTRA":
        adjacency, source, targets, distances = (tensor.to(device) for tensor in batch)
        decoder_inputs = shift_targets_right(
            targets,
            bos_token_id=model.dijkstra.bos_token,
            pad_token_id=model.dijkstra.num_nodes,
        )
        cycle_logits, probabilities = model.dijkstra.forward_cycle_logits(
            adjacency, source, decoder_inputs
        )
        cycle_correct = torch.stack(
            [
                dijkstra_correct_mask(
                    logits.argmax(dim=-1), adjacency, distances, source
                )
                for logits in cycle_logits
            ]
        )
        loss_targets = targets
        ignore_index = -100
    elif task_name == "MAZE":
        grids, targets = (tensor.to(device) for tensor in batch)
        decoder_inputs = shift_targets_right(
            targets,
            bos_token_id=model.maze.bos_token,
            pad_token_id=MAZE_PAD,
        )
        cycle_logits, probabilities = model.maze.forward_cycle_logits(
            grids, decoder_inputs
        )
        size = model.maze.grid_size
        cycle_correct = torch.stack(
            [
                maze_correct_mask(logits.argmax(dim=-1), grids.view(-1, size, size))
                for logits in cycle_logits
            ]
        )
        loss_targets = targets
        ignore_index = MAZE_PAD
    else:
        questions, question_mask, decoder_inputs, targets = (
            tensor.to(device) for tensor in batch
        )
        cycle_logits, probabilities = model.gsm8k.forward_cycle_logits(
            questions, question_mask, decoder_inputs
        )
        cycle_correct = torch.stack(
            [_sequence_correct(logits, targets) for logits in cycle_logits]
        )
        loss_targets = targets
        ignore_index = -100

    loss = compute_bce_joint_loss(
        cycle_logits[-1],
        loss_targets,
        probabilities,
        cycle_correct.float(),
        alpha=1.0,
        ignore_index=ignore_index,
        min_cycles=model.reasoning_encoder.config.M_min,
    )
    return loss, float(cycle_correct[-1].float().mean().item())


def main() -> None:
    required = [
        "data/sudoku.jsonl",
        "data/dijkstra.pt",
        "data/maze_dryrun.pt",
        "data/gsm8k_train.jsonl",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing multitask datasets: {', '.join(missing)}")
    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_everything(42)

    with open("data/sudoku.jsonl") as sudoku_file:
        sudoku_samples = [json.loads(line) for _, line in zip(range(1000), sudoku_file)]
    dijkstra_samples: list[dict[str, Any]] = torch.load("data/dijkstra.pt")[:1000]
    maze_samples: list[dict[str, Any]] = torch.load("data/maze_dryrun.pt")
    grid_size = len(maze_samples[0]["grid"])
    maze_length = max(len(sample["path"]) for sample in maze_samples)

    config = MambaHybridConfig(
        d_model=64,
        n_meta=16,
        l_ans=81,
        n_steps=2,
        t_cycles=2,
        M_min=1,
        M_max=2,
        use_moe=True,
        vocab_size=BYTE_VOCAB_SIZE,
    )
    model = NativeMultiTaskModel(config, grid_size=grid_size).to(device)
    loaders: dict[TaskName, DataLoader[Any]] = {
        "SUDOKU": DataLoader(
            SudokuDataset(sudoku_samples, augment=True), batch_size=8, shuffle=True
        ),
        "DIJKSTRA": DataLoader(
            DijkstraDataset(dijkstra_samples, augment=True), batch_size=8, shuffle=True
        ),
        "MAZE": DataLoader(
            MazeDataset(
                "data/maze_dryrun.pt",
                size=grid_size,
                max_path_len=maze_length,
            ),
            batch_size=8,
            shuffle=True,
        ),
        "GSM8K": DataLoader(
            GSM8KDataset("data/gsm8k_train.jsonl"),
            batch_size=8,
            shuffle=True,
            collate_fn=collate_gsm8k,
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    for epoch in range(1, 6):
        model.train()
        totals = {name: 0.0 for name in loaders}
        counts = {name: 0 for name in loaders}
        for task_name, batch in round_robin_batches(loaders):
            optimizer.zero_grad()
            loss, exact = native_task_loss(model, task_name, batch, device)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            totals[task_name] += exact
            counts[task_name] += 1
        summary = " | ".join(
            f"{name} Exact: {totals[name] / counts[name]:.4f}" for name in loaders
        )
        print(f"Epoch {epoch}/5 | {summary}", flush=True)

    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "task": "multitask_native",
            "state_dict": model.state_dict(),
            "config": vars(config),
            "task_config": {
                "tasks": list(loaders),
                "batching": "homogeneous_round_robin",
                "target_encodings": {
                    "SUDOKU": "blank_digits_autoregressive_v2",
                    "DIJKSTRA": "parents_with_unreachable_v2",
                    "MAZE": "moves_with_eos_v1",
                    "GSM8K": "normalized_integer_bytes_v1",
                },
            },
        },
        "data/unified_model.pt",
    )


UnifiedReasoningLLM = NativeMultiTaskModel

__all__ = [
    "NativeMultiTaskModel",
    "TaskName",
    "UnifiedReasoningLLM",
    "native_task_loss",
    "round_robin_batches",
]


if __name__ == "__main__":
    main()
