"""Train the task-native maze action model."""

import os
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.printer import AutoregressivePrinter
from mamba_hybrid.tasks.common import shift_targets_right
from mamba_hybrid.tasks.maze import (
    EOS,
    PAD,
    VOCAB_SIZE,
    maze_correct_mask,
    pad_moves,
    path_to_moves,
)
from scripts.utils import deterministic_split_indices, seed_everything

MazeBatch = tuple[torch.Tensor, torch.Tensor]


class MazeDataset(Dataset[MazeBatch]):
    """Encode shortest paths as directional actions followed by EOS."""

    def __init__(
        self, data_path: str, size: int | None = None, max_path_len: int = 64
    ) -> None:
        self.samples: list[dict[str, Any]] = torch.load(data_path)
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

    def __getitem__(self, index: int) -> MazeBatch:
        sample = self.samples[index]
        grid = torch.tensor(sample["grid"], dtype=torch.long)
        moves = path_to_moves([tuple(coordinate) for coordinate in sample["path"]])
        return grid.flatten(), pad_moves(moves, self.max_path_len)


class MazeReasoningModel(nn.Module):
    """Adapt a marked maze grid to move-token predictions."""

    def __init__(
        self,
        config: MambaHybridConfig,
        grid_size: int = 30,
        *,
        reasoning_encoder: MambaAttentionHybrid | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.grid_size = grid_size
        self.state_embed = nn.Embedding(4, config.d_model)
        self.row_embed = nn.Parameter(torch.randn(1, grid_size, config.d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(1, grid_size, config.d_model // 2))
        if reasoning_encoder is None:
            self.reasoning_encoder = MambaAttentionHybrid(config)
        else:
            object.__setattr__(self, "reasoning_encoder", reasoning_encoder)
        self.bos_token = VOCAB_SIZE
        self.printer = AutoregressivePrinter(
            config,
            vocab_size=VOCAB_SIZE + 1,
            output_vocab_size=VOCAB_SIZE,
            max_length=config.l_ans,
            pad_token_id=PAD,
        )

    def encode_inputs(self, grid_flat: torch.Tensor) -> torch.Tensor:
        """Mark open/wall/start/goal cells and add 2D positions. [B,S*S,D]."""
        if grid_flat.ndim != 2 or grid_flat.shape[1] != self.grid_size**2:
            raise ValueError("grid_flat must have shape [batch_size, grid_size ** 2]")
        batch_size = grid_flat.shape[0]
        states = grid_flat.long().clone()
        states[:, 0] = 2
        states[:, -1] = 3
        state_features = self.state_embed(states)
        row_positions = self.row_embed.unsqueeze(2).expand(
            batch_size, self.grid_size, self.grid_size, -1
        )
        column_positions = self.col_embed.unsqueeze(1).expand(
            batch_size, self.grid_size, self.grid_size, -1
        )
        positions = torch.cat([row_positions, column_positions], dim=-1).view(
            batch_size, self.grid_size**2, -1
        )
        return cast(torch.Tensor, state_features + positions)

    def forward(
        self, grid_flat: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_raw = self.encode_inputs(grid_flat)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["MAZE"] * x_raw.shape[0]
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        return self.printer(prefix, decoder_input_ids, prefix_mask), probabilities

    def forward_cycle_logits(
        self, grid_flat: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Decode every completed cycle for ACT correctness targets."""
        x_raw = self.encode_inputs(grid_flat)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["MAZE"] * x_raw.shape[0]
        )
        cycle_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for state in states[:-1]:
                prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
                    x_raw, state
                )
                cycle_logits.append(
                    self.printer(prefix, decoder_input_ids, prefix_mask)
                )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        cycle_logits.append(self.printer(prefix, decoder_input_ids, prefix_mask))
        return cycle_logits, probabilities

    @torch.no_grad()
    def generate(
        self, grid_flat: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate moves causally until EOS, never emitting PAD as an action."""
        x_raw = self.encode_inputs(grid_flat)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["MAZE"] * x_raw.shape[0]
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        allowed = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=grid_flat.device)
        allowed[PAD] = False
        tokens = self.printer.generate(
            prefix,
            bos_token_id=self.bos_token,
            eos_token_id=EOS,
            prefix_mask=prefix_mask,
            allowed_tokens=allowed,
            max_new_tokens=self.config.l_ans,
        )
        return tokens, probabilities


def main(
    *,
    data_path: str = "data/maze_dryrun.pt",
    checkpoint_path: str = "data/maze_model.pt",
    d_model: int = 64,
    n_meta: int = 16,
    n_steps: int = 2,
    max_cycles: int = 2,
    batch_size: int = 16,
    epochs: int = 20,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found; run the data generation command first"
        )
    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    raw_samples: list[dict[str, Any]] = torch.load(data_path)
    grid_size = len(raw_samples[0]["grid"])
    # moves + EOS has the same length as a coordinate path including its start.
    answer_length = max(len(sample["path"]) for sample in raw_samples)
    config = MambaHybridConfig(
        d_model=d_model,
        n_meta=n_meta,
        l_ans=answer_length,
        n_steps=n_steps,
        t_cycles=max_cycles,
        M_min=1,
        M_max=max_cycles,
        vocab_size=VOCAB_SIZE,
    )
    dataset = MazeDataset(data_path, size=grid_size, max_path_len=answer_length)
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(len(dataset), seed)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    validation_set = torch.utils.data.Subset(dataset, validation_indices)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model = MazeReasoningModel(config, grid_size=grid_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        solved = 0
        samples_seen = 0
        for grids, targets in train_loader:
            grids = grids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            decoder_inputs = shift_targets_right(
                targets, bos_token_id=model.bos_token, pad_token_id=PAD
            )
            cycle_logits, probabilities = model.forward_cycle_logits(
                grids, decoder_inputs
            )
            logits = cycle_logits[-1]
            predictions = logits.argmax(dim=-1)
            cycle_correct = torch.stack(
                [
                    maze_correct_mask(
                        cycle_prediction.argmax(dim=-1),
                        grids.view(-1, grid_size, grid_size),
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
                ignore_index=PAD,
                min_cycles=config.M_min,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            target_mask = targets.ne(PAD)
            correct_tokens += int(
                predictions[target_mask].eq(targets[target_mask]).sum().item()
            )
            total_tokens += int(target_mask.sum().item())
            solved += int(cycle_correct[-1].sum().item())
            samples_seen += grids.shape[0]
            total_loss += loss.item() * grids.shape[0]

        model.eval()
        validation_solved = 0
        validation_exact = 0
        validation_tokens = 0
        validation_correct_tokens = 0
        with torch.no_grad():
            for grids, targets in validation_loader:
                grids = grids.to(device)
                targets = targets.to(device)
                decoder_inputs = shift_targets_right(
                    targets, bos_token_id=model.bos_token, pad_token_id=PAD
                )
                predictions = model(grids, decoder_inputs)[0].argmax(dim=-1)
                validation_solved += int(
                    maze_correct_mask(predictions, grids.view(-1, grid_size, grid_size))
                    .sum()
                    .item()
                )
                target_mask = targets.ne(PAD)
                validation_exact += int(
                    (predictions.eq(targets) | ~target_mask).all(dim=-1).sum().item()
                )
                validation_correct_tokens += int(
                    predictions[target_mask].eq(targets[target_mask]).sum().item()
                )
                validation_tokens += int(target_mask.sum().item())
        print(
            f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(train_set):.4f} | "
            f"Train Move Acc: {correct_tokens / total_tokens:.4f} | "
            f"Train Solve: {solved / samples_seen:.4f} | "
            f"Val Move Acc: {validation_correct_tokens / validation_tokens:.4f} | "
            f"Val Reference Exact: {validation_exact / len(validation_set):.4f} | "
            f"Val Solve: {validation_solved / len(validation_set):.4f}",
            flush=True,
        )

    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "task": "maze",
            "state_dict": model.state_dict(),
            "config": vars(config),
            "task_config": {
                "grid_size": grid_size,
                "target_encoding": "moves_v1",
                "padding_index": PAD,
            },
            "dataset": data_path,
            "seed": seed,
            "validation_indices": validation_indices,
        },
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
