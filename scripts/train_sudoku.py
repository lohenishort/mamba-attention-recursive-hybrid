import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, cast

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.printer import AutoregressivePrinter
from mamba_hybrid.tasks.common import shift_targets_right
from scripts.utils import deterministic_split_indices, seed_everything


# --- 1. Custom Dataset for Sudoku ---
import random


def augment_sudoku(
    puzzle: List[List[int]], solution: List[List[int]]
) -> Tuple[List[int], List[int]]:
    # 1. Permute numbers 1-9
    digits = list(range(1, 10))
    shuffled = digits.copy()
    random.shuffle(shuffled)
    mapping = {i: shuffled[i - 1] for i in range(1, 10)}
    mapping[0] = 0

    p_grid = [[mapping[val] for val in row] for row in puzzle]
    s_grid = [[mapping[val] for val in row] for row in solution]

    # 2. Transposition (50% chance)
    if random.random() < 0.5:
        p_grid = [list(x) for x in zip(*p_grid)]
        s_grid = [list(x) for x in zip(*s_grid)]

    # 3. Flips (horizontal / vertical)
    if random.random() < 0.5:
        p_grid = p_grid[::-1]
        s_grid = s_grid[::-1]
    if random.random() < 0.5:
        p_grid = [row[::-1] for row in p_grid]
        s_grid = [row[::-1] for row in s_grid]

    p_flat = [val for row in p_grid for val in row]
    s_flat = [val for row in s_grid for val in row]
    return p_flat, s_flat


class SudokuDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: List[Dict[str, Any]], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        puzzle = sample["puzzle"]  # 9x9 list of ints
        solution = sample["solution"]  # 9x9 list of ints

        if self.augment:
            puzzle_flat, solution_flat = augment_sudoku(puzzle, solution)
        else:
            puzzle_flat = [val for row in puzzle for val in row]
            solution_flat = [val for row in solution for val in row]

        return torch.tensor(puzzle_flat, dtype=torch.long), torch.tensor(
            solution_flat, dtype=torch.long
        )


def sudoku_completion_targets(
    input_ids: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Ignore fixed clues so task loss supervises only blank cells. [B, 81]."""
    if input_ids.shape != target_ids.shape:
        raise ValueError("input_ids and target_ids must have matching shapes")
    return target_ids.masked_fill(input_ids.ne(0), -100)


# --- 2. Sudoku Reasoning Model ---
class SudokuReasoningModel(nn.Module):
    def __init__(
        self,
        config: MambaHybridConfig,
        vocab_size: int = 10,
        *,
        reasoning_encoder: MambaAttentionHybrid | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        # 2D row & col positional embeddings
        self.row_embed = nn.Parameter(torch.randn(9, config.d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(9, config.d_model // 2))
        if reasoning_encoder is None:
            self.reasoning_encoder = MambaAttentionHybrid(config)
        else:
            object.__setattr__(self, "reasoning_encoder", reasoning_encoder)
        self.bos_token = vocab_size
        self.pad_token = vocab_size + 1
        self.printer = AutoregressivePrinter(
            config,
            vocab_size=vocab_size + 2,
            output_vocab_size=vocab_size,
            max_length=81,
            pad_token_id=self.pad_token,
        )

    @staticmethod
    def _apply_clues(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        clue_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        clue_logits.scatter_(-1, input_ids.unsqueeze(-1), 0.0)
        return torch.where(input_ids.ne(0).unsqueeze(-1), clue_logits, logits)

    def _encode_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        row_pos = self.row_embed.unsqueeze(1).expand(-1, 9, -1)
        col_pos = self.col_embed.unsqueeze(0).expand(9, -1, -1)
        pos_2d = (
            torch.cat([row_pos, col_pos], dim=-1)
            .view(81, -1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return cast(torch.Tensor, self.embed(input_ids) + pos_2d)

    def _generate_from_prefix(
        self,
        prefix: torch.Tensor,
        prefix_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        cache = self.printer.prefill(prefix, prefix_mask, capacity=81)
        current = torch.full(
            (batch_size,),
            self.bos_token,
            dtype=torch.long,
            device=input_ids.device,
        )
        outputs: list[torch.Tensor] = []
        for position in range(81):
            logits, cache = self.printer.decode_step(current, cache, position=position)
            clue = input_ids[:, position]
            forced = torch.full_like(logits, torch.finfo(logits.dtype).min)
            forced.scatter_(-1, clue.unsqueeze(-1), 0.0)
            logits = torch.where(clue.ne(0).unsqueeze(-1), forced, logits)
            outputs.append(logits)
            next_token = logits.argmax(dim=-1)
            current = next_token
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # input_ids shape: [B, 81]
        x_raw = self._encode_inputs(input_ids)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["SUDOKU"] * x_raw.shape[0]
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        if decoder_input_ids is None:
            logits = self._generate_from_prefix(prefix, prefix_mask, input_ids)
        else:
            logits = self._apply_clues(
                self.printer(prefix, decoder_input_ids, prefix_mask), input_ids
            )
        return logits, probabilities

    def forward_cycle_logits(
        self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Decode every completed cycle for ACT correctness targets."""
        x_raw = self._encode_inputs(input_ids)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw, task_names=["SUDOKU"] * x_raw.shape[0]
        )
        cycle_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for state in states[:-1]:
                prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
                    x_raw, state
                )
                cycle_logits.append(
                    self._apply_clues(
                        self.printer(prefix, decoder_input_ids, prefix_mask), input_ids
                    )
                )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1]
        )
        cycle_logits.append(
            self._apply_clues(
                self.printer(prefix, decoder_input_ids, prefix_mask), input_ids
            )
        )
        return cycle_logits, probabilities


# --- 3. Main Training Driver ---
def main() -> None:
    data_path = "data/sudoku.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run download_all_datasets first.")
        return

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Configure model for exactly 81 answer slots
    l_ans = 81
    config = MambaHybridConfig(
        d_model=128,
        n_meta=32,
        l_ans=l_ans,
        n_steps=4,
        t_cycles=3,
        M_min=1,
        M_max=3,
        vocab_size=10,
    )

    # Load all Sudoku samples
    all_samples: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            all_samples.append(json.loads(line))
            if len(all_samples) >= 50000:
                break

    # Shuffle and split
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(
        len(all_samples), seed
    )
    train_samples = [all_samples[index] for index in train_indices]
    val_samples_list = [all_samples[index] for index in validation_indices]

    train_set = SudokuDataset(train_samples, augment=True)
    val_set = SudokuDataset(val_samples_list, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Initialize model & optimizer
    model = SudokuReasoningModel(config, vocab_size=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    epochs = 100
    print(f"Starting training on {len(train_set)} Sudoku puzzles...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        correct_cells = 0
        correct_blank_cells = 0
        total_blank_cells = 0
        total_samples = 0

        for input_ids, target_ids in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            decoder_inputs = shift_targets_right(
                target_ids,
                bos_token_id=model.bos_token,
                pad_token_id=model.pad_token,
            )
            cycle_logits, bce_probs = model.forward_cycle_logits(
                input_ids, decoder_inputs
            )
            logits = cycle_logits[-1]

            # Check if predicted board matches solution exactly
            preds = logits.argmax(dim=-1)
            cycle_correct = torch.stack(
                [
                    cycle_prediction.argmax(dim=-1).eq(target_ids).all(dim=-1)
                    for cycle_prediction in cycle_logits
                ]
            )
            is_correct = cycle_correct[-1]
            completion_targets = sudoku_completion_targets(input_ids, target_ids)

            loss = compute_bce_joint_loss(
                logits,
                completion_targets,
                bce_probs,
                cycle_correct.float(),
                alpha=1.0,
                min_cycles=config.M_min,
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            correct_count += int(is_correct.sum().item())
            correct_cells += int(preds.eq(target_ids).sum().item())
            blank_mask = input_ids.eq(0)
            correct_blank_cells += int(
                preds[blank_mask].eq(target_ids[blank_mask]).sum().item()
            )
            total_blank_cells += int(blank_mask.sum().item())
            total_samples += input_ids.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_count / total_samples
        train_cell_acc = correct_cells / (total_samples * l_ans)
        train_blank_acc = correct_blank_cells / total_blank_cells

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_correct_cells = 0
        val_correct_blank_cells = 0
        val_total_blank_cells = 0
        val_samples = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                decoder_inputs = shift_targets_right(
                    target_ids,
                    bos_token_id=model.bos_token,
                    pad_token_id=model.pad_token,
                )
                logits, bce_probs = model(input_ids, decoder_inputs)
                preds = logits.argmax(dim=-1)
                is_correct = (preds == target_ids).all(dim=-1)
                completion_targets = sudoku_completion_targets(input_ids, target_ids)
                loss = compute_bce_joint_loss(
                    logits,
                    completion_targets,
                    bce_probs,
                    is_correct.float(),
                    alpha=1.0,
                )

                val_loss += loss.item() * input_ids.size(0)
                val_correct += is_correct.sum().item()
                val_correct_cells += preds.eq(target_ids).sum().item()
                blank_mask = input_ids.eq(0)
                val_correct_blank_cells += (
                    preds[blank_mask].eq(target_ids[blank_mask]).sum().item()
                )
                val_total_blank_cells += blank_mask.sum().item()
                val_samples += input_ids.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        val_cell_acc = val_correct_cells / (val_samples * l_ans)
        val_blank_acc = val_correct_blank_cells / val_total_blank_cells

        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Blank Acc: {train_blank_acc:.4f} | Train Cell Acc: {train_cell_acc:.4f} | "
            f"Train Exact: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Blank Acc: {val_blank_acc:.4f} | Val Cell Acc: {val_cell_acc:.4f} | "
            f"Val Exact: {val_acc:.4f}",
            flush=True,
        )

        # Save checkpoint at the end of each epoch to prevent data loss
        os.makedirs("data", exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "task": "sudoku",
                "state_dict": model.state_dict(),
                "config": vars(config),
                "task_config": {
                    "target_encoding": "autoregressive_digits_v2",
                    "clues": "immutable",
                    "loss": "blank_cells_only",
                },
                "seed": seed,
                "dataset": data_path,
                "validation_indices": validation_indices,
            },
            "data/sudoku_model.pt",
        )


if __name__ == "__main__":
    main()
