# Training Guide

All commands run from the repository root through `uv`. Install dependencies with:

```bash
uv sync
```

Run only one GPU training process at a time on an 8 GB device.

## Architecture Semantics

- One ACT decision is made after each complete planning cycle.
- `M_min` and `M_max` count cycles, not latent micro-steps.
- The task loss is applied only to the final cycle.
- ACT targets are computed from each cycle's own decoded prediction.
- Full-recursion backpropagation remains connected to `M_meta`, answer initialization, and input adapters.
- Planning uses bidirectional attention and bidirectional SSD; printing remains causal.
- The printer prefix is `[meta memory, answer memory, raw context]`.
- Q-learning uses binary correctness rewards only. There is no handcrafted negative step reward.

## Data

Generate/download all datasets:

```bash
uv run python -m scripts.download_all_datasets
```

Sudoku generation guarantees one solution. Dijkstra schema v2 stores an explicit source and distinguishes unreachable vertices. Maze targets are moves plus EOS. GSM8K targets contain only the normalized integer after `####`.

## Standalone Training

```bash
uv run python -m scripts.train_sudoku
uv run python -m scripts.train_maze
uv run python -m scripts.train_maze_laptop
uv run python -m scripts.train_dijkstra
uv run python -m scripts.train_gsm8k
```

Task metrics:

- Sudoku: blank-cell accuracy, valid/exact board rate.
- Maze: legal goal-reaching solve rate and shortest-path solve rate.
- Dijkstra: legal-parent rate, optimal-parent rate, exact optimal-tree rate.
- GSM8K: normalized final-answer exact match on the official test split.

## Native Multitask Training

```bash
uv run python -m scripts.train_multitask
```

The multitask trainer does not serialize grids or graphs as ASCII. It schedules homogeneous task batches in deterministic round-robin order. Task adapters and autoregressive printers are separate; the recursive MoE planner is shared.

## Evaluation

```bash
uv run python -m scripts.evaluate_sudoku
uv run python -m scripts.evaluate_maze
uv run python -m scripts.evaluate_dijkstra
uv run python -m scripts.evaluate_gsm8k
uv run python -m scripts.evaluate_multitask
```

Schema-v2 evaluators reject semantically incompatible legacy checkpoints instead of loading mismatched heads with `strict=False`.

## Background Runs

Use the project virtual environment selected by `uv` and preserve the PID:

```bash
nohup uv run python -u -m scripts.train_sudoku > data/train_sudoku.log 2>&1 &
printf '%s\n' "$!" > data/train_sudoku.pid
```

Monitor and stop gracefully:

```bash
tail -f data/train_sudoku.log
kill "$(cat data/train_sudoku.pid)"
```

## Verification

```bash
uv run ruff format .
uv run ruff check .
uv run mypy . --strict
uv run pytest -v
```
