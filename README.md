# Mamba-Attention Recursive Reasoning Hybrid Framework

A PyTorch research framework that separates recursive latent planning from causal answer printing.

The shared planner combines bidirectional attention, bidirectional planning-time SSD scans, answer memory, cycle-level ACT, and optional task-routed MoE layers. A prefix-causal Mamba-attention printer then generates answers from meta memory, answer memory, and raw input context.

Task-native pipelines are included for:

- Sudoku: immutable clues, blank-only loss, unique-solution data, autoregressive digits.
- Maze: directional moves with EOS and legal goal-reaching evaluation.
- Dijkstra: raw weighted adjacency, explicit source/unreachable classes, legal-parent constraints, tie-tolerant optimal-tree evaluation.
- GSM8K: UTF-8 questions and normalized final-integer generation without rationale imitation.
- Native multitask: homogeneous round-robin batches through one shared MoE planner and task-specific adapters/printers.

See `TRAINING.md` for commands and checkpoint semantics.

## Native acceleration

The optional Maturin/PyO3 extension accelerates CPU-side batched maze validation, exact-token PTRM consensus, path validation, and Sudoku validation. These helpers retain Python fallbacks when the extension is unavailable. Build the local release extension with:

```bash
uv run --with maturin maturin develop --release
```

Autograd and device-resident tensor loops remain in PyTorch. For supported GPUs, enable the optional official `mamba-ssm` Triton scan with `use_cuda_kernels=True`; moving those operations through PyO3 would force host transfers and break the intended tensor execution path.
