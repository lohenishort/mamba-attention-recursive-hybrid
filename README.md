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
