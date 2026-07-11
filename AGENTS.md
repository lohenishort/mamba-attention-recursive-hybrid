# Project Overview
The Mamba-Attention Recursive Reasoning Hybrid framework is a modular, high-performance PyTorch library designed to research, train, and evaluate hybrid sequence models that combine Structured State Duality (SSD/Mamba-2) and attention mechanisms. The core architecture implements a decoupled "thinking fast and slow" paradigm: a bidirectional attention-based latent planning loop recursively refines memory and task-constraint representations (meta-tokens) over variable compute steps, followed by an autoregressive Mamba-2 generator that decodes the final solution sequence. The framework integrates Adaptive Computation Time (ACT) halting policies (via Q-learning or BCE stopping probability heads), Probabilistic Tiny Recursive Model (PTRM) stochastic inference-time scaling (via Gaussian noise injection and Q-head selection), and sparse final-step supervision to prevent shortcut memorization on complex, deterministic reasoning tasks.

# Build & Test Commands
All development commands must be executed using Poetry.
- **Install Dependencies:** `poetry install --all-groups`
- **Run Unit & Integration Tests:** `poetry run pytest -v`
- **Run Type Checks:** `poetry run mypy . --strict`
- **Run Lint Checks:** `poetry run ruff check .`
- **Format Code:** `poetry run ruff format .`
- **Run Synthetic Maze-Hard Generator Check:** `poetry run python -m scripts.generate_data --task maze --size 30 --num-samples 10`
- **Run Single-Step Baseline Training:** `poetry run python -m train.run_experiment --config configs/baseline_sudoku.yaml --dry-run`

# Code Style & Conventions
- **ALWAYS** write the Mamba-2 / SSD scan operations in pure, readable PyTorch tensor operations as the default fallback to ensure CPU/GPU compatibility and easy local debugging.
- **ALWAYS** make the GPU-accelerated Triton/CUDA kernels from the official `mamba-ssm` library optional via a runtime check and configuration toggle (`use_cuda_kernels`).
- **ALWAYS** type-hint all function signatures, class initializers, and tensor shapes using PEP 484 type annotations and comments showing expected tensor dimensions (e.g., `# [batch_size, seq_len, d_model]`).
- **NEVER** apply loss to every intermediate recursion step in the latent planning loop; **ALWAYS** supervise sparsely at the final step of each segment to prevent the model from learning shortcut latent sequence replay behavior.
- **NEVER** use the 1-step Implicit Function Theorem (IFT) gradient approximation for backpropagation; **ALWAYS** perform full-recursion backpropagation through the computational graph of the supervision segments.
- **NEVER** detach the latent planning state's computational graph at the boundaries of each supervision segment to prevent gradient explosion and instability during long recursive unrolls.
- **NEVER** hardcode reward-shaping values or negative step penalties inside the Q-learning ACT halting head; **ALWAYS** use the mathematically verified binary reward scheme (1 for correct, 0 for incorrect) with explicit hyperparameter-driven compute bounds (`M_min`/`M_max`).

# File System Guardrails
- **NEVER** modify or delete files under `tests/conftest.py` or the test verification suites unless explicitly instructed.
- **NEVER** modify `poetry.lock` or add dependencies directly to `pyproject.toml` without verifying compatibility via `poetry check`.
- **NEVER** create or modify temporary scratch directories or raw dataset dumps outside of the designated `data/` and `tmp/` gitignored folders.
- **NEVER** commit model checkpoints, tensorboard logs, or raw generated dataset files (e.g., `.pt`, `.safetensors`, `.jsonl`, `.log`) to the repository.

# Git & PR Workflow
- **ALWAYS** run linting (`poetry run ruff check .`), formatting (`poetry run ruff format .`), and type-checking (`poetry run mypy . --strict`) locally before committing.
- **ALWAYS** write descriptive, structured commit messages prefixing changes with `feat:`, `fix:`, `docs:`, `refactor:`, or `test:`.
- **ALWAYS** ensure that any new architectural blocks or training/evaluation features are accompanied by corresponding unit tests in the `tests/` directory.
- **ALWAYS** check that the test suite passes completely (`poetry run pytest`) before creating a pull request.

# Known Gotchas & Troubleshooting
- **Mamba-2 Custom Kernels:** Compilation of `mamba-ssm` and `causal-conv1d` fails on machines without compatible CUDA toolkits or with mismatched PyTorch/CUDA versions. Always run with the configuration `use_cuda_kernels: false` when training or testing on CPU or generic CI/CD pipelines.
- **Memory Consumption in Full-Recursion BPTT:** Backpropagating through long recursion loops can cause Out-Of-Memory (OOM) errors on standard GPUs. If OOM occurs, decrease the supervision segment length or enable activation checkpointing (`torch.utils.checkpoint`) in the planning loop.
- **ACT Q-Learning Instability:** Training the ACT head via Q-learning can be unstable early in training. Ensure target network updates are configured properly (`tau` or update frequency) and that the experience replay buffer size is sufficiently large.
- **Noise Scale in PTRM:** The performance of the PTRM inference mode is highly sensitive to the scale of the injected Gaussian noise ($\sigma$). Always sweep $\sigma$ in a range of $[0.01, 0.1]$ on a validation subset before running full evaluation.
