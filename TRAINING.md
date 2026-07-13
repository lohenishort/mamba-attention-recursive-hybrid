# Training Guide: Mamba-Attention Recursive reasoning Hybrid

This guide provides the instructions and commands to train and evaluate the four reasoning models (Sudoku, Maze, Dijkstra, and Unified Multitask) on the remote server or locally.

---

## 1. Setup & Environment

All commands should be executed from the root directory of the repository (`/srv/mamba-attention-recursive-hybrid` on the remote server). 

Install dependencies with `uv sync`. Run every Python command through `uv run`; do not invoke the project virtual environment directly.

---

## 2. Training the Models Separately

To prevent GPU Out-Of-Memory (OOM) errors on the single **NVIDIA RTX A1000 (8 GB)** GPU, it is recommended to run training jobs **individually** (one at a time) or configure batch sizes carefully if running concurrently.

### A. Sudoku Solver Model
Trains the model to solve 9x9 Sudoku boards using 2D row/column positional embeddings.
* **Command (Background):**
  ```bash
  nohup uv run python -u -m scripts.train_sudoku > /srv/sudoku_train.log 2>&1 &
  ```
* **Command (Foreground):**
  ```bash
  uv run python -u -m scripts.train_sudoku
  ```
* **Logs & Progress:** `/srv/sudoku_train.log`

### B. Maze Solver Model (30x30 Hard Maze)
Trains the model on a 30x30 grid (sequence length 900) using 2D positional embeddings.
To run this model on the RTX A1000 without OOM, the script uses a batch size of 4 and 8 gradient accumulation steps:
* **Command (Background):**
  ```bash
  nohup uv run python -u -m scripts.train_maze_laptop > /srv/maze_train.log 2>&1 &
  ```
* **Command (Foreground):**
  ```bash
  uv run python -u -m scripts.train_maze_laptop
  ```
* **Logs & Progress:** `/srv/maze_train.log`

### C. Dijkstra Graph Routing Model
Trains the model to find shortest path trees (SPT) on 20-node graphs using 1D positional embeddings.
* **Command (Background):**
  ```bash
  nohup uv run python -u -m scripts.train_dijkstra > /srv/dijkstra_train.log 2>&1 &
  ```
* **Command (Foreground):**
  ```bash
  uv run python -u -m scripts.train_dijkstra
  ```
* **Logs & Progress:** `/srv/dijkstra_train.log`

### D. Unified Multi-Task Model (MoE)
Trains a generalist sequence-to-sequence model handling Maze, Sudoku, Dijkstra, and GSM8K (math reasoning) using task-prefixed expert routing.
* **Run on GPU (Ensure other GPU training is stopped):**
  ```bash
  nohup uv run python -u -m scripts.train_multitask > /srv/multitask_train.log 2>&1 &
  ```
* **Run on CPU (Forces PyTorch to run on host CPU to prevent OOM):**
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  nohup uv run python -u -m scripts.train_multitask > /srv/multitask_train.log 2>&1 &
  ```
* **Logs & Progress:** `/srv/multitask_train.log`

---

## 3. Monitoring & Managing Background Jobs

* **Check running training processes:**
  ```bash
  ps aux | grep train_
  ```
* **Monitor progress in real-time:**
  ```bash
  tail -f /srv/sudoku_train.log
  # or dijkstra_train.log, maze_train.log, multitask_train.log
  ```
* **Stop a training process:**
  ```bash
  kill -9 <PID>
  # or kill all python training runs
  pkill -9 -f train_
  ```

---

## 4. Model Evaluation & Diagnostics

After training, you can evaluate model accuracy using the diagnostics and evaluation scripts:

* **Evaluate Sudoku Checkpoint:**
  ```bash
  uv run python -m scripts.evaluate_sudoku
  ```
* **Evaluate Dijkstra Checkpoint:**
  ```bash
  uv run python -m scripts.evaluate_dijkstra
  ```
* **Evaluate Maze Checkpoint:**
  ```bash
  uv run python -m scripts.evaluate_maze
  ```
* **Evaluate Multitask Unified Checkpoint:**
  ```bash
  uv run python -m scripts.evaluate_multitask
  ```
