import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.utils import deterministic_split_indices, exact_match, seed_everything


def tokenize_string(
    s: str, max_len: int, *, allow_truncation: bool = False
) -> List[int]:
    """Converts a string to a list of ASCII token IDs, padded to max_len."""
    tokens = []
    for c in s:
        val = ord(c)
        if val > 127:
            val = 63  # ASCII for '?'
        tokens.append(val)
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    elif allow_truncation:
        tokens = tokens[:max_len]
    else:
        raise ValueError(
            f"Encoded text length {len(tokens)} exceeds configured limit {max_len}"
        )
    return tokens


# --- 1. Multi-Task Dataset Loader ---
class MultiTaskDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 128,
        l_ans: int = 128,
        max_samples_per_task: int = 100,
        required_tasks: Tuple[str, ...] = ("MAZE", "SUDOKU", "DIJKSTRA", "GSM8K"),
    ) -> None:
        self.samples: List[Tuple[str, str, str]] = []
        self.max_seq_len = max_seq_len
        self.l_ans = l_ans
        loaded_tasks: set[str] = set()

        def add_sample(inp: str, tgt: str, task: str) -> None:
            if len(inp) <= self.max_seq_len and len(tgt) <= self.l_ans:
                self.samples.append((inp, tgt, task))
                loaded_tasks.add(task)

        # 1. Maze
        maze_path = os.path.join(data_dir, "maze_dryrun.pt")
        if os.path.exists(maze_path):
            maze_data = torch.load(maze_path)[:max_samples_per_task]
            for s in maze_data:
                grid_str = "".join(str(int(val)) for row in s["grid"] for val in row)
                inp = f"MAZE: {grid_str}"
                tgt = "PATH: " + " ".join(f"({r},{c})" for r, c in s["path"])
                add_sample(inp, tgt, "MAZE")

        # 2. Sudoku
        sudoku_path = os.path.join(data_dir, "sudoku.jsonl")
        if os.path.exists(sudoku_path):
            count = 0
            with open(sudoku_path, "r") as f:
                for line in f:
                    s = json.loads(line)
                    grid_str = "".join(str(val) for row in s["puzzle"] for val in row)
                    inp = f"SUDOKU: {grid_str}"
                    sol_str = "".join(str(val) for row in s["solution"] for val in row)
                    tgt = f"SOL: {sol_str}"
                    add_sample(inp, tgt, "SUDOKU")
                    count += 1
                    if count >= max_samples_per_task:
                        break

        # 3. Dijkstra Graphs
        dijkstra_path = os.path.join(data_dir, "dijkstra.pt")
        if os.path.exists(dijkstra_path):
            dijkstra_data = torch.load(dijkstra_path)[:max_samples_per_task]
            for s in dijkstra_data:
                adj = s["adjacency"]
                edges = []
                for i in range(len(adj)):
                    for j in range(len(adj)):
                        if adj[i][j] > 0:
                            edges.append(f"{i}->{j}:{adj[i][j]:.1f}")
                inp = "DIJKSTRA: " + ",".join(edges)
                tgt = "DIST: " + " ".join(f"{d:.1f}" for d in s["distances"])
                add_sample(inp, tgt, "DIJKSTRA")

        # 4. GSM8K
        gsm8k_path = os.path.join(data_dir, "gsm8k_train.jsonl")
        if os.path.exists(gsm8k_path):
            count = 0
            with open(gsm8k_path, "r") as f:
                for line in f:
                    s = json.loads(line)
                    inp = f"GSM8K: {s['question']}"
                    tgt = f"ANS: {s['answer']}"
                    add_sample(inp, tgt, "GSM8K")
                    count += 1
                    if count >= max_samples_per_task:
                        break
        missing = set(required_tasks) - loaded_tasks
        if missing:
            raise FileNotFoundError(
                f"Missing required multitask datasets: {', '.join(sorted(missing))}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        inp_str, tgt_str, task_name = self.samples[idx]
        inp_tokens = tokenize_string(inp_str, self.max_seq_len)
        tgt_tokens = tokenize_string(tgt_str, self.l_ans)
        return (
            torch.tensor(inp_tokens, dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long),
            task_name,
        )


# --- 2. Unified Reasoning LLM Model with Task-Specific Heads ---
class UnifiedReasoningLLM(nn.Module):
    def __init__(
        self, config: MambaHybridConfig, vocab_size: int = 128, max_seq_len: int = 128
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        # Learnable 1D positional embeddings for input context
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, config.d_model))
        self.reasoning_encoder = MambaAttentionHybrid(config)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(config.d_model, vocab_size)
                for task in ("MAZE", "SUDOKU", "DIJKSTRA", "GSM8K")
            }
        )

        if config.vocab_size != vocab_size:
            raise ValueError("config.vocab_size must match the multitask vocabulary")

    def forward(
        self, input_ids: torch.Tensor, task_names: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # input_ids shape: [B, L_raw]
        X_raw = (
            self.embed(input_ids) + self.pos_embed[:, : input_ids.shape[1], :]
        )  # [B, L_raw, D]
        answer_states, probabilities = self.reasoning_encoder.forward_states(
            X_raw, task_names=task_names
        )
        logits = torch.stack(
            [
                self.heads[task](answer_states[index])
                for index, task in enumerate(task_names)
            ]
        )
        return logits, probabilities


# --- 3. Main Training Driver ---
def main() -> None:
    data_dir = "data"

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Set up config with max answer length (128 for GSM8K)
    l_ans = 128
    max_seq_len = 2048
    config = MambaHybridConfig(
        d_model=64,
        n_meta=16,
        l_ans=l_ans,
        n_steps=2,
        t_cycles=2,
        use_moe=True,
        vocab_size=128,
    )

    # Initialize Dataset (load 100 samples per task to train fast on CPU)
    dataset = MultiTaskDataset(
        data_dir,
        max_seq_len=max_seq_len,
        l_ans=l_ans,
        max_samples_per_task=100,
    )
    if len(dataset) == 0:
        print(
            "Error: No datasets found in data/ directory. Please run scripts.download_all_datasets first."
        )
        return

    print(f"Loaded total of {len(dataset)} multi-task samples.")

    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(len(dataset), seed)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, validation_indices)

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Initialize Unified Model
    model = UnifiedReasoningLLM(config, vocab_size=128, max_seq_len=max_seq_len).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    epochs = 5
    print("\nStarting Multi-Task Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_count = 0
        total_samples = 0

        for input_ids, target_ids, task_names in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(input_ids, task_names)

            # Accuracy on non-padding tokens
            preds = logits.argmax(dim=-1)
            is_correct = exact_match(preds, target_ids, 0)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0, ignore_index=0
            )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            correct_count += int(is_correct.sum().item())
            total_samples += input_ids.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_count / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for input_ids, target_ids, task_names in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, bce_probs = model(input_ids, task_names)
                preds = logits.argmax(dim=-1)
                is_correct = exact_match(preds, target_ids, 0)
                loss = compute_bce_joint_loss(
                    logits,
                    target_ids,
                    bce_probs,
                    is_correct.float(),
                    alpha=1.0,
                    ignore_index=0,
                )

                val_loss += loss.item() * input_ids.size(0)
                val_correct += int(is_correct.sum().item())
                val_samples += input_ids.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save checkpoint
    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(config),
            "seed": seed,
            "validation_indices": validation_indices,
            "max_seq_len": max_seq_len,
            "tasks": sorted(dataset.samples[index][2] for index in range(len(dataset))),
        },
        "data/unified_model.pt",
    )
    print(
        "\nSuccessfully saved trained Unified Multi-Task model to data/unified_model.pt"
    )


if __name__ == "__main__":
    main()
