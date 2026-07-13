import os
import json
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from scripts.train_multitask import (
    UnifiedReasoningLLM,
    tokenize_string,
)


def main() -> None:
    data_dir = "data"

    # Configuration for MultiTask overfit test
    config = MambaHybridConfig(
        d_model=64,
        n_meta=16,
        l_ans=128,
        n_steps=2,
        t_cycles=2,
        vocab_size=128,
    )

    # Initialize custom small dataset with exactly 1 sample per task
    class SmallMultiTaskDataset(
        torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor, str]]
    ):
        def __init__(self) -> None:
            self.samples: List[Tuple[str, str, str]] = []

            # 1. Maze
            maze_path = os.path.join(data_dir, "maze_dryrun.pt")
            if os.path.exists(maze_path):
                maze_data = torch.load(maze_path)
                s = maze_data[0]
                grid_str = "".join(str(int(val)) for row in s["grid"] for val in row)
                self.samples.append(
                    (
                        f"MAZE: {grid_str}",
                        "PATH: " + " ".join(f"({r},{c})" for r, c in s["path"]),
                        "MAZE",
                    )
                )

            # 2. Sudoku
            sudoku_path = os.path.join(data_dir, "sudoku.jsonl")
            if os.path.exists(sudoku_path):
                with open(sudoku_path, "r") as f:
                    s = json.loads(f.readline())
                    puzzle_str = "".join(str(val) for row in s["puzzle"] for val in row)
                    sol_str = "".join(str(val) for row in s["solution"] for val in row)
                    self.samples.append(
                        (f"SUDOKU: {puzzle_str}", f"SOL: {sol_str}", "SUDOKU")
                    )

            # 3. Dijkstra
            dijkstra_path = os.path.join(data_dir, "dijkstra.pt")
            if os.path.exists(dijkstra_path):
                dijkstra_data = torch.load(dijkstra_path)
                s = dijkstra_data[0]
                adj_str = ",".join(
                    "".join(str(int(val)) for val in row) for row in s["adjacency"]
                )
                self.samples.append(
                    (
                        f"DIJKSTRA: {adj_str}",
                        "PARENTS: " + " ".join(str(p) for p in s["parents"]),
                        "DIJKSTRA",
                    )
                )

            # 4. GSM8K
            gsm8k_path = os.path.join(data_dir, "gsm8k_train.jsonl")
            if os.path.exists(gsm8k_path):
                with open(gsm8k_path, "r") as f:
                    s = json.loads(f.readline())
                    self.samples.append(
                        (f"GSM8K: {s['question']}", f"ANSWER: {s['answer']}", "GSM8K")
                    )

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
            inp, tgt, task = self.samples[idx]
            inp_tok = tokenize_string(inp, 128)
            tgt_tok = tokenize_string(tgt, 128)
            return (
                torch.tensor(inp_tok, dtype=torch.long),
                torch.tensor(tgt_tok, dtype=torch.long),
                task,
            )

    dataset = SmallMultiTaskDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multi-Task overfit test device: {device}")
    print(f"Loaded tasks: {[s[2] for s in dataset.samples]}")

    model = UnifiedReasoningLLM(config, vocab_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    print("Starting overfit training for 150 epochs...")
    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        for input_ids, target_ids, task_names in loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            logits, bce_probs = model(input_ids, task_names)
            preds = logits.argmax(dim=-1)

            # Accuracy on non-padding tokens (target not 0)
            correct_tokens += ((preds == target_ids) & (target_ids != 0)).sum().item()
            total_tokens += (target_ids != 0).sum().item()

            is_correct = (preds == target_ids).all(dim=-1)
            correct_mask = is_correct.float()

            loss = compute_bce_joint_loss(
                logits, target_ids, bce_probs, correct_mask, alpha=1.0, ignore_index=0
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(dataset)
        acc = (correct_tokens / max(1, total_tokens)) * 100
        if epoch % 25 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Non-Pad Token Acc: {acc:.2f}%"
            )

    # Evaluate final predictions
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            input_ids, target_ids, task = dataset[i]
            logits, _ = model(input_ids.unsqueeze(0).to(device), [task])
            pred = logits.argmax(dim=-1).squeeze(0).cpu()

            # Convert back to strings
            pred_str = "".join(chr(t) for t in pred.tolist() if t > 0)
            target_str = "".join(chr(t) for t in target_ids.tolist() if t > 0)

            print(f"\nTask: {task}")
            print("Predict:  ", pred_str[:120])
            print("Solution: ", target_str[:120])


if __name__ == "__main__":
    main()
