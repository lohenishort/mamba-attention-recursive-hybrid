import os
import torch
from typing import List, Tuple

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_multitask import UnifiedReasoningLLM, MultiTaskDataset


def decode_tokens(tokens: List[int]) -> str:
    """Converts ASCII token IDs back to a string, skipping padding NULL chars."""
    chars = []
    for t in tokens:
        if t == 0:
            break
        chars.append(chr(t))
    return "".join(chars)


def main() -> None:
    model_path = "data/unified_model.pt"
    data_dir = "data"

    if not os.path.exists(model_path):
        print("Error: Trained model not found. Run train_multitask first.")
        return

    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading Multi-Task Evaluation Environment on {device}...")

    # Load checkpoint and extract config / state_dict
    checkpoint = torch.load(model_path, map_location=device)
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and "config" in checkpoint
    ):
        config_dict = checkpoint["config"]
        import inspect
        sig = inspect.signature(MambaHybridConfig.__init__)
        valid_keys = {k for k in sig.parameters.keys() if k != "self"}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = MambaHybridConfig(**filtered_config)
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        d_model = state_dict["embed.weight"].shape[1]
        n_meta = state_dict["reasoning_encoder.M_meta"].shape[1]
        config = MambaHybridConfig(
            d_model=d_model, n_meta=n_meta, l_ans=128, n_steps=2, t_cycles=2
        )

    # Load dataset (returns inp_ids, tgt_ids, task_name)
    dataset = MultiTaskDataset(
        data_dir, max_seq_len=128, l_ans=config.l_ans, max_samples_per_task=10
    )

    # Load model
    model = UnifiedReasoningLLM(config, vocab_size=128).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Group samples by task type
    task_samples: dict[str, list[Tuple[torch.Tensor, torch.Tensor, str]]] = {
        "MAZE": [],
        "SUDOKU": [],
        "DIJKSTRA": [],
        "GSM8K": [],
    }

    for i in range(len(dataset)):
        inp_ids, tgt_ids, task_name = dataset[i]
        if task_name in task_samples:
            task_samples[task_name].append((inp_ids, tgt_ids, task_name))

    print("\n--- Evaluating Unified Multi-Task Model ---")

    for task_name, samples in task_samples.items():
        if not samples:
            print(f"\n[Task: {task_name}] No samples available.")
            continue

        print(f"\n[Task: {task_name}]")
        inp_ids, tgt_ids, task_name_str = samples[0]

        with torch.no_grad():
            inp_batch = inp_ids.unsqueeze(0).to(device)
            # Pass task names in list format to task-routing model
            logits, _ = model(inp_batch, [task_name_str])  # [1, l_ans, vocab_size]
            preds = logits.argmax(dim=-1).squeeze(0).tolist()  # [l_ans]

        inp_text = decode_tokens(inp_ids.tolist())
        tgt_text = decode_tokens(tgt_ids.tolist())
        pred_text = decode_tokens(preds)

        short_inp = inp_text if len(inp_text) < 70 else inp_text[:70] + "..."

        print(f"  Input:     {short_inp}")
        print(f"  Expected:  {tgt_text}")
        print(f"  Predicted: {pred_text}")


if __name__ == "__main__":
    main()
