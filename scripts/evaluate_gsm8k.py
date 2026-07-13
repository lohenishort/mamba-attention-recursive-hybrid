"""Evaluate GSM8K on the official test split by normalized final answer."""

import torch
from torch.utils.data import DataLoader

from mamba_hybrid.tasks.gsm8k import decode_bytes
from scripts.train_gsm8k import (
    GSM8KDataset,
    GSM8KReasoningModel,
    collate_gsm8k,
)
from scripts.utils import config_from_dict, require_file


def main() -> None:
    checkpoint_path = "data/gsm8k_model.pt"
    require_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "gsm8k":
        raise ValueError("legacy GSM8K checkpoint is not compatible; retrain schema v2")
    test_path = str(checkpoint["official_test_dataset"])
    require_file(test_path)
    task_config = checkpoint["task_config"]
    dataset = GSM8KDataset(
        test_path, max_question_bytes=int(task_config["max_question_bytes"])
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_gsm8k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GSM8KReasoningModel(
        config_from_dict(checkpoint["config"]),
        max_question_bytes=int(task_config["max_question_bytes"]),
        max_answer_length=int(task_config["max_answer_length"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    parsed = 0
    correct = 0
    total = 0
    for questions, question_mask, _, targets in loader:
        generated = model.generate(questions.to(device), question_mask.to(device)).cpu()
        for prediction, target in zip(generated.tolist(), targets.tolist()):
            try:
                predicted_answer = decode_bytes(prediction)
                parsed += 1
            except (UnicodeDecodeError, ValueError):
                predicted_answer = ""
            expected_answer = decode_bytes([token for token in target if token != -100])
            correct += int(predicted_answer == expected_answer)
            total += 1
    print(f"Parse rate: {parsed / total:.4f}")
    print(f"Official normalized answer exact match: {correct / total:.4f}")


if __name__ == "__main__":
    main()
