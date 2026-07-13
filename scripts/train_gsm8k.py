"""Train GSM8K as normalized final-answer generation."""

import json
import os
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.loss import compute_bce_joint_loss
from mamba_hybrid.model import MambaAttentionHybrid
from mamba_hybrid.printer import AutoregressivePrinter
from mamba_hybrid.tasks.gsm8k import (
    BOS,
    EOS,
    PAD,
    VOCAB_SIZE,
    allowed_answer_tokens,
    encode_answer,
    encode_bytes,
    extract_answer,
)
from scripts.utils import deterministic_split_indices, seed_everything

GSM8KItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GSM8KBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class GSM8KDataset(Dataset[GSM8KItem]):
    """Load questions and normalized final answers without rationale supervision."""

    def __init__(self, path: str, *, max_question_bytes: int = 1024) -> None:
        self.records: list[dict[str, Any]] = []
        with open(path) as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                record = json.loads(line)
                question = encode_bytes(record["question"])
                if len(question) > max_question_bytes:
                    raise ValueError(
                        f"{path}:{line_number} question exceeds {max_question_bytes} bytes"
                    )
                answer = extract_answer(record["answer"])
                decoder_input, target = encode_answer(answer)
                self.records.append(
                    {
                        "question": question,
                        "decoder_input": decoder_input,
                        "target": target,
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GSM8KItem:
        record = self.records[index]
        return (
            torch.tensor(record["question"], dtype=torch.long),
            torch.tensor(record["decoder_input"], dtype=torch.long),
            torch.tensor(record["target"], dtype=torch.long),
        )


def collate_gsm8k(items: list[GSM8KItem]) -> GSM8KBatch:
    """Dynamically pad questions and answers only to this batch's maxima."""
    if not items:
        raise ValueError("GSM8K batch must not be empty")
    question_length = max(item[0].numel() for item in items)
    answer_length = max(item[1].numel() for item in items)
    questions = torch.full((len(items), question_length), PAD, dtype=torch.long)
    question_mask = torch.zeros_like(questions, dtype=torch.bool)
    decoder_inputs = torch.full((len(items), answer_length), PAD, dtype=torch.long)
    targets = torch.full((len(items), answer_length), -100, dtype=torch.long)
    for index, (question, decoder_input, target) in enumerate(items):
        questions[index, : question.numel()] = question
        question_mask[index, : question.numel()] = True
        decoder_inputs[index, : decoder_input.numel()] = decoder_input
        targets[index, : target.numel()] = target
    return questions, question_mask, decoder_inputs, targets


class GSM8KReasoningModel(nn.Module):
    """Plan over a question, then print its normalized integer answer causally."""

    def __init__(
        self,
        config: MambaHybridConfig,
        *,
        max_question_bytes: int = 1024,
        max_answer_length: int = 16,
        reasoning_encoder: MambaAttentionHybrid | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_question_bytes = max_question_bytes
        self.max_answer_length = max_answer_length
        self.question_embed = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD)
        self.question_positions = nn.Parameter(
            torch.randn(1, max_question_bytes, config.d_model) * 0.02
        )
        if reasoning_encoder is None:
            self.reasoning_encoder = MambaAttentionHybrid(config)
        else:
            object.__setattr__(self, "reasoning_encoder", reasoning_encoder)
        self.printer = AutoregressivePrinter(
            config,
            vocab_size=VOCAB_SIZE,
            max_length=max_answer_length,
            pad_token_id=PAD,
        )
        allowed = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        allowed[list(allowed_answer_tokens())] = True
        self.register_buffer("allowed_tokens", allowed)

    def encode_questions(
        self, question_ids: torch.Tensor, question_mask: torch.Tensor
    ) -> torch.Tensor:
        if question_ids.shape != question_mask.shape:
            raise ValueError("question mask must match question IDs")
        if question_ids.shape[1] > self.max_question_bytes:
            raise ValueError("question exceeds model capacity")
        embeddings = self.question_embed(question_ids)
        embeddings = embeddings + self.question_positions[:, : question_ids.shape[1]]
        return cast(
            torch.Tensor,
            embeddings * question_mask.to(embeddings.dtype).unsqueeze(-1),
        )

    def forward_cycle_logits(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x_raw = self.encode_questions(question_ids, question_mask)
        states, probabilities = self.reasoning_encoder.forward_state_trajectory(
            x_raw,
            task_names=["GSM8K"] * x_raw.shape[0],
            x_mask=question_mask,
        )
        cycle_logits: list[torch.Tensor] = []
        for state in states:
            prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
                x_raw, state, question_mask
            )
            cycle_logits.append(self.printer(prefix, decoder_input_ids, prefix_mask))
        return cycle_logits, probabilities

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cycle_logits, probabilities = self.forward_cycle_logits(
            question_ids, question_mask, decoder_input_ids
        )
        return cycle_logits[-1], probabilities

    @torch.no_grad()
    def generate(
        self, question_ids: torch.Tensor, question_mask: torch.Tensor
    ) -> torch.Tensor:
        x_raw = self.encode_questions(question_ids, question_mask)
        states, _ = self.reasoning_encoder.forward_state_trajectory(
            x_raw,
            task_names=["GSM8K"] * x_raw.shape[0],
            x_mask=question_mask,
        )
        prefix, prefix_mask = self.reasoning_encoder.build_memory_prefix(
            x_raw, states[-1], question_mask
        )
        return self.printer.generate(
            prefix,
            bos_token_id=BOS,
            eos_token_id=EOS,
            prefix_mask=prefix_mask,
            allowed_tokens=cast(torch.Tensor, self.allowed_tokens),
            max_new_tokens=self.max_answer_length,
        )


def _sequence_correct(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = targets.ne(-100)
    return (logits.argmax(dim=-1).eq(targets) | ~mask).all(dim=-1)


def main() -> None:
    train_path = "data/gsm8k_train.jsonl"
    test_path = "data/gsm8k_test.jsonl"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("GSM8K train/test files are required")
    device = torch.device(
        "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = MambaHybridConfig(
        d_model=128,
        n_meta=32,
        l_ans=1,
        n_steps=4,
        t_cycles=3,
        M_min=1,
        M_max=3,
        vocab_size=VOCAB_SIZE,
    )
    dataset = GSM8KDataset(train_path)
    seed = 42
    generator = seed_everything(seed)
    train_indices, validation_indices = deterministic_split_indices(len(dataset), seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=16,
        shuffle=True,
        generator=generator,
        collate_fn=collate_gsm8k,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(dataset, validation_indices),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_gsm8k,
    )
    model = GSM8KReasoningModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        correct_answers = 0
        samples_seen = 0
        for questions, question_mask, decoder_inputs, targets in train_loader:
            questions = questions.to(device)
            question_mask = question_mask.to(device)
            decoder_inputs = decoder_inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            cycle_logits, probabilities = model.forward_cycle_logits(
                questions, question_mask, decoder_inputs
            )
            cycle_correct = torch.stack(
                [_sequence_correct(logits, targets) for logits in cycle_logits]
            )
            loss = compute_bce_joint_loss(
                cycle_logits[-1],
                targets,
                probabilities,
                cycle_correct.float(),
                alpha=1.0,
                min_cycles=config.M_min,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * questions.shape[0]
            correct_answers += int(cycle_correct[-1].sum().item())
            samples_seen += questions.shape[0]

        model.eval()
        validation_correct = 0
        validation_samples = 0
        validation_loss = 0.0
        with torch.no_grad():
            for questions, question_mask, decoder_inputs, targets in validation_loader:
                questions = questions.to(device)
                question_mask = question_mask.to(device)
                decoder_inputs = decoder_inputs.to(device)
                targets = targets.to(device)
                logits, probabilities = model(questions, question_mask, decoder_inputs)
                correct = _sequence_correct(logits, targets)
                validation_loss += float(
                    F.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
                    ).item()
                    * questions.shape[0]
                )
                validation_correct += int(correct.sum().item())
                validation_samples += questions.shape[0]
        print(
            f"Epoch {epoch:02d}/20 | Train Loss: {total_loss / samples_seen:.4f} | "
            f"Train Exact: {correct_answers / samples_seen:.4f} | "
            f"Val Loss: {validation_loss / validation_samples:.4f} | "
            f"Val Exact: {validation_correct / validation_samples:.4f}",
            flush=True,
        )

    os.makedirs("data", exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "task": "gsm8k",
            "state_dict": model.state_dict(),
            "config": vars(config),
            "task_config": {
                "tokenization": "utf8_bytes_v1",
                "target": "normalized_final_integer",
                "max_question_bytes": model.max_question_bytes,
                "max_answer_length": model.max_answer_length,
            },
            "dataset": train_path,
            "official_test_dataset": test_path,
            "seed": seed,
            "validation_indices": validation_indices,
        },
        "data/gsm8k_model.pt",
    )


if __name__ == "__main__":
    main()
