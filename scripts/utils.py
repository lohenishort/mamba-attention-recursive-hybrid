import inspect
import os
import random
from pathlib import Path
from typing import Any, Sequence

import torch

from mamba_hybrid.config import MambaHybridConfig


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required dataset/checkpoint not found: {path}")


def exact_match(
    predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int | None = None
) -> torch.Tensor:
    matches = predictions.eq(targets)
    if ignore_index is not None:
        matches = matches | targets.eq(ignore_index)
    return matches.all(dim=-1)


def config_from_dict(values: dict[str, Any]) -> MambaHybridConfig:
    valid = inspect.signature(MambaHybridConfig).parameters
    return MambaHybridConfig(
        **{key: value for key, value in values.items() if key in valid}
    )


def deterministic_split_indices(
    length: int, seed: int, validation_fraction: float = 0.2
) -> tuple[list[int], list[int]]:
    if length < 2:
        raise ValueError(
            "At least two samples are required for train/validation splits"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    validation_size = max(1, int(length * validation_fraction))
    return order[validation_size:], order[:validation_size]


def save_split(
    path: str, train_indices: Sequence[int], validation_indices: Sequence[int]
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_indices": list(train_indices),
            "validation_indices": list(validation_indices),
        },
        path,
    )


def load_validation_indices(checkpoint: dict[str, Any]) -> list[int]:
    indices = checkpoint.get("validation_indices")
    if not isinstance(indices, list) or not all(
        isinstance(index, int) for index in indices
    ):
        raise ValueError("Checkpoint does not contain persisted validation_indices")
    return indices
