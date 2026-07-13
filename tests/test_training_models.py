import torch

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_dijkstra import DijkstraReasoningModel
from scripts.train_maze import MazeReasoningModel
from scripts.train_multitask import UnifiedReasoningLLM
from scripts.train_sudoku import SudokuReasoningModel


def test_task_wrappers_return_task_vocabulary_logits() -> None:
    sudoku_config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=81, n_steps=1, t_cycles=1, vocab_size=10
    )
    sudoku_logits, _ = SudokuReasoningModel(sudoku_config)(
        torch.zeros(1, 81, dtype=torch.long)
    )
    assert sudoku_logits.shape == (1, 81, 10)

    dijkstra_config = MambaHybridConfig(
        d_model=24, n_meta=2, l_ans=20, n_steps=1, t_cycles=1, vocab_size=20
    )
    dijkstra_logits, _ = DijkstraReasoningModel(dijkstra_config)(torch.zeros(1, 20, 24))
    assert dijkstra_logits.shape == (1, 20, 20)

    maze_config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=4, n_steps=1, t_cycles=1, vocab_size=10
    )
    maze_logits, _ = MazeReasoningModel(maze_config, grid_size=3)(
        torch.zeros(1, 9, dtype=torch.long)
    )
    assert maze_logits.shape == (1, 4, 10)

    multitask_config = MambaHybridConfig(
        d_model=8,
        n_meta=2,
        l_ans=4,
        n_steps=1,
        t_cycles=1,
        vocab_size=128,
        use_moe=True,
    )
    multitask_logits, _ = UnifiedReasoningLLM(multitask_config)(
        torch.zeros(1, 4, dtype=torch.long), ["SUDOKU"]
    )
    assert multitask_logits.shape == (1, 4, 128)
