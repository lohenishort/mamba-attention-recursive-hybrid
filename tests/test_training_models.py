import torch

from mamba_hybrid.config import MambaHybridConfig
from scripts.train_dijkstra import DijkstraReasoningModel
from scripts.train_gsm8k import GSM8KReasoningModel
from scripts.train_maze import MazeReasoningModel
from scripts.train_multitask import NativeMultiTaskModel
from scripts.train_sudoku import SudokuReasoningModel, sudoku_completion_targets


def test_task_wrappers_return_task_vocabulary_logits() -> None:
    sudoku_config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=81, n_steps=1, t_cycles=1, vocab_size=10
    )
    sudoku_model = SudokuReasoningModel(sudoku_config)
    sudoku_decoder = torch.full((1, 81), sudoku_model.pad_token, dtype=torch.long)
    sudoku_decoder[:, 0] = sudoku_model.bos_token
    sudoku_logits, _ = sudoku_model(
        torch.zeros(1, 81, dtype=torch.long), sudoku_decoder
    )
    assert sudoku_logits.shape == (1, 81, 10)

    dijkstra_config = MambaHybridConfig(
        d_model=24, n_meta=2, l_ans=20, n_steps=1, t_cycles=1, vocab_size=21
    )
    dijkstra_model = DijkstraReasoningModel(dijkstra_config)
    dijkstra_decoder = torch.zeros(1, 20, dtype=torch.long)
    dijkstra_decoder[:, 0] = dijkstra_model.bos_token
    dijkstra_logits, _ = dijkstra_model(
        torch.zeros(1, 20, 20),
        torch.zeros(1, dtype=torch.long),
        dijkstra_decoder,
    )
    assert dijkstra_logits.shape == (1, 20, 21)

    maze_config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=4, n_steps=1, t_cycles=1, vocab_size=6
    )
    maze_model = MazeReasoningModel(maze_config, grid_size=3)
    maze_decoder = torch.zeros(1, 4, dtype=torch.long)
    maze_decoder[:, 0] = maze_model.bos_token
    maze_logits, _ = maze_model(torch.zeros(1, 9, dtype=torch.long), maze_decoder)
    assert maze_logits.shape == (1, 4, 6)

    multitask_config = MambaHybridConfig(
        d_model=8,
        n_meta=2,
        l_ans=81,
        n_steps=1,
        M_max=1,
        t_cycles=1,
        vocab_size=259,
        use_moe=True,
    )
    multitask_model = NativeMultiTaskModel(multitask_config, grid_size=3)
    multitask_decoder = torch.full(
        (1, 81), multitask_model.sudoku.pad_token, dtype=torch.long
    )
    multitask_decoder[:, 0] = multitask_model.sudoku.bos_token
    multitask_logits, _ = multitask_model.forward_task(
        "SUDOKU",
        {
            "input_ids": torch.zeros(1, 81, dtype=torch.long),
            "decoder_input_ids": multitask_decoder,
        },
    )
    assert multitask_logits.shape == (1, 81, 10)


def test_sudoku_model_preserves_given_clues() -> None:
    config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=81, n_steps=1, t_cycles=1, vocab_size=10
    )
    model = SudokuReasoningModel(config).eval()
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    puzzle[0, [0, 17, 40, 80]] = torch.tensor([8, 3, 5, 9])

    decoder = torch.full((1, 81), model.pad_token, dtype=torch.long)
    decoder[:, 0] = model.bos_token
    logits, _ = model(puzzle, decoder)
    predictions = logits.argmax(dim=-1)

    clue_mask = puzzle.ne(0)
    assert torch.equal(predictions[clue_mask], puzzle[clue_mask])


def test_sudoku_completion_targets_supervise_only_blanks() -> None:
    puzzle = torch.tensor([[8, 0, 3, 0]])
    solution = torch.tensor([[8, 7, 3, 2]])

    targets = sudoku_completion_targets(puzzle, solution)

    assert torch.equal(targets, torch.tensor([[-100, 7, -100, 2]]))


def test_task_cycle_logits_only_track_gradients_for_final_cycle() -> None:
    config = MambaHybridConfig(
        d_model=8,
        n_meta=2,
        l_ans=4,
        n_steps=1,
        M_min=1,
        M_max=2,
        vocab_size=259,
    )

    maze = MazeReasoningModel(config, grid_size=2)
    maze_logits, _ = maze.forward_cycle_logits(
        torch.zeros(1, 4, dtype=torch.long),
        torch.tensor([[maze.bos_token, 0, 0, 0]]),
    )

    dijkstra = DijkstraReasoningModel(config, num_nodes=4)
    dijkstra_logits, _ = dijkstra.forward_cycle_logits(
        torch.zeros(1, 4, 4),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([[dijkstra.bos_token, 0, 0, 0]]),
    )

    sudoku = SudokuReasoningModel(config)
    sudoku_logits, _ = sudoku.forward_cycle_logits(
        torch.zeros(1, 81, dtype=torch.long),
        torch.cat(
            [
                torch.tensor([[sudoku.bos_token]]),
                torch.full((1, 80), sudoku.pad_token, dtype=torch.long),
            ],
            dim=1,
        ),
    )

    gsm8k = GSM8KReasoningModel(config, max_question_bytes=8, max_answer_length=4)
    gsm8k_logits, _ = gsm8k.forward_cycle_logits(
        torch.tensor([[49, 258]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[256, 50]]),
    )

    for cycle_logits in [maze_logits, dijkstra_logits, sudoku_logits, gsm8k_logits]:
        assert len(cycle_logits) == 2
        assert cycle_logits[0].requires_grad is False
        assert cycle_logits[-1].requires_grad is True


def test_sudoku_cached_generation_preserves_position_dependent_clues() -> None:
    config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=81, n_steps=1, M_max=1, vocab_size=10
    )
    model = SudokuReasoningModel(config).eval()
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    puzzle[0, [0, 17, 40, 80]] = torch.tensor([8, 3, 5, 9])

    logits, _ = model(puzzle)

    assert logits.shape == (1, 81, 10)
    assert torch.equal(logits.argmax(dim=-1)[puzzle.ne(0)], puzzle[puzzle.ne(0)])


def test_dijkstra_cached_generation_preserves_per_node_parent_constraints() -> None:
    config = MambaHybridConfig(
        d_model=8, n_meta=2, l_ans=4, n_steps=1, M_max=1, vocab_size=5
    )
    model = DijkstraReasoningModel(config, num_nodes=4).eval()
    adjacency = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    generated, _ = model.generate(adjacency, torch.tensor([0]))

    assert generated.shape == (1, 4)
    assert generated[0, 0].item() == 0
    assert generated[0, 1].item() in {0, 2, 4}
    assert generated[0, 2].item() in {1, 4}
    assert generated[0, 3].item() == 4
