"""Memory-conscious entry point for the native 30x30 maze trainer."""

from scripts.train_maze import main as train_maze


def main() -> None:
    train_maze(
        data_path="data/maze_hard.pt",
        checkpoint_path="data/maze_model.pt",
        d_model=128,
        n_meta=32,
        n_steps=4,
        max_cycles=3,
        batch_size=4,
        epochs=30,
    )


if __name__ == "__main__":
    main()
