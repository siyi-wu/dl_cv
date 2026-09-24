from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "linear": "Linear",
    "two_linear": "Two linear layers",
    "mlp_relu": "MLP + ReLU",
}


def save_training_curves(history, model_name: str, path: Path, dpi: int):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[1].plot(
        epochs, [row["train_accuracy"] for row in history], label="Train"
    )
    axes[1].plot(
        epochs, [row["val_accuracy"] for row in history], label="Validation"
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(DISPLAY_NAMES[model_name])
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_model_comparison(rows: list[dict], path: Path, dpi: int):
    names = [DISPLAY_NAMES[row["model"]] for row in rows]
    validation = [row["best_val_accuracy"] for row in rows]
    test = [row["test_accuracy"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    axis.bar([position - width / 2 for position in positions], validation, width, label="Validation")
    axis.bar([position + width / 2 for position in positions], test, width, label="Test")
    axis.set(
        title="Controlled model comparison",
        ylabel="Accuracy",
        xticks=positions,
        xticklabels=names,
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

