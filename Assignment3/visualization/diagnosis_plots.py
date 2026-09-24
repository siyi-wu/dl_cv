from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_early_stopping_comparison(
    control_history,
    early_stopping_history,
    best_epoch: int,
    path: Path,
    dpi: int,
):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=dpi)

    control_epochs = [row["epoch"] for row in control_history]
    stopping_epochs = [row["epoch"] for row in early_stopping_history]

    axes[0].plot(
        control_epochs,
        [row["train_accuracy"] for row in control_history],
        label="Train",
    )
    axes[0].plot(
        control_epochs,
        [row["val_accuracy"] for row in control_history],
        label="Validation",
    )
    axes[0].axvline(best_epoch, color="black", linestyle="--", label="Best epoch")
    axes[0].set(title="Control: no early stopping", xlabel="Epoch", ylabel="Accuracy")

    axes[1].plot(
        stopping_epochs,
        [row["val_accuracy"] for row in early_stopping_history],
        label="Validation",
        color="tab:orange",
    )
    axes[1].axvline(best_epoch, color="black", linestyle="--", label="Best epoch")
    axes[1].axvline(
        stopping_epochs[-1], color="tab:red", linestyle=":", label="Stop epoch"
    )
    axes[1].set(
        title="Treatment: patience = 3",
        xlabel="Epoch",
        ylabel="Validation accuracy",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Training diagnosis: early stopping")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)

