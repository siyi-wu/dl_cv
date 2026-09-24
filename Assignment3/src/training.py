from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


@torch.no_grad()
def evaluate_during_training(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    device,
    checkpoint_path: Path,
    early_stopping_patience: int | None = None,
):
    """使用统一训练条件训练一个模型，并保存最佳验证权重。"""
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = []
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        val_loss, val_accuracy = evaluate_during_training(
            model, val_loader, loss_function, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train {train_accuracy:.4f} | val {val_accuracy:.4f}"
        )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    return history
