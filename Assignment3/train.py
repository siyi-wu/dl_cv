from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent

from src.data import create_train_val_loaders
from src.models import create_linear_model


def evaluate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_curves(history, output_path, dpi):
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")

    axes[1].plot(
        epochs, [item["train_accuracy"] for item in history], label="Train"
    )
    axes[1].plot(
        epochs, [item["val_accuracy"] for item in history], label="Validation"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Linear classifier training")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main():
    config_path = PROJECT_ROOT / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    training_config = config["training"]

    torch.manual_seed(training_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config["seed"])

    device = torch.device(training_config["device"])
    data_dir = PROJECT_ROOT / data_config["data_dir"]
    output_dir = PROJECT_ROOT / training_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_train_val_loaders(
        data_dir=data_dir,
        batch_size=data_config["batch_size"],
        train_size=data_config["train_size"],
        seed=data_config["split_seed"],
        num_workers=data_config["num_workers"],
    )

    model = create_linear_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training_config["learning_rate"]
    )

    history = []
    best_val_accuracy = 0.0

    for epoch in range(1, training_config["epochs"] + 1):
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
        val_loss, val_accuracy = evaluate(model, val_loader, loss_function, device)

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
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(
            f"Epoch {epoch:02d}/{training_config['epochs']} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, accuracy {val_accuracy:.4f}"
        )

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_curves(
        history,
        output_dir / "training_curves.png",
        config["visualization"]["figure_dpi"],
    )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
