from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_confusion_matrix(
    matrix: torch.Tensor, class_names: list[str], path: Path, dpi: int
):
    values = matrix.numpy()
    figure, axis = plt.subplots(figsize=(8.5, 7), dpi=dpi)
    image = axis.imshow(values, cmap="Blues")
    axis.set(
        title="Linear classifier confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(matrix)),
        yticks=range(len(matrix)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")

    threshold = values.max() / 2
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[row, column] > threshold else "black",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_recall_chart(
    recall: torch.Tensor, class_names: list[str], path: Path, dpi: int
):
    labels = list(range(len(recall)))
    figure, axis = plt.subplots(figsize=(10, 4.5), dpi=dpi)
    axis.bar(labels, recall.numpy())
    axis.set(
        title="Recall for each class",
        xlabel="Class",
        ylabel="Recall",
        xticks=labels,
        xticklabels=class_names,
        ylim=(0, 1),
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_typical_errors(
    results, class_names: list[str], path: Path, count: int, dpi: int
):
    wrong = torch.where(results["labels"] != results["predictions"])[0]
    order = torch.argsort(results["confidences"][wrong], descending=True)
    selected = wrong[order[:count]]

    columns = 4
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.4 * columns, 2.7 * rows), dpi=dpi, squeeze=False
    )

    for index, axis in zip(selected, axes.flat):
        true_label = int(results["labels"][index])
        predicted_label = int(results["predictions"][index])
        confidence = float(results["confidences"][index])
        axis.imshow(results["images"][index].squeeze(0), cmap="gray")
        axis.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[predicted_label]} ({confidence:.2f})",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")

    figure.suptitle("Typical high-confidence errors")
    figure.subplots_adjust(top=0.92, hspace=0.45, wspace=0.15)
    figure.savefig(path)
    plt.close(figure)
