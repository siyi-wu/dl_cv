from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_datasets


def get_labels(dataset) -> torch.Tensor:
    """取得 TensorDataset 或 Subset 中的全部标签。"""
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1]
        return labels[dataset.indices]
    return dataset.tensors[1]


def find_label_values(*datasets) -> list[int]:
    """从数据本身发现所有标签值，不预先假定类别数量。"""
    all_labels = torch.cat([get_labels(dataset) for dataset in datasets])
    return [int(value) for value in torch.unique(all_labels, sorted=True)]


def class_counts(dataset, label_values: list[int]) -> list[int]:
    labels = get_labels(dataset)
    return [int((labels == value).sum()) for value in label_values]


def save_class_distribution(
    train_dataset,
    val_dataset,
    test_dataset,
    label_values: list[int],
    class_names: list[str],
    output_dir: Path,
    dpi: int,
):
    train_counts = class_counts(train_dataset, label_values)
    val_counts = class_counts(val_dataset, label_values)
    test_counts = class_counts(test_dataset, label_values)

    positions = torch.arange(len(label_values)).numpy()
    tick_labels = [
        f"Label {value}: {class_names[value]}" for value in label_values
    ]
    width = 0.25
    figure, axis = plt.subplots(figsize=(11, 5), dpi=dpi)
    axis.bar(positions - width, train_counts, width, label="Train")
    axis.bar(positions, val_counts, width, label="Validation")
    axis.bar(positions + width, test_counts, width, label="Test")
    axis.set_title("Fashion-MNIST class distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")
    axis.set_xticks(positions, tick_labels, rotation=35, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png")
    plt.close(figure)

    return train_counts, val_counts, test_counts


def save_class_examples(
    train_dataset,
    label_values: list[int],
    class_names: list[str],
    output_dir: Path,
    dpi: int,
):
    labels = get_labels(train_dataset)
    columns = min(5, len(label_values))
    rows = math.ceil(len(label_values) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2 * columns, 2.8 * rows),
        dpi=dpi,
        squeeze=False,
    )

    for label_value, axis in zip(label_values, axes.flat):
        subset_position = (labels == label_value).nonzero(as_tuple=True)[0][0].item()
        image, _ = train_dataset[subset_position]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"Label {label_value}\n{class_names[label_value]}")
        axis.axis("off")

    for axis in axes.flat[len(label_values):]:
        axis.axis("off")

    figure.suptitle("One training example from each class")
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.88,
        wspace=0.12,
        hspace=0.45,
    )
    figure.savefig(output_dir / "class_examples.png")
    plt.close(figure)


def main():
    config_path = PROJECT_ROOT / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = PROJECT_ROOT / "config" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    visualization_config = config["visualization"]
    class_names = metadata["label_names"]

    data_dir = PROJECT_ROOT / data_config["data_dir"]
    output_dir = PROJECT_ROOT / visualization_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_datasets(
        data_dir,
        train_size=data_config["train_size"],
        seed=data_config["split_seed"],
    )
    label_values = find_label_values(train_dataset, val_dataset, test_dataset)
    train_counts, val_counts, test_counts = save_class_distribution(
        train_dataset,
        val_dataset,
        test_dataset,
        label_values,
        class_names,
        output_dir,
        visualization_config["figure_dpi"],
    )
    save_class_examples(
        train_dataset,
        label_values,
        class_names,
        output_dir,
        visualization_config["figure_dpi"],
    )

    all_images = train_dataset.dataset.tensors[0]
    summary = {
        "image_shape": list(all_images.shape[1:]),
        "pixel_range": [float(all_images.min()), float(all_images.max())],
        "pixel_mean": float(all_images.mean()),
        "pixel_std": float(all_images.std()),
        "split_seed": data_config["split_seed"],
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "test_size": len(test_dataset),
        "number_of_classes": len(label_values),
        "label_values": label_values,
        "train_class_counts": train_counts,
        "validation_class_counts": val_counts,
        "test_class_counts": test_counts,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
