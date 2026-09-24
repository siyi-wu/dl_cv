from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import load_train_val_datasets
from src.numpy_mlp import gradient_check, initialize_parameters


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads(
        (PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8")
    )
    data_config = config["data"]
    check_config = config["gradient_check"]

    train_dataset, _ = load_train_val_datasets(
        PROJECT_ROOT / data_config["data_dir"],
        train_size=data_config["train_size"],
        seed=data_config["split_seed"],
    )

    batch = [train_dataset[index] for index in range(check_config["batch_size"])]
    images = np.stack([image.numpy().reshape(-1) for image, _ in batch])
    labels = np.array([int(label) for _, label in batch])
    all_training_labels = train_dataset.dataset.tensors[1].numpy()
    number_of_classes = len(np.unique(all_training_labels))

    parameters = initialize_parameters(
        input_size=images.shape[1],
        hidden_size=check_config["hidden_size"],
        number_of_classes=number_of_classes,
        seed=check_config["seed"],
    )
    result = gradient_check(
        images,
        labels,
        parameters,
        epsilon=check_config["epsilon"],
        checks_per_parameter=check_config["checks_per_parameter"],
        seed=check_config["seed"],
    )

    output_path = PROJECT_ROOT / check_config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
