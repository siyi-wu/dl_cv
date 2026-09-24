from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data import create_test_loader
from src.evaluation import evaluate_classifier
from src.models import create_linear_model
from visualization.evaluation_plots import (
    save_confusion_matrix,
    save_recall_chart,
    save_typical_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads(
        (PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (PROJECT_ROOT / "config" / "metadata.json").read_text(encoding="utf-8")
    )
    data_config = config["data"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    class_names = metadata["label_names"]

    device = torch.device(training_config["device"])
    data_dir = PROJECT_ROOT / data_config["data_dir"]
    output_dir = PROJECT_ROOT / training_config["output_dir"]

    test_loader = create_test_loader(
        data_dir,
        batch_size=evaluation_config["batch_size"],
        num_workers=data_config["num_workers"],
    )
    model = create_linear_model().to(device)
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )

    results = evaluate_classifier(model, test_loader, device)
    metrics = {
        "test_accuracy": results["accuracy"],
        "per_class_recall": {
            name: value
            for name, value in zip(class_names, results["recall"].tolist())
        },
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dpi = config["visualization"]["figure_dpi"]
    save_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        dpi,
    )
    save_recall_chart(
        results["recall"],
        class_names,
        output_dir / "per_class_recall.png",
        dpi,
    )
    save_typical_errors(
        results,
        class_names,
        output_dir / "typical_errors.png",
        evaluation_config["typical_error_count"],
        dpi,
    )

    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
