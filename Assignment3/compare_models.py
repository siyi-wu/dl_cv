from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.data import create_test_loader, create_train_val_loaders
from src.evaluation import evaluate_classifier
from src.models import create_model
from src.training import train_model
from visualization.comparison_plots import (
    DISPLAY_NAMES,
    save_model_comparison,
    save_training_curves,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads(
        (PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8")
    )
    data_config = config["data"]
    training_config = config["training"]
    comparison_config = config["comparison"]

    device = torch.device(training_config["device"])
    data_dir = PROJECT_ROOT / data_config["data_dir"]
    output_dir = PROJECT_ROOT / comparison_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = config["visualization"]["figure_dpi"]

    rows = []
    for model_name in comparison_config["models"]:
        print(f"\nTraining {DISPLAY_NAMES[model_name]}")
        torch.manual_seed(training_config["seed"])
        torch.cuda.manual_seed_all(training_config["seed"])

        train_loader, val_loader = create_train_val_loaders(
            data_dir=data_dir,
            batch_size=data_config["batch_size"],
            train_size=data_config["train_size"],
            seed=data_config["split_seed"],
            num_workers=data_config["num_workers"],
        )
        model = create_model(
            model_name, hidden_size=comparison_config["hidden_size"]
        ).to(device)
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            device=device,
            checkpoint_path=model_dir / "best_model.pt",
        )
        (model_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_training_curves(
            history, model_name, model_dir / "training_curves.png", dpi
        )

        best_epoch = max(history, key=lambda row: row["val_accuracy"])
        rows.append(
            {
                "model": model_name,
                "hidden_size": (
                    0 if model_name == "linear" else comparison_config["hidden_size"]
                ),
                "activation": "ReLU" if model_name == "mlp_relu" else "None",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "best_epoch": best_epoch["epoch"],
                "best_val_accuracy": best_epoch["val_accuracy"],
            }
        )

    test_loader = create_test_loader(
        data_dir,
        batch_size=config["evaluation"]["batch_size"],
        num_workers=data_config["num_workers"],
    )
    for row in rows:
        model = create_model(
            row["model"], hidden_size=comparison_config["hidden_size"]
        ).to(device)
        model.load_state_dict(
            torch.load(
                output_dir / row["model"] / "best_model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        row["test_accuracy"] = evaluate_classifier(
            model, test_loader, device
        )["accuracy"]

    (output_dir / "model_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    save_model_comparison(rows, output_dir / "model_comparison.png", dpi)
    print("\nModel comparison")
    for row in rows:
        print(
            f"{DISPLAY_NAMES[row['model']]}: "
            f"val={row['best_val_accuracy']:.4f}, test={row['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

