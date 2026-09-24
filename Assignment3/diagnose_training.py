from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.data import create_train_val_loaders
from src.models import create_model
from src.training import train_model
from visualization.diagnosis_plots import save_early_stopping_comparison


PROJECT_ROOT = Path(__file__).resolve().parent


def run_experiment(config, patience, experiment_dir):
    data_config = config["data"]
    training_config = config["training"]
    comparison_config = config["comparison"]
    diagnosis_config = config["diagnosis"]
    device = torch.device(training_config["device"])

    torch.manual_seed(training_config["seed"])
    torch.cuda.manual_seed_all(training_config["seed"])
    train_loader, val_loader = create_train_val_loaders(
        data_dir=PROJECT_ROOT / data_config["data_dir"],
        batch_size=data_config["batch_size"],
        train_size=data_config["train_size"],
        seed=data_config["split_seed"],
        num_workers=data_config["num_workers"],
    )
    model = create_model(
        diagnosis_config["model"], comparison_config["hidden_size"]
    ).to(device)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        device=device,
        checkpoint_path=experiment_dir / "best_model.pt",
        early_stopping_patience=patience,
    )
    elapsed_seconds = time.perf_counter() - start_time
    (experiment_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return history, elapsed_seconds


def main():
    config = json.loads(
        (PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8")
    )
    diagnosis_config = config["diagnosis"]
    output_dir = PROJECT_ROOT / diagnosis_config["output_dir"]

    print("\nControl: no early stopping")
    control_history, control_time = run_experiment(
        config, None, output_dir / "control"
    )

    print("\nTreatment: early stopping")
    stopping_history, stopping_time = run_experiment(
        config,
        diagnosis_config["early_stopping_patience"],
        output_dir / "early_stopping",
    )

    best_row = max(control_history, key=lambda row: row["val_accuracy"])
    summary = {
        "problem": "Training accuracy improves while validation performance degrades late in training",
        "hypothesis": "The MLP begins to overfit after its best validation epoch",
        "changed_factor": "Early stopping",
        "patience": diagnosis_config["early_stopping_patience"],
        "control": {
            "epochs_completed": len(control_history),
            "best_epoch": best_row["epoch"],
            "best_val_accuracy": best_row["val_accuracy"],
            "final_val_accuracy": control_history[-1]["val_accuracy"],
            "elapsed_seconds": control_time,
        },
        "early_stopping": {
            "epochs_completed": len(stopping_history),
            "best_epoch": max(
                stopping_history, key=lambda row: row["val_accuracy"]
            )["epoch"],
            "best_val_accuracy": max(
                row["val_accuracy"] for row in stopping_history
            ),
            "final_val_accuracy": stopping_history[-1]["val_accuracy"],
            "elapsed_seconds": stopping_time,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_early_stopping_comparison(
        control_history,
        stopping_history,
        best_epoch=best_row["epoch"],
        path=output_dir / "early_stopping_comparison.png",
        dpi=config["visualization"]["figure_dpi"],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

