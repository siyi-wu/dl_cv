from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from miniddpm.data import MNISTDataset
from miniddpm.evaluation import (
    MNISTClassifier,
    class_distribution_tvd,
    classification_statistics,
    frechet_feature_distance,
)
from miniddpm.utils import atomic_json_dump, resolve_device, save_image_grid, seed_everything
from sample import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifier-based evaluation for generated MNIST samples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--force-train-classifier", action="store_true")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 200, 50])
    parser.add_argument("--sampler", choices=("ancestral", "ddim"), default="ancestral")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_classifier(
    model: MNISTClassifier,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        progress = tqdm(loader, desc=f"classifier epoch {epoch}/{epochs}", unit="batch")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            count = labels.shape[0]
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += count
            progress.set_postfix(loss=f"{loss_sum / seen:.4f}", accuracy=f"{correct / seen:.4f}")
        history.append({"epoch": epoch, "loss": loss_sum / seen, "train_accuracy": correct / seen})
    return history


@torch.inference_mode()
def classify_dataset(model: MNISTClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    features = []
    logits = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_features = model.forward_features(images)
        features.append(batch_features.cpu())
        logits.append(model.classifier(batch_features).cpu())
        labels.append(torch.as_tensor(batch_labels).cpu())
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


@torch.inference_mode()
def classify_images(
    model: MNISTClassifier,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    features = []
    logits = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        batch_features = model.forward_features(batch)
        features.append(batch_features.cpu())
        logits.append(model.classifier(batch_features).cpu())
    return torch.cat(features), torch.cat(logits)


@torch.inference_mode()
def generate_samples(
    model: nn.Module,
    diffusion,
    initial_noise: torch.Tensor,
    steps: int,
    sampler: str,
    eta: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    generated = []
    progress = tqdm(total=initial_noise.shape[0], desc=f"evaluate {steps} steps", unit="image")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    for start in range(0, initial_noise.shape[0], batch_size):
        noise_batch = initial_noise[start : start + batch_size].to(device)
        batch = diffusion.sample(
            model,
            tuple(noise_batch.shape),
            steps=steps,
            sampler=sampler,
            eta=eta,
            initial_noise=noise_batch,
        )
        generated.append(batch.cpu())
        progress.update(batch.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    progress.close()
    return torch.cat(generated), elapsed


def plot_class_distributions(real_distribution: list[float], records: list[dict], output: Path) -> None:
    labels = list(range(10))
    width = 0.8 / (len(records) + 1)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar([x - 0.4 + width / 2 for x in labels], real_distribution, width, label="MNIST test")
    for index, record in enumerate(records, start=1):
        positions = [x - 0.4 + width / 2 + index * width for x in labels]
        axis.bar(positions, record["class_proportions"], width, label=f"{record['steps']} steps")
    axis.set_xticks(labels)
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("Proportion")
    axis.set_title("Generated class distribution")
    axis.set_ylim(0, max(0.2, axis.get_ylim()[1]))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.classifier_epochs < 1 or args.num_samples < 2 or args.sample_batch_size < 1:
        raise ValueError("classifier-epochs and sample-batch-size must be positive; num-samples must be >= 2")
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("step counts must be unique")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier_checkpoint = args.classifier_checkpoint or args.output_dir / "checkpoint_classifier.pt"

    train_dataset = MNISTDataset(args.data_dir, train=True, download=True)
    test_dataset = MNISTDataset(args.data_dir, train=False, download=True)
    classifier = MNISTClassifier().to(device)
    classifier_history: list[dict[str, float | int]] = []
    if classifier_checkpoint.exists() and not args.force_train_classifier:
        classifier_state = torch.load(classifier_checkpoint, map_location=device, weights_only=False)
        classifier.load_state_dict(classifier_state["model"])
        classifier_history = classifier_state.get("history", [])
        print(f"Loaded classifier from {classifier_checkpoint}")
    else:
        train_loader = make_loader(
            train_dataset, args.classifier_batch_size, True, args.seed, args.num_workers
        )
        classifier_history = train_classifier(
            classifier, train_loader, device, args.classifier_epochs
        )

    test_loader = make_loader(test_dataset, args.classifier_batch_size, False, args.seed, args.num_workers)
    _, test_logits, test_labels = classify_dataset(classifier, test_loader, device)
    test_accuracy = float((test_logits.argmax(dim=1) == test_labels).float().mean())
    torch.save(
        {
            "model": classifier.state_dict(),
            "feature_dim": classifier.feature_dim,
            "history": classifier_history,
            "test_accuracy": test_accuracy,
        },
        classifier_checkpoint,
    )

    reference_generator = torch.Generator().manual_seed(args.seed)
    reference_indices = torch.randperm(len(test_dataset), generator=reference_generator)[: args.num_samples]
    reference_dataset = Subset(test_dataset, reference_indices.tolist())
    reference_loader = make_loader(
        reference_dataset, args.classifier_batch_size, False, args.seed, args.num_workers
    )
    real_features, _, real_labels = classify_dataset(classifier, reference_loader, device)
    real_counts = torch.bincount(real_labels, minlength=10)
    real_distribution = (real_counts.float() / real_labels.shape[0]).tolist()

    diffusion_model, diffusion, weight_key, ddpm_checkpoint = load_model(args.checkpoint, device)
    if any(step < 1 or step > diffusion.timesteps for step in args.steps):
        raise ValueError(f"all step counts must be in [1, {diffusion.timesteps}]")
    initial_noise = torch.randn(
        (args.num_samples, 1, 28, 28), generator=torch.Generator().manual_seed(args.seed)
    )
    _ = diffusion_model(
        initial_noise[:1].to(device), torch.zeros(1, dtype=torch.long, device=device)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    records = []
    for steps in args.steps:
        seed_everything(args.seed)
        generated, elapsed = generate_samples(
            diffusion_model,
            diffusion,
            initial_noise,
            steps,
            args.sampler,
            args.eta,
            args.sample_batch_size,
            device,
        )
        generated_features, generated_logits = classify_images(
            classifier, generated, args.classifier_batch_size, device
        )
        statistics = classification_statistics(generated_logits, args.confidence_threshold)
        record = {
            "steps": steps,
            "elapsed_seconds": elapsed,
            "samples_per_second": args.num_samples / elapsed,
            "mnist_feature_fid": frechet_feature_distance(real_features, generated_features),
            **statistics,
        }
        record["class_distribution_tvd"] = class_distribution_tvd(
            record["class_proportions"], real_distribution
        )
        records.append(record)
        save_image_grid(
            generated[: min(64, generated.shape[0])],
            args.output_dir / f"grid_{steps}.png",
            columns=8,
        )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_training_steps": ddpm_checkpoint.get("global_step"),
        "weights": weight_key,
        "sampler": args.sampler,
        "eta": args.eta,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": str(device),
        "confidence_threshold": args.confidence_threshold,
        "evaluator": {
            "name": "MNISTClassifier",
            "feature_dim": classifier.feature_dim,
            "test_accuracy": test_accuracy,
            "training_history": classifier_history,
            "reference_samples": len(reference_dataset),
            "real_class_distribution": real_distribution,
        },
        "results": records,
    }
    atomic_json_dump(payload, args.output_dir / "evaluation.json")
    fieldnames = [
        "steps", "elapsed_seconds", "samples_per_second", "mnist_feature_fid",
        "mean_confidence", "median_confidence", "low_confidence_ratio",
        "class_coverage", "normalized_class_entropy", "class_distribution_tvd",
    ] + [f"class_{digit}_proportion" for digit in range(10)]
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key in record}
            row.update(
                {f"class_{digit}_proportion": record["class_proportions"][digit] for digit in range(10)}
            )
            writer.writerow(row)
    plot_class_distributions(real_distribution, records, args.output_dir / "class_distribution.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
