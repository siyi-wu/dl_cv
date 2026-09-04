from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miniddpm.utils import atomic_json_dump, make_labeled_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and enhanced DDPM results")
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--enhanced-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument("--enhanced-training", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "model_comparison")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def plot_quality(records: list[dict], output: Path) -> None:
    definitions = [
        ("mnist_feature_fid", "MNIST feature FID (lower is better)", 1.0),
        ("mean_confidence", "Mean confidence (%)", 100.0),
        ("low_confidence_ratio", "Low-confidence samples (%)", 100.0),
        ("class_distribution_tvd", "Class distribution TVD (lower is better)", 1.0),
    ]
    positions = list(range(len(records)))
    labels = [str(record["steps"]) for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=160)
    for axis, (key, title, multiplier) in zip(axes.flat, definitions):
        baseline = [record[f"baseline_{key}"] * multiplier for record in records]
        enhanced = [record[f"enhanced_{key}"] * multiplier for record in records]
        axis.bar([x - 0.19 for x in positions], baseline, 0.38, label="MiniUNet 1.16M")
        axis.bar([x + 0.19 for x in positions], enhanced, 0.38, label="EnhancedUNet 3.33M")
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Ancestral sampling steps")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Baseline vs paper-inspired EnhancedUNet", fontsize=15)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_epoch_losses(baseline_path: Path, enhanced_path: Path, output: Path) -> None:
    baseline = read_jsonl(baseline_path)
    enhanced = read_jsonl(enhanced_path)
    figure, axis = plt.subplots(figsize=(7.4, 4.4), dpi=160)
    axis.plot([row["epoch"] for row in baseline], [row["mean_loss"] for row in baseline], marker="o", label="MiniUNet 1.16M")
    axis.plot([row["epoch"] for row in enhanced], [row["mean_loss"] for row in enhanced], marker="o", label="EnhancedUNet 3.33M")
    axis.set(title="Epoch mean noise-prediction loss", xlabel="Epoch", ylabel="MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    baseline_evaluation = read_json(args.baseline_evaluation)
    enhanced_evaluation = read_json(args.enhanced_evaluation)
    baseline_training = read_json(args.baseline_training / "training_summary.json")
    enhanced_training = read_json(args.enhanced_training / "training_summary.json")
    baseline_by_step = {record["steps"]: record for record in baseline_evaluation["results"]}
    enhanced_by_step = {record["steps"]: record for record in enhanced_evaluation["results"]}
    if baseline_by_step.keys() != enhanced_by_step.keys():
        raise ValueError("Baseline and enhanced evaluations must contain identical step counts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    metric_names = [
        "mnist_feature_fid",
        "mean_confidence",
        "low_confidence_ratio",
        "class_coverage",
        "normalized_class_entropy",
        "class_distribution_tvd",
        "samples_per_second",
    ]
    for steps in baseline_by_step:
        baseline = baseline_by_step[steps]
        enhanced = enhanced_by_step[steps]
        record = {"steps": steps}
        for metric in metric_names:
            record[f"baseline_{metric}"] = baseline[metric]
            record[f"enhanced_{metric}"] = enhanced[metric]
        record["fid_reduction_percent"] = (
            (baseline["mnist_feature_fid"] - enhanced["mnist_feature_fid"])
            / baseline["mnist_feature_fid"]
            * 100
        )
        record["confidence_gain_percentage_points"] = (
            enhanced["mean_confidence"] - baseline["mean_confidence"]
        ) * 100
        record["low_confidence_reduction_percent"] = (
            (baseline["low_confidence_ratio"] - enhanced["low_confidence_ratio"])
            / baseline["low_confidence_ratio"]
            * 100
        )
        record["sampling_slowdown"] = (
            baseline["samples_per_second"] / enhanced["samples_per_second"]
        )
        records.append(record)

    payload = {
        "controlled_settings": {
            "training_samples": 60000,
            "epochs": 20,
            "optimization_steps": 9380,
            "schedule": "cosine",
            "diffusion_timesteps": 1000,
            "training_seed": 42,
            "evaluation_seed": baseline_evaluation["seed"],
            "evaluation_samples_per_setting": baseline_evaluation["num_samples"],
            "shared_evaluator_test_accuracy": baseline_evaluation["evaluator"]["test_accuracy"],
        },
        "baseline_training": baseline_training,
        "enhanced_training": enhanced_training,
        "results": records,
    }
    atomic_json_dump(payload, args.output_dir / "model_comparison.json")
    with (args.output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    plot_quality(records, args.output_dir / "model_quality_comparison.png")
    plot_epoch_losses(
        args.baseline_training / "metrics.jsonl",
        args.enhanced_training / "metrics.jsonl",
        args.output_dir / "epoch_loss_comparison.png",
    )
    for steps in baseline_by_step:
        make_labeled_comparison(
            [
                args.baseline_evaluation.parent / f"grid_{steps}.png",
                args.enhanced_evaluation.parent / f"grid_{steps}.png",
            ],
            [f"MiniUNet 1.16M | {steps} steps", f"EnhancedUNet 3.33M | {steps} steps"],
            args.output_dir / f"grid_comparison_{steps}.png",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
