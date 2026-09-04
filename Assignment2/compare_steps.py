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

import torch
from tqdm import tqdm

from miniddpm.utils import (
    atomic_json_dump,
    make_labeled_comparison,
    resolve_device,
    sample_statistics,
    save_image_grid,
    seed_everything,
)
from sample import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reverse diffusion step counts")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "comparison")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 200, 50])
    parser.add_argument("--sampler", choices=("ancestral", "ddim"), default="ancestral")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, diffusion, weight_key, checkpoint = load_model(args.checkpoint, device)
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("step counts must be unique")
    if any(step < 1 or step > diffusion.timesteps for step in args.steps):
        raise ValueError(f"all step counts must be in [1, {diffusion.timesteps}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn((args.num_samples, 1, 28, 28), generator=generator, device=device)
    _ = model(initial_noise[:1], torch.zeros(1, dtype=torch.long, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    records: list[dict] = []
    grids: list[Path] = []
    labels: list[str] = []
    for steps in args.steps:
        progress = tqdm(total=steps, desc=f"{steps} steps", unit="step")
        start = time.perf_counter()
        images = diffusion.sample(
            model,
            tuple(initial_noise.shape),
            steps=steps,
            sampler=args.sampler,
            eta=args.eta,
            initial_noise=initial_noise,
            progress_callback=lambda completed, total: progress.update(completed - progress.n),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        progress.close()
        grid_path = args.output_dir / f"grid_{steps}.png"
        save_image_grid(images, grid_path, columns=8)
        records.append(
            {
                "steps": steps,
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / steps,
                **sample_statistics(images),
            }
        )
        grids.append(grid_path)
        labels.append(f"{steps} steps | {elapsed:.2f} s")

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_training_steps": checkpoint.get("global_step"),
        "weights": weight_key,
        "sampler": args.sampler,
        "eta": args.eta,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": str(device),
        "shared_initial_noise": True,
        "results": records,
    }
    atomic_json_dump(payload, args.output_dir / "benchmark.json")
    with (args.output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    make_labeled_comparison(grids, labels, args.output_dir / "step_comparison.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
