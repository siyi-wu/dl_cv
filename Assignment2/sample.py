from __future__ import annotations

import argparse
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

from miniddpm.diffusion import GaussianDiffusion
from miniddpm.model import build_model
from miniddpm.utils import atomic_json_dump, resolve_device, sample_statistics, save_image_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MNIST samples from a trained DDPM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "samples" / "grid.png")
    parser.add_argument("--sampler", choices=("ancestral", "ddim"), default="ancestral")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-weights", action="store_true", help="Use non-EMA weights")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device, raw_weights: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("model_name", "mini"), **checkpoint["model_config"]).to(device)
    weight_key = "model" if raw_weights or "ema_model" not in checkpoint else "ema_model"
    model.load_state_dict(checkpoint[weight_key])
    model.eval()
    diffusion = GaussianDiffusion(device=device, **checkpoint["diffusion_config"])
    return model, diffusion, weight_key, checkpoint


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.columns < 1:
        raise ValueError("num-samples and columns must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, diffusion, weight_key, checkpoint = load_model(args.checkpoint, device, args.raw_weights)
    progress = tqdm(total=args.steps, desc=f"{args.sampler} sampling", unit="step")
    start = time.perf_counter()
    images = diffusion.sample(
        model,
        (args.num_samples, 1, 28, 28),
        steps=args.steps,
        sampler=args.sampler,
        eta=args.eta,
        progress_callback=lambda completed, total: progress.update(completed - progress.n),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    progress.close()
    save_image_grid(images, args.output, columns=args.columns)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_training_steps": checkpoint.get("global_step"),
        "weights": weight_key,
        "sampler": args.sampler,
        "steps": args.steps,
        "eta": args.eta,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        **sample_statistics(images),
    }
    atomic_json_dump(metadata, args.output.with_suffix(".json"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
