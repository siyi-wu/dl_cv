"""Shared utilities for reproducibility and artifact creation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def atomic_json_dump(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def tensor_to_uint8(images: torch.Tensor) -> np.ndarray:
    images = images.detach().float().cpu().clamp(-1, 1)
    images = ((images + 1) * 127.5).round().byte()
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale images shaped [N, 1, H, W]")
    return images[:, 0].numpy()


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    columns: int = 8,
    padding: int = 2,
    scale: int = 4,
) -> None:
    array = tensor_to_uint8(images)
    count, height, width = array.shape
    rows = (count + columns - 1) // columns
    canvas = Image.new(
        "L",
        (columns * width + (columns + 1) * padding, rows * height + (rows + 1) * padding),
        color=255,
    )
    for index, image in enumerate(array):
        row, column = divmod(index, columns)
        tile = Image.fromarray(image)
        canvas.paste(tile, (padding + column * (width + padding), padding + row * (height + padding)))
    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def sample_statistics(images: torch.Tensor) -> dict[str, float]:
    values = (images.detach().float().cpu().clamp(-1, 1) + 1) / 2
    horizontal = (values[:, :, :, 1:] - values[:, :, :, :-1]).abs().mean()
    vertical = (values[:, :, 1:, :] - values[:, :, :-1, :]).abs().mean()
    return {
        "sharpness": float((horizontal + vertical).item() / 2),
        "diversity": float(values.flatten(1).std(dim=0).mean().item()),
        "foreground_ratio": float((values > 0.5).float().mean().item()),
        "pixel_mean": float(values.mean().item()),
        "pixel_std": float(values.std().item()),
    }


def make_labeled_comparison(image_paths: list[Path], labels: list[str], output: Path) -> None:
    if len(image_paths) != len(labels) or not image_paths:
        raise ValueError("image_paths and labels must have equal non-zero length")
    images = [Image.open(path).convert("L") for path in image_paths]
    width = max(image.width for image in images)
    header = 34
    canvas = Image.new("L", (width * len(images), max(image.height for image in images) + header), 255)
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        draw.text((x + 8, 10), label, fill=0)
        canvas.paste(image, (x, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
