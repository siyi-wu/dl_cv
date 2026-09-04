"""Small MNIST loader that does not require torchvision."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(root: str | Path) -> Path:
    """Download and verify the four MNIST gzip archives."""

    raw_dir = Path(root) / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in MNIST_FILES.items():
        destination = raw_dir / filename
        if destination.exists() and _md5(destination) == expected_md5:
            continue
        destination.unlink(missing_ok=True)
        print(f"Downloading {filename} ...", flush=True)
        urllib.request.urlretrieve(MNIST_BASE_URL + filename, destination)
        if _md5(destination) != expected_md5:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {filename}")
    return raw_dir


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic in {path}: {magic}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"Truncated image IDX file {path}: {data.size} != {expected}")
    return data.reshape(count, rows, cols).copy()


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic in {path}: {magic}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"Truncated label IDX file {path}: {data.size} != {count}")
    return data.copy()


class MNISTDataset(Dataset[tuple[torch.Tensor, int]]):
    """MNIST images normalized from uint8 to [-1, 1]."""

    def __init__(self, root: str | Path, train: bool = True, download: bool = True):
        raw_dir = Path(root) / "MNIST" / "raw"
        if download:
            raw_dir = download_mnist(root)
        image_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
        label_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"
        if not (raw_dir / image_name).exists() or not (raw_dir / label_name).exists():
            raise FileNotFoundError("MNIST archives are missing; rerun with download=True")
        self.images = _read_images(raw_dir / image_name)
        self.labels = _read_labels(raw_dir / label_name)

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index]).float().unsqueeze(0) / 127.5 - 1.0
        return image, int(self.labels[index])
