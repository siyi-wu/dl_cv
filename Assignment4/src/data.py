from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def read_images(path: Path) -> torch.Tensor:
    """读取 gzip 压缩的 IDX 图像，返回 [N, 1, 28, 28]、范围 [0, 1] 的张量。"""
    with gzip.open(path, "rb") as file:
        _, count, rows, columns = struct.unpack(">IIII", file.read(16))
        images = np.frombuffer(file.read(), dtype=np.uint8).copy()

    images = torch.from_numpy(images.reshape(count, rows, columns)).float() / 255.0
    return images.unsqueeze(1)


def read_labels(path: Path) -> torch.Tensor:
    """读取 gzip 压缩的 IDX 标签。"""
    with gzip.open(path, "rb") as file:
        _, count = struct.unpack(">II", file.read(8))
        labels = np.frombuffer(file.read(), dtype=np.uint8).copy()

    return torch.from_numpy(labels.reshape(count)).long()


def load_train_val_datasets(
    data_dir: Path | str,
    train_size: int = 50_000,
    seed: int = 42,
):
    """读取官方训练集，并固定划分训练集和验证集。"""
    data_dir = Path(data_dir)

    train_images = read_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_labels(data_dir / "train-labels-idx1-ubyte.gz")

    full_train_dataset = TensorDataset(train_images, train_labels)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train_dataset), generator=generator)
    train_dataset = Subset(full_train_dataset, indices[:train_size].tolist())
    val_dataset = Subset(full_train_dataset, indices[train_size:].tolist())

    return train_dataset, val_dataset


def load_test_dataset(data_dir: Path | str):
    """单独读取官方测试集。"""
    data_dir = Path(data_dir)
    test_images = read_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    return TensorDataset(test_images, test_labels)


def load_datasets(
    data_dir: Path | str,
    train_size: int = 50_000,
    seed: int = 42,
):
    """读取训练集、验证集和测试集，供数据分析使用。"""
    train_dataset, val_dataset = load_train_val_datasets(data_dir, train_size, seed)
    test_dataset = load_test_dataset(data_dir)
    return train_dataset, val_dataset, test_dataset


def create_train_val_loaders(
    data_dir: Path | str,
    batch_size: int = 128,
    train_size: int = 50_000,
    seed: int = 42,
    num_workers: int = 0,
):
    """创建训练和验证 DataLoader，不读取测试集。"""
    train_dataset, val_dataset = load_train_val_datasets(data_dir, train_size, seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def create_dataloaders(
    data_dir: Path | str,
    batch_size: int = 128,
    train_size: int = 50_000,
    seed: int = 42,
    num_workers: int = 0,
):
    """创建训练、验证和测试 DataLoader。"""
    train_dataset, val_dataset, test_dataset = load_datasets(
        data_dir, train_size, seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def create_test_loader(
    data_dir: Path | str,
    batch_size: int = 256,
    num_workers: int = 0,
):
    """创建最终评价使用的测试 DataLoader。"""
    test_dataset = load_test_dataset(data_dir)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
