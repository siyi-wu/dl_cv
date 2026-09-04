"""Classifier-based evaluation utilities for MNIST generation."""

from __future__ import annotations

import math

import torch
from torch import nn


class MNISTClassifier(nn.Module):
    """Small CNN exposing a 128-dimensional feature embedding for MNIST FID."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_projection(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


def classification_statistics(
    logits: torch.Tensor,
    confidence_threshold: float = 0.8,
) -> dict[str, float | int | list[int] | list[float]]:
    """Summarize recognizability and class coverage from classifier logits."""

    if logits.ndim != 2 or logits.shape[1] != 10 or logits.shape[0] < 1:
        raise ValueError("Expected logits shaped [N, 10] with N >= 1")
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be in (0, 1)")
    probabilities = logits.float().softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    counts = torch.bincount(predictions, minlength=10)
    proportions = counts.float() / logits.shape[0]
    nonzero = proportions > 0
    normalized_entropy = -(proportions[nonzero] * proportions[nonzero].log()).sum() / math.log(10)
    return {
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(confidence.median()),
        "low_confidence_ratio": float((confidence < confidence_threshold).float().mean()),
        "class_coverage": int((counts > 0).sum()),
        "normalized_class_entropy": float(normalized_entropy),
        "class_counts": [int(value) for value in counts.tolist()],
        "class_proportions": [float(value) for value in proportions.tolist()],
    }


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / (features.shape[0] - 1)


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def frechet_feature_distance(real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    """Compute Fréchet distance between two feature sets using a stable PSD square root."""

    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("Feature tensors must be two-dimensional")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("Feature dimensions must match")
    if real_features.shape[0] < 2 or generated_features.shape[0] < 2:
        raise ValueError("At least two samples are required for each feature set")
    real = real_features.detach().double().cpu()
    generated = generated_features.detach().double().cpu()
    real_mean = real.mean(dim=0)
    generated_mean = generated.mean(dim=0)
    real_covariance = _covariance(real)
    generated_covariance = _covariance(generated)
    identity = torch.eye(real.shape[1], dtype=torch.float64)
    real_covariance = real_covariance + identity * 1e-6
    generated_covariance = generated_covariance + identity * 1e-6
    real_sqrt = _symmetric_matrix_sqrt(real_covariance)
    covariance_middle = real_sqrt @ generated_covariance @ real_sqrt
    covariance_mean = _symmetric_matrix_sqrt(covariance_middle)
    mean_term = (real_mean - generated_mean).square().sum()
    covariance_term = torch.trace(real_covariance + generated_covariance - 2 * covariance_mean)
    return float((mean_term + covariance_term).clamp_min(0))


def class_distribution_tvd(first: list[float], second: list[float]) -> float:
    """Total variation distance between two class-probability vectors."""

    if len(first) != 10 or len(second) != 10:
        raise ValueError("Class distributions must each contain 10 values")
    first_tensor = torch.tensor(first, dtype=torch.float64)
    second_tensor = torch.tensor(second, dtype=torch.float64)
    return float(0.5 * (first_tensor - second_tensor).abs().sum())
