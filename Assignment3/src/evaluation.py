from __future__ import annotations

import torch


@torch.no_grad()
def evaluate_classifier(model, data_loader, device, number_of_classes: int = 10):
    """计算准确率、混淆矩阵和逐类召回率，并保留典型错误所需数据。"""
    model.eval()
    image_parts = []
    label_parts = []
    prediction_parts = []
    confidence_parts = []

    for images, labels in data_loader:
        probabilities = model(images.to(device)).softmax(dim=1).cpu()
        confidences, predictions = probabilities.max(dim=1)

        image_parts.append(images)
        label_parts.append(labels)
        prediction_parts.append(predictions)
        confidence_parts.append(confidences)

    images = torch.cat(image_parts)
    labels = torch.cat(label_parts)
    predictions = torch.cat(prediction_parts)
    confidences = torch.cat(confidence_parts)

    positions = labels * number_of_classes + predictions
    matrix = torch.bincount(
        positions, minlength=number_of_classes**2
    ).reshape(number_of_classes, number_of_classes)
    recall = matrix.diag().float() / matrix.sum(dim=1)
    accuracy = (labels == predictions).float().mean().item()

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "recall": recall,
        "images": images,
        "labels": labels,
        "predictions": predictions,
        "confidences": confidences,
    }

