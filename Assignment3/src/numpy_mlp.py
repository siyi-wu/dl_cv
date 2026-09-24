from __future__ import annotations

import numpy as np


def initialize_parameters(
    input_size: int,
    hidden_size: int,
    number_of_classes: int,
    seed: int,
):
    """初始化单隐藏层 MLP 的参数。"""
    random = np.random.default_rng(seed)
    return {
        "w1": random.normal(0, 0.01, size=(input_size, hidden_size)),
        "b1": np.full(hidden_size, 0.1),
        "w2": random.normal(0, 0.01, size=(hidden_size, number_of_classes)),
        "b2": np.zeros(number_of_classes),
    }


def forward(x: np.ndarray, parameters: dict[str, np.ndarray]):
    """使用 NumPy 完成前向传播。"""
    hidden_before_relu = x @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(hidden_before_relu, 0)
    logits = hidden @ parameters["w2"] + parameters["b2"]

    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)

    cache = (x, hidden_before_relu, hidden, probabilities)
    return probabilities, cache


def loss_and_backward(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
):
    """计算交叉熵损失，并用 NumPy 手动完成反向传播。"""
    probabilities, cache = forward(x, parameters)
    x, hidden_before_relu, hidden, probabilities = cache
    batch_size = len(labels)

    loss = -np.log(probabilities[np.arange(batch_size), labels] + 1e-12).mean()

    logits_gradient = probabilities.copy()
    logits_gradient[np.arange(batch_size), labels] -= 1
    logits_gradient /= batch_size

    gradients = {
        "w2": hidden.T @ logits_gradient,
        "b2": logits_gradient.sum(axis=0),
    }
    hidden_gradient = logits_gradient @ parameters["w2"].T
    hidden_gradient[hidden_before_relu <= 0] = 0
    gradients["w1"] = x.T @ hidden_gradient
    gradients["b1"] = hidden_gradient.sum(axis=0)

    return float(loss), gradients


def gradient_check(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    epsilon: float,
    checks_per_parameter: int,
    seed: int,
):
    """用中心差分抽查各参数的梯度。"""
    loss, analytical_gradients = loss_and_backward(x, labels, parameters)
    random = np.random.default_rng(seed)
    details = {}

    for name, parameter in parameters.items():
        flat_parameter = parameter.reshape(-1)
        flat_analytical = analytical_gradients[name].reshape(-1)
        check_count = min(checks_per_parameter, flat_parameter.size)
        indices = random.choice(flat_parameter.size, size=check_count, replace=False)
        relative_errors = []

        for index in indices:
            original_value = flat_parameter[index]

            flat_parameter[index] = original_value + epsilon
            loss_plus, _ = loss_and_backward(x, labels, parameters)

            flat_parameter[index] = original_value - epsilon
            loss_minus, _ = loss_and_backward(x, labels, parameters)

            flat_parameter[index] = original_value
            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            analytical_gradient = flat_analytical[index]
            denominator = max(
                1e-8, abs(numerical_gradient) + abs(analytical_gradient)
            )
            relative_errors.append(
                abs(numerical_gradient - analytical_gradient) / denominator
            )

        details[name] = {
            "checked_elements": check_count,
            "maximum_relative_error": float(max(relative_errors)),
            "mean_relative_error": float(np.mean(relative_errors)),
        }

    maximum_error = max(
        item["maximum_relative_error"] for item in details.values()
    )
    return {
        "loss": loss,
        "epsilon": epsilon,
        "maximum_relative_error": maximum_error,
        "passed": maximum_error < 1e-5,
        "parameters": details,
    }

