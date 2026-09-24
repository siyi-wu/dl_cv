from torch import nn


def create_linear_model() -> nn.Module:
    """线性 Softmax 分类器；Softmax 由训练时的交叉熵损失完成。"""
    return create_model("linear")


def create_model(name: str, hidden_size: int = 128) -> nn.Module:
    """创建受控比较使用的三个模型。"""
    if name == "linear":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
        )
    if name == "two_linear":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.Linear(hidden_size, 10),
        )
    if name == "mlp_relu":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
    raise ValueError(f"Unknown model: {name}")
