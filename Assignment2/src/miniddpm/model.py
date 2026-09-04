"""A compact time-conditioned U-Net for 28x28 grayscale images."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(embedding, (0, self.dim % 2))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class AdaptiveResBlock(nn.Module):
    """AdaGN residual block following the guided-diffusion architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_scale_shift = nn.Linear(time_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_scale_shift(F.silu(time_embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + self.skip(x)


class SpatialAttention(nn.Module):
    """Residual multi-head self-attention used at the 7x7 bottleneck."""

    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        if heads < 1 or channels % heads != 0:
            raise ValueError("heads must be positive and divide channels")
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x).reshape(batch, channels, height * width)
        query, key, value = self.qkv(normalized).chunk(3, dim=1)
        positions = height * width
        query = query.reshape(batch, self.heads, self.head_dim, positions)
        key = key.reshape(batch, self.heads, self.head_dim, positions)
        value = value.reshape(batch, self.heads, self.head_dim, positions)
        weights = torch.matmul(query.transpose(-2, -1), key) * self.scale
        weights = weights.softmax(dim=-1)
        attended = torch.matmul(value, weights.transpose(-2, -1)).reshape(batch, channels, positions)
        return x + self.proj(attended).reshape(batch, channels, height, width)


class MiniUNet(nn.Module):
    """Noise-prediction U-Net. The default is well below 5M parameters."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down_block1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.down_block2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.middle1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.attention = SpatialAttention(base_channels * 4)
        self.middle2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up_block2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up_block1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input_conv(x)
        skip1 = self.down_block1(hidden, time_embedding)
        hidden = self.downsample1(skip1)
        skip2 = self.down_block2(hidden, time_embedding)
        hidden = self.downsample2(skip2)
        hidden = self.middle1(hidden, time_embedding)
        hidden = self.attention(hidden)
        hidden = self.middle2(hidden, time_embedding)
        hidden = self.upsample2(hidden)
        hidden = self.up_block2(torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = self.upsample1(hidden)
        hidden = self.up_block1(torch.cat((hidden, skip1), dim=1), time_embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class EnhancedUNet(nn.Module):
    """Paper-inspired AdaGN U-Net that remains below the assignment's 5M limit."""

    def __init__(self, in_channels: int = 1, base_channels: int = 48, time_dim: int = 192):
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down_block1 = AdaptiveResBlock(base_channels, base_channels, time_dim)
        self.down_refine1 = AdaptiveResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.down_block2 = AdaptiveResBlock(base_channels, base_channels * 2, time_dim)
        self.down_refine2 = AdaptiveResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.attention14 = SpatialAttention(base_channels * 2, heads=4)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)

        self.middle1 = AdaptiveResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.attention = SpatialAttention(base_channels * 4, heads=4)
        self.middle2 = AdaptiveResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 4, stride=2, padding=1
        )
        self.up_block2 = AdaptiveResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up_refine2 = AdaptiveResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.up_attention14 = SpatialAttention(base_channels * 2, heads=4)
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, stride=2, padding=1
        )
        self.up_block1 = AdaptiveResBlock(base_channels * 2, base_channels, time_dim)
        self.up_refine1 = AdaptiveResBlock(base_channels, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input_conv(x)
        hidden = self.down_block1(hidden, time_embedding)
        skip1 = self.down_refine1(hidden, time_embedding)
        hidden = self.downsample1(skip1)
        hidden = self.down_block2(hidden, time_embedding)
        skip2 = self.down_refine2(hidden, time_embedding)
        skip2 = self.attention14(skip2)
        hidden = self.downsample2(skip2)
        hidden = self.middle1(hidden, time_embedding)
        hidden = self.attention(hidden)
        hidden = self.middle2(hidden, time_embedding)
        hidden = self.upsample2(hidden)
        hidden = self.up_block2(torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = self.up_refine2(hidden, time_embedding)
        hidden = self.up_attention14(hidden)
        hidden = self.upsample1(hidden)
        hidden = self.up_block1(torch.cat((hidden, skip1), dim=1), time_embedding)
        hidden = self.up_refine1(hidden, time_embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(model_name: str = "mini", **model_config) -> nn.Module:
    """Build a checkpoint-compatible model variant by name."""

    models = {"mini": MiniUNet, "enhanced": EnhancedUNet}
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'; choose from {sorted(models)}")
    return models[model_name](**model_config)
