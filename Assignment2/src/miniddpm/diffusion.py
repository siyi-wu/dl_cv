"""Forward diffusion and DDPM/DDIM reverse samplers."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, min(scale * 2e-2, 0.999), timesteps)


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        device: str | torch.device = "cpu",
    ):
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.timesteps = timesteps
        self.schedule = schedule
        self.device = torch.device(device)
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        gathered = values.gather(0, timesteps)
        return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw x_t directly from q(x_t | x_0)."""

        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._extract(self.alpha_bars.sqrt(), timesteps, x_start.shape)
        sqrt_one_minus = self._extract((1 - self.alpha_bars).sqrt(), timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise, noise

    def inference_timesteps(self, steps: int) -> list[int]:
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f"steps must be in [1, {self.timesteps}]")
        values = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        unique = list(dict.fromkeys(values))
        if len(unique) != steps:
            raise RuntimeError("Failed to construct unique inference timesteps")
        return unique

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        steps: int | None = None,
        sampler: str = "ancestral",
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Sample on a respaced timeline with ancestral DDPM or DDIM."""

        steps = steps or self.timesteps
        timeline = self.inference_timesteps(steps)
        if sampler not in {"ancestral", "ddim"}:
            raise ValueError("sampler must be 'ancestral' or 'ddim'")
        if eta < 0:
            raise ValueError("eta must be non-negative")
        x = (
            torch.randn(shape, device=self.device)
            if initial_noise is None
            else initial_noise.clone().to(self.device)
        )
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"initial_noise has shape {tuple(x.shape)}, expected {shape}")

        model.eval()
        for index, timestep in enumerate(timeline):
            next_timestep = timeline[index + 1] if index + 1 < len(timeline) else -1
            t_batch = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            alpha_bar_t = self.alpha_bars[timestep]
            alpha_bar_s = (
                self.alpha_bars[next_timestep]
                if next_timestep >= 0
                else torch.ones((), device=self.device)
            )
            predicted_start = (
                x - torch.sqrt(1 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            predicted_start = predicted_start.clamp(-1, 1)

            if sampler == "ancestral":
                effective_alpha = alpha_bar_t / alpha_bar_s
                effective_beta = 1 - effective_alpha
                denominator = (1 - alpha_bar_t).clamp_min(1e-12)
                mean = (
                    torch.sqrt(alpha_bar_s) * effective_beta / denominator * predicted_start
                    + torch.sqrt(effective_alpha) * (1 - alpha_bar_s) / denominator * x
                )
                variance = ((1 - alpha_bar_s) / denominator * effective_beta).clamp_min(0)
                x = mean if next_timestep < 0 else mean + variance.sqrt() * torch.randn_like(x)
            else:
                sigma = eta * torch.sqrt(
                    ((1 - alpha_bar_s) / (1 - alpha_bar_t).clamp_min(1e-12))
                    * (1 - alpha_bar_t / alpha_bar_s)
                ).clamp_min(0)
                direction_scale = (1 - alpha_bar_s - sigma.square()).clamp_min(0).sqrt()
                x = torch.sqrt(alpha_bar_s) * predicted_start + direction_scale * predicted_noise
                if next_timestep >= 0 and eta > 0:
                    x = x + sigma * torch.randn_like(x)
            if progress_callback is not None:
                progress_callback(index + 1, steps)
        return x.clamp(-1, 1)
