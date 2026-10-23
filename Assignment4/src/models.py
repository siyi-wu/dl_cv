"""Two classifiers and a small time-conditioned U-Net, implemented in PyTorch."""
import math
import torch
from torch import nn


def classifier(name):
    if name == 'mlp':
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    if name == 'cnn':
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10))
    raise ValueError(name)


class Block(nn.Module):
    def __init__(self, inputs, outputs, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.time = nn.Linear(time_dim, outputs)
        self.net = nn.Sequential(nn.GroupNorm(8, outputs), nn.SiLU(),
                                 nn.Conv2d(outputs, outputs, 3, padding=1),
                                 nn.GroupNorm(8, outputs), nn.SiLU())

    def forward(self, x, time):
        return self.net(self.conv(x) + self.time(time)[:, :, None, None])


class Denoiser(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000)*torch.arange(32)/31))
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.down = Block(1, channels, 128)
        self.mid = Block(channels, channels*2, 128)
        self.up = Block(channels*3, channels, 128)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x, t):
        angle = t.float()[:, None] * self.freq[None]
        time = self.time(torch.cat([angle.sin(), angle.cos()], 1))
        skip = self.down(x, time)
        low = self.mid(torch.nn.functional.avg_pool2d(skip, 2), time)
        high = torch.nn.functional.interpolate(low, size=x.shape[-2:], mode='nearest')
        return self.out(self.up(torch.cat([skip, high], 1), time))


class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps
        self.beta = torch.linspace(1e-4, 0.05, steps, device=device)
        self.alpha = 1 - self.beta
        self.abar = self.alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, device=device), self.abar[:-1]])
        self.variance = self.beta * (1-previous) / (1-self.abar)

    def add_noise(self, clean, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt()*clean + (1-a).sqrt()*noise

    def reverse_step(self, model, x, step, noise):
        t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
        eps = model(x, t)
        # Predict x_0, clip to the training range, then use the DDPM posterior mean.
        x0 = ((x-(1-self.abar[step]).sqrt()*eps)/self.abar[step].sqrt()).clamp(-1, 1)
        previous = self.abar[step-1] if step else x.new_tensor(1.)
        mean = (previous.sqrt()*self.beta[step]/(1-self.abar[step])*x0
                + self.alpha[step].sqrt()*(1-previous)/(1-self.abar[step])*x)
        return mean + self.variance[step].sqrt()*noise if step else mean

    @torch.no_grad()
    def sample(self, model, initial, seed):
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        x = initial.clone()
        snapshots = [x[:8].cpu()]
        selected = {self.steps*3//4, self.steps//2, self.steps//4, 0}
        for step in reversed(range(self.steps)):
            noise = torch.randn(x.shape, generator=generator, device=x.device)
            x = self.reverse_step(model, x, step, noise)
            if step in selected:
                snapshots.append(x[:8].cpu())
        return x, snapshots
