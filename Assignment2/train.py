from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from miniddpm.data import MNISTDataset
from miniddpm.diffusion import GaussianDiffusion
from miniddpm.model import build_model
from miniddpm.utils import atomic_json_dump, resolve_device, save_image_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact DDPM on MNIST")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "train_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--model", choices=("mini", "enhanced"), default="mini")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--time-dim", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint")
    parser.add_argument("--preview-every", type=int, default=1, help="Generate a preview every N epochs; 0 disables")
    parser.add_argument("--preview-steps", type=int, default=50)
    parser.add_argument("--preview-samples", type=int, default=16)
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-project", default="mnist-ddpm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-every", type=int, default=10)
    parser.add_argument("--wandb-watch", action="store_true", help="Log parameter and gradient histograms")
    return parser.parse_args()


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in ema_model.named_parameters():
            ema_parameter.mul_(decay).add_(model_parameters[name], alpha=1 - decay)
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            ema_buffer.copy_(model_buffers[name])


def save_loss_curve(losses: list[float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    axis.plot(range(1, len(losses) + 1), losses, linewidth=1.1, color="#2563eb")
    axis.set(title="DDPM training loss", xlabel="Optimization step", ylabel="Noise MSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def checkpoint_payload(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    epoch_completed: bool,
    global_step: int,
    losses: list[float],
    wandb_run_id: str | None,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_name": args.model,
        "model_config": {"in_channels": 1, "base_channels": args.base_channels, "time_dim": args.time_dim},
        "diffusion_config": {"timesteps": args.timesteps, "schedule": args.schedule},
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "epoch": epoch,
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "losses": losses,
        "wandb_run_id": wandb_run_id,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.wandb_log_every < 1:
        raise ValueError("epochs, batch-size and wandb-log-every must be positive")
    if args.base_channels is None:
        args.base_channels = 48 if args.model == "enhanced" else 32
    if args.time_dim is None:
        args.time_dim = 192 if args.model == "enhanced" else 128
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.resume is None:
        metrics_path.unlink(missing_ok=True)

    dataset = MNISTDataset(args.data_dir, train=True, download=True)
    if args.max_train_samples is not None:
        if not 1 <= args.max_train_samples <= len(dataset):
            raise ValueError(f"max-train-samples must be in [1, {len(dataset)}]")
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.max_train_samples].tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        args.model, in_channels=1, base_channels=args.base_channels, time_dim=args.time_dim
    ).to(device)
    if model.parameter_count > 5_000_000:
        raise RuntimeError(f"Model has {model.parameter_count:,} parameters, exceeding the 5M limit")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(args.timesteps, args.schedule, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    losses: list[float] = []
    global_step = 0
    start_epoch = 1
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        expected_model_name = args.model
        expected_model_config = {"in_channels": 1, "base_channels": args.base_channels, "time_dim": args.time_dim}
        expected_diffusion_config = {"timesteps": args.timesteps, "schedule": args.schedule}
        if (
            resume_state.get("model_name", "mini") != expected_model_name
            or resume_state["model_config"] != expected_model_config
            or resume_state["diffusion_config"] != expected_diffusion_config
        ):
            raise ValueError("Resume checkpoint model/diffusion settings do not match the command line")
        model.load_state_dict(resume_state["model"])
        ema_model.load_state_dict(resume_state["ema_model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        losses = list(resume_state.get("losses", []))
        global_step = int(resume_state.get("global_step", 0))
        resume_epoch = int(resume_state.get("epoch", 0))
        start_epoch = resume_epoch + 1 if resume_state.get("epoch_completed", True) else resume_epoch
        print(f"resuming={args.resume} start_epoch={start_epoch} global_step={global_step}")

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B logging requested; install requirements.txt first") from error
        resume_run_id = resume_state.get("wandb_run_id") if resume_state else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            id=resume_run_id,
            resume="allow" if resume_run_id else None,
            dir=str(args.output_dir),
        )
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("train/*", step_metric="global_step")
        wandb_run.define_metric("epoch/*", step_metric="epoch")
        if args.wandb_watch:
            wandb_run.watch(model, log="all", log_freq=args.wandb_log_every)

    start_time = time.perf_counter()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"device={device} ({device_name}) samples={len(dataset)} parameters={model.parameter_count:,}")
    # If a completed checkpoint is resumed with the same --epochs value, the
    # training loop is intentionally skipped and the recorded epoch stays exact.
    epoch = max(start_epoch - 1, 0)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
            for images, _ in progress:
                images = images.to(device, non_blocking=True)
                timesteps = torch.randint(0, args.timesteps, (images.shape[0],), device=device)
                noisy_images, target_noise = diffusion.q_sample(images, timesteps)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    predicted_noise = model(noisy_images, timesteps)
                    loss = nn.functional.mse_loss(predicted_noise, target_noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                update_ema(ema_model, model, args.ema_decay)

                value = float(loss.detach().cpu())
                losses.append(value)
                epoch_losses.append(value)
                global_step += 1
                progress.set_postfix(loss=f"{value:.4f}")
                if wandb_run is not None and global_step % args.wandb_log_every == 0:
                    wandb_run.log(
                        {
                            "global_step": global_step,
                            "train/loss": value,
                            "train/gradient_norm": float(gradient_norm),
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        }
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "last_loss": losses[-1],
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            preview_path = None
            if args.preview_every > 0 and epoch % args.preview_every == 0:
                preview = diffusion.sample(
                    ema_model,
                    (args.preview_samples, 1, 28, 28),
                    steps=args.preview_steps,
                    sampler="ddim",
                    eta=0,
                )
                preview_path = args.output_dir / "previews" / f"epoch_{epoch:03d}.png"
                save_image_grid(preview, preview_path, columns=min(8, args.preview_samples))
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch,
                    "epoch/mean_loss": epoch_record["mean_loss"],
                    "epoch/last_loss": epoch_record["last_loss"],
                }
                if preview_path is not None:
                    log_payload["epoch/samples"] = wandb.Image(str(preview_path), caption=f"epoch {epoch}, DDIM {args.preview_steps} steps")
                wandb_run.log(log_payload)
            torch.save(
                checkpoint_payload(
                    model, ema_model, optimizer, scaler, args, epoch, True, global_step, losses,
                    wandb_run.id if wandb_run is not None else None,
                ),
                args.output_dir / "checkpoint_last.pt",
            )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
    except KeyboardInterrupt:
        interrupted_path = args.output_dir / "checkpoint_interrupted.pt"
        torch.save(
            checkpoint_payload(
                model, ema_model, optimizer, scaler, args, epoch, False, global_step, losses,
                wandb_run.id if wandb_run is not None else None,
            ),
            interrupted_path,
        )
        save_loss_curve(losses, args.output_dir / "loss_curve.png")
        print(f"Training interrupted; resumable checkpoint saved to {interrupted_path}")
        if wandb_run is not None:
            wandb_run.summary["interrupted_at_step"] = global_step
            wandb_run.finish(exit_code=130)
        raise SystemExit(130)

    elapsed = time.perf_counter() - start_time
    checkpoint = checkpoint_payload(
        model, ema_model, optimizer, scaler, args, epoch, True, global_step, losses,
        wandb_run.id if wandb_run is not None else None,
    )
    torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
    save_loss_curve(losses, args.output_dir / "loss_curve.png")
    summary = {
        "device": str(device),
        "device_name": device_name,
        "dataset_samples": len(dataset),
        "parameter_count": model.parameter_count,
        "epochs_completed": epoch,
        "optimization_steps": global_step,
        "elapsed_seconds": elapsed,
        "final_loss": losses[-1],
        "mean_last_10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "checkpoint": str((args.output_dir / "checkpoint_last.pt").resolve()),
    }
    atomic_json_dump(summary, args.output_dir / "training_summary.json")
    if wandb_run is not None:
        for key, value in summary.items():
            if key != "checkpoint":
                wandb_run.summary[key] = value
        # wandb.Run.save() creates a symlink and fails for standard Windows
        # users without SeCreateSymbolicLinkPrivilege. Media logging uploads a
        # copy and works without administrator or Developer Mode privileges.
        wandb_run.log(
            {
                "global_step": global_step,
                "training/loss_curve": wandb.Image(
                    str(args.output_dir / "loss_curve.png"),
                    caption="Complete DDPM training loss",
                ),
            }
        )
        wandb_run.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
