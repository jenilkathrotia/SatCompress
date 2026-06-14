"""Training entry point for SatCompress.

Quantizer (--quantizer):
  none   : vanilla autoencoder, no quantization (Phase 2 reconstruction sanity)
  scalar : uniform scalar quantization + STE (ablation control)
  polar  : PolarQuant (proposed method)

Research extensions:
  --radial-mode {linear,log}      log-polar companding of the radius
  --entropy-model {empirical,rayleigh-vm}
                                  rate model: simple empirical entropy (logging
                                  only, non-differentiable) or the matched
                                  Rayleigh-von Mises model (learned, trainable)
  --complex                       use the complex-valued encoder/decoder
  --phase-weight FLOAT            weight of the Fourier phase loss (>0 enables it)

Features
--------
* Mixed precision (bf16/fp16) for H100 Tensor Cores (auto-off for --complex).
* Weights & Biases logging — disabled gracefully if unavailable or --no-wandb.
* Works out-of-the-box on synthetic data (--synthetic).

Examples
--------
    python -m satcompress.train --quantizer polar --synthetic --no-wandb
    python -m satcompress.train --quantizer polar --radial-mode log \
        --entropy-model rayleigh-vm --synthetic --no-wandb
    python -m satcompress.train --complex --quantizer polar --phase-weight 0.1 \
        --synthetic --no-wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import RandomPatchDataset, Sentinel2PatchDataset, build_dataloader
from .losses import RateDistortionLoss
from .metrics import estimate_entropy_bits, psnr, ssim
from .models import CompressionAutoencoder, ComplexCompressionAutoencoder, ComplexPolarQuant
from .quant import PolarQuant, PolarRateModel, UniformScalarQuant


def build_model_and_quant(args, device):
    """Construct (model, quantizer, rate_model) for the chosen configuration."""
    # --- quantizer ---
    if args.complex:
        if args.quantizer not in ("polar", "none"):
            raise ValueError("--complex supports only --quantizer polar|none")
        quant = (
            None
            if args.quantizer == "none"
            else ComplexPolarQuant(
                r_step=args.r_step, n_theta=args.n_theta,
                radial_mode=args.radial_mode, learnable=args.learnable_quant,
            )
        )
        model = ComplexCompressionAutoencoder(
            in_channels=args.channels, latent=args.latent, quantizer=quant
        )
        n_pairs = args.latent  # each complex channel is one (x, y) pair
    else:
        quant = _build_real_quant(args)
        model = CompressionAutoencoder(
            in_channels=args.channels, latent=args.latent, quantizer=quant
        )
        n_pairs = args.latent // 2

    # --- matched entropy model (only for polar quantizers) ---
    rate_model = None
    if args.entropy_model == "rayleigh-vm":
        is_polar = isinstance(quant, (PolarQuant, ComplexPolarQuant))
        if not is_polar:
            raise ValueError("--entropy-model rayleigh-vm requires a polar quantizer")
        rate_model = PolarRateModel(n_pairs=n_pairs, n_theta=args.n_theta)

    model = model.to(device)
    if rate_model is not None:
        rate_model = rate_model.to(device)
    return model, quant, rate_model


def _build_real_quant(args):
    if args.quantizer == "none":
        return None
    if args.quantizer == "scalar":
        return UniformScalarQuant(step=args.r_step, learnable_step=args.learnable_quant)
    if args.quantizer == "polar":
        return PolarQuant(
            r_step=args.r_step, n_theta=args.n_theta,
            radial_mode=args.radial_mode, learnable=args.learnable_quant,
        )
    raise ValueError(f"unknown quantizer: {args.quantizer}")


def autodevice(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # NOTE: Apple MPS currently mishandles some ops used here (atan2/logsumexp)
    # and can yield NaNs; prefer CPU on Mac for these research configs, or pass
    # --device mps explicitly to try it.
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Train SatCompress")
    p.add_argument("--data-root", type=str, default=None, help="dir of Sentinel-2 scenes")
    p.add_argument("--reflectance-scale", type=float, default=10000.0,
                   help="divide raw pixels by this to reach [0,1]; 255 for 8-bit (EuroSAT)")
    p.add_argument("--synthetic", action="store_true", help="use synthetic patches")
    p.add_argument("--quantizer", choices=["none", "scalar", "polar"], default="polar")
    p.add_argument("--r-step", type=float, default=1.0)
    p.add_argument("--n-theta", type=int, default=16)
    p.add_argument("--radial-mode", choices=["linear", "log"], default="linear")
    p.add_argument("--entropy-model", choices=["empirical", "rayleigh-vm"], default="empirical")
    p.add_argument("--complex", action="store_true", help="use complex-valued network")
    p.add_argument("--phase-weight", type=float, default=0.0, help="Fourier phase loss weight")
    p.add_argument("--learnable-quant", action="store_true")
    p.add_argument("--latent", type=int, default=192)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patches-per-scene", type=int, default=16,
                   help="random crops per scene per epoch; use 1 for pre-extracted tiles")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-rate", type=float, default=0.01)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="satcompress")
    return p.parse_args()


def main():
    args = parse_args()
    device = autodevice(args.device)
    print(f"[satcompress] device={device} quantizer={args.quantizer} "
          f"radial={args.radial_mode} entropy={args.entropy_model} complex={args.complex}")

    # --- data ---
    if args.synthetic or not args.data_root:
        ds = RandomPatchDataset(length=512, channels=args.channels, patch_size=args.patch_size)
        workers = 0
    else:
        ds = Sentinel2PatchDataset(
            args.data_root, patch_size=args.patch_size,
            reflectance_scale=args.reflectance_scale,
            patches_per_scene=args.patches_per_scene,
        )
        workers = args.num_workers
    loader = build_dataloader(
        ds, batch_size=args.batch_size, num_workers=workers, pin_memory=(device.type == "cuda")
    )

    # --- model + quantizer + entropy model ---
    model, quant, rate_model = build_model_and_quant(args, device)
    params = list(model.parameters()) + (list(rate_model.parameters()) if rate_model else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    criterion = RateDistortionLoss(lambda_rate=args.lambda_rate, phase_weight=args.phase_weight)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    # complex tensors don't play well with autocast -> disable amp for --complex
    use_amp = amp_dtype is not None and device.type == "cuda" and not args.complex
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and use_amp))

    run = _init_wandb(args)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                rate_bits = _compute_rate(quant, rate_model, out["z"])
                loss_d = criterion(x, out["x_hat"], out["z_hat"], rate_bits)

            loss = loss_d["loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()

            if step % 20 == 0:
                with torch.no_grad():
                    p = float(psnr(x, out["x_hat"]))
                    s = float(ssim(x, out["x_hat"]))
                    loss_v = loss.item()
                print(
                    f"epoch {epoch} step {step} loss {loss_v:.4f} "
                    f"psnr {p:.2f} ssim {s:.4f} bpp {float(loss_d['rate_bpp']):.4f} "
                    f"phase {float(loss_d['phase']):.4f}"
                )
                _log(run, {
                    "loss": loss_v, "mse": float(loss_d["mse"]),
                    "psnr": p, "ssim": s, "rate_bpp": float(loss_d["rate_bpp"]),
                    "phase": float(loss_d["phase"]), "epoch": epoch,
                }, step)
            step += 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{'complex_' if args.complex else ''}{args.quantizer}"
    ckpt = out_dir / f"satcompress_{tag}.pt"
    save = {"model": model.state_dict(), "args": vars(args)}
    if rate_model is not None:
        save["rate_model"] = rate_model.state_dict()
    torch.save(save, ckpt)
    print(f"[satcompress] saved {ckpt}")
    if run is not None:
        run.finish()


def _compute_rate(quant, rate_model, z):
    """Differentiable matched-model rate if available, else a logged estimate."""
    if quant is None:
        return None
    if rate_model is not None:
        return rate_model.rate_bits(quant, z)  # carries gradient -> trains encoder
    if hasattr(quant, "symbols"):
        return _estimate_rate_bits(quant, z)  # detached measurement
    return None


def _estimate_rate_bits(quant, z) -> torch.Tensor:
    """Empirical-entropy rate estimate for the quantizer's symbols (non-diff)."""
    sym = quant.symbols(z)
    if isinstance(sym, tuple):  # polar: (r_idx, theta_idx)
        bits = sum(estimate_entropy_bits(s) for s in sym)
    else:
        bits = estimate_entropy_bits(sym)
    return torch.tensor(bits, device=z.device)


def _init_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb

        return wandb.init(project=args.wandb_project, config=vars(args))
    except Exception as e:  # not installed / not logged in
        print(f"[satcompress] W&B disabled: {e}")
        return None


def _log(run, metrics: dict, step: int):
    if run is not None:
        run.log(metrics, step=step)


if __name__ == "__main__":
    main()
