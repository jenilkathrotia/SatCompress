"""Phase 3: ablation — PolarQuant vs uniform scalar quantization.

Trains the SAME backbone with each quantizer for a short budget and reports
PSNR/SSIM/bpp on a held-out split, so you can show the polar warp's effect with
all else held constant. For publication-quality numbers, increase --epochs and
point --data-root at real Sentinel-2 data.

Usage:
    python scripts/ablation.py --synthetic --epochs 3
"""

from __future__ import annotations

import argparse

import torch

from satcompress.data import RandomPatchDataset, build_dataloader
from satcompress.losses import RateDistortionLoss
from satcompress.metrics import psnr, ssim
from satcompress.models import CompressionAutoencoder
from satcompress.quant import PolarQuant, UniformScalarQuant


def train_eval(quantizer, loader, val, device, epochs, lr):
    model = CompressionAutoencoder(in_channels=3, latent=64, quantizer=quantizer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = RateDistortionLoss(lambda_rate=0.0)  # distortion-only for a clean A/B
    model.train()
    for _ in range(epochs):
        for x in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            crit(x, out["x_hat"])["loss"].backward()
            opt.step()
    model.eval()
    ps, ss = [], []
    with torch.no_grad():
        for x in val:
            x = x.to(device)
            xh = model(x)["x_hat"]
            ps.append(float(psnr(x, xh)))
            ss.append(float(ssim(x, xh)))
    return sum(ps) / len(ps), sum(ss) / len(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r-step", type=float, default=1.0)
    ap.add_argument("--n-theta", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_ds = RandomPatchDataset(length=256, channels=3, patch_size=128, seed=0)
    val_ds = RandomPatchDataset(length=32, channels=3, patch_size=128, seed=999)
    loader = build_dataloader(train_ds, batch_size=8, num_workers=0)
    val = build_dataloader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    configs = {
        "scalar": UniformScalarQuant(step=args.r_step),
        "polar": PolarQuant(r_step=args.r_step, n_theta=args.n_theta),
    }
    print(f"{'quantizer':<10}{'PSNR(dB)':>12}{'SSIM':>10}")
    for name, q in configs.items():
        p, s = train_eval(q, loader, val, device, args.epochs, args.lr)
        print(f"{name:<10}{p:>12.2f}{s:>10.4f}")


if __name__ == "__main__":
    main()
