"""Rate-distortion sweep — the fair, publication-style comparison.

Single points at different bitrates are meaningless (a smaller file *should* look
worse). The honest comparison is a rate-distortion CURVE: run each method at
several operating points, then plot bits-per-pixel (bpp) vs quality (PSNR/SSIM).
The better method's curve sits higher (more quality per bit) / further left.

The "rate knob" here is quantization coarseness: a finer grid (smaller r_step,
more angular bins) keeps more detail and costs more bits; a coarser grid costs
fewer. Each setting -> one point on the curve.

bpp is measured the SAME way for every neural method — the empirical (Shannon)
entropy of the quantized symbols — so methods are comparable regardless of which
entropy model they trained with. Classical JPEG / JPEG2000 are added on the same
held-out tiles.

Output: results/rd_results.csv with columns method,setting,bpp,psnr,ssim.

Usage:
    python scripts/rd_sweep.py --data-root data/s2 --channels 4 --patch-size 256 \
        --reflectance-scale 10000 --epochs 10 --batch-size 48 --amp fp16
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from satcompress.baselines import rate_distortion_sweep
from satcompress.data import RandomPatchDataset, Sentinel2PatchDataset
from satcompress.losses import RateDistortionLoss
from satcompress.metrics import estimate_entropy_bits, psnr, ssim
from satcompress.models import CompressionAutoencoder
from satcompress.quant import PolarQuant, PolarRateModel, UniformScalarQuant


def autodevice(choice="auto"):
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Rate-distortion sweep")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--reflectance-scale", type=float, default=10000.0)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--latent", type=int, default=192)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="fp16")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--baseline-limit", type=int, default=120)
    p.add_argument("--out", type=str, default="results/rd_results.csv")
    return p.parse_args()


def sweep_configs(latent):
    """(method, setting_label, build_quantizer, build_rate_model_or_None, lambda)."""
    pairs = latent // 2
    cfgs = []
    # scalar control: rate knob = step
    for s in (0.5, 1.0, 2.0, 4.0):
        cfgs.append(("scalar", f"step={s}", (lambda s=s: UniformScalarQuant(step=s)), None, 0.0))
    # PolarQuant (linear): rate knob = r_step (n_theta fixed fine at 32)
    for rs in (0.5, 1.0, 2.0, 4.0):
        cfgs.append(("polar", f"r={rs},nθ=32",
                     (lambda rs=rs: PolarQuant(r_step=rs, n_theta=32)), None, 0.0))
    # PolarQuant + log-polar + matched Rayleigh-vM entropy model
    for rs in (0.5, 1.0, 2.0, 4.0):
        cfgs.append(("polar-log-rvm", f"r={rs},nθ=32",
                     (lambda rs=rs: PolarQuant(r_step=rs, n_theta=32, radial_mode="log")),
                     (lambda: PolarRateModel(n_pairs=pairs, n_theta=32)), 0.01))
    return cfgs


def _quant_of(model):
    return model.module.quantizer if isinstance(model, nn.DataParallel) else model.quantizer


def _symbol_bits(quant, z):
    sym = quant.symbols(z)
    if isinstance(sym, tuple):
        return sum(float(estimate_entropy_bits(s)) for s in sym)
    return float(estimate_entropy_bits(sym))


@torch.no_grad()
def evaluate(model, loader, device):
    """Held-out (bpp, psnr, ssim). bpp = empirical entropy of symbols / pixels."""
    model.eval()
    quant = _quant_of(model)
    tot_bits, tot_px, n, ps, ss = 0.0, 0, 0, 0.0, 0.0
    for x in loader:
        x = x.to(device)
        out = model(x)
        xh = out["x_hat"]
        bs = x.size(0)
        ps += float(psnr(x, xh)) * bs
        ss += float(ssim(x, xh)) * bs
        tot_bits += _symbol_bits(quant, out["z"])
        tot_px += bs * x.shape[-1] * x.shape[-2]
        n += bs
    return tot_bits / tot_px, ps / n, ss / n


def train_and_eval(make_quant, make_rate, lam, train_loader, val_loader, args, device):
    quant = make_quant()
    model = CompressionAutoencoder(in_channels=args.channels, latent=args.latent, quantizer=quant).to(device)
    rate_model = make_rate().to(device) if make_rate else None
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    params = list(model.parameters()) + (list(rate_model.parameters()) if rate_model else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    crit = RateDistortionLoss(lambda_rate=lam)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and use_amp))

    model.train()
    for _ in range(args.epochs):
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                q = _quant_of(model)
                rate_bits = rate_model.rate_bits(q, out["z"]) if rate_model else None
                loss = crit(x, out["x_hat"], out["z_hat"], rate_bits)["loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
    return evaluate(model, val_loader, device)


def main():
    args = parse_args()
    device = autodevice(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[rd-sweep] device={device}")

    if args.synthetic or not args.data_root:
        ds = RandomPatchDataset(length=400, channels=args.channels, patch_size=args.patch_size)
    else:
        ds = Sentinel2PatchDataset(
            args.data_root, patch_size=args.patch_size,
            reflectance_scale=args.reflectance_scale, patches_per_scene=1,
        )
    n = len(ds)
    val_n = max(20, int(n * args.val_frac))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:val_n], perm[val_n:]
    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"[rd-sweep] train={len(train_idx)} val={len(val_idx)} tiles")

    rows = []
    for method, setting, mq, mr, lam in sweep_configs(args.latent):
        bpp, p, s = train_and_eval(mq, mr, lam, train_loader, val_loader, args, device)
        print(f"[rd-sweep] {method:<14} {setting:<12} bpp={bpp:.3f} psnr={p:.2f} ssim={s:.4f}")
        rows.append({"method": method, "setting": setting, "bpp": bpp, "psnr": p, "ssim": s})

    # classical baselines on the same held-out tiles
    val_images = [ds[i] for i in val_idx[: args.baseline_limit]]
    for rec in rate_distortion_sweep(val_images):
        if "error" in rec:
            continue
        rows.append({"method": rec["codec"], "setting": str(rec["setting"]),
                     "bpp": rec["bpp"], "psnr": rec["psnr"], "ssim": rec["ssim"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "setting", "bpp", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print(f"[rd-sweep] wrote {out} ({len(rows)} points). Plot with scripts/plot_rd.py")


if __name__ == "__main__":
    main()
