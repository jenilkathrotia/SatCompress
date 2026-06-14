"""Image-quality metrics: PSNR, SSIM, and rate (bits-per-pixel) helpers.

All functions operate on tensors/arrays in [0, 1]. PSNR and SSIM are reported
per-image and then averaged over a batch, matching the convention used in the
neural-compression literature (Ballé et al. 2018, Cheng et al. 2020).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio, averaged over the batch.

    Args:
        x, y: tensors of shape (B, C, H, W) in [0, max_val].
    Returns:
        Scalar tensor: mean PSNR (dB) across the batch.
    """
    x, y = x.float(), y.float()  # compute in fp32 (AMP-safe, and more accurate)
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    return (10.0 * torch.log10((max_val ** 2) / (mse + eps))).mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Structural Similarity Index (Gaussian-windowed), averaged over batch & channels.

    Differentiable, so it can also be used as (1 - SSIM) loss term. Computed in
    fp32 so it is safe under autocast/AMP (avoids fp16/fp32 conv dtype clashes).
    Args:
        x, y: tensors of shape (B, C, H, W) in [0, max_val].
    """
    x, y = x.float(), y.float()  # AMP-safe: keep window + convs in one dtype
    c, = (x.shape[1],)
    win = _gaussian_window(window_size, sigma, x.device, x.dtype)
    win = win.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(t):
        return F.conv2d(t, win, padding=pad, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = filt(x * x) - mu_x2
    sigma_y2 = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den).mean()


def bpp(num_bits: float, image_hw: tuple[int, int]) -> float:
    """Bits per pixel for a compressed code of `num_bits` over an H x W image."""
    h, w = image_hw
    return num_bits / float(h * w)


def estimate_entropy_bits(symbols: torch.Tensor) -> float:
    """Lower-bound code length (bits) via the empirical Shannon entropy of symbols.

    Used to estimate the achievable rate of a quantized latent without a full
    entropy coder. `symbols` should be an integer-valued tensor (quantization
    indices). Returns total bits = N * H(p).
    """
    flat = symbols.detach().round().long().flatten()
    counts = torch.bincount(flat - flat.min())
    probs = counts.float() / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-(probs * torch.log2(probs)).sum())
    return entropy * flat.numel()
