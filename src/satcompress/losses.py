"""Rate-distortion(-phase) loss for learned compression.

The compression objective is a Lagrangian trade-off

    L = D(x, x_hat) + lambda_rate * R(z_hat) + phase_weight * P(x, x_hat)

where D is distortion, R is the (estimated) rate of the quantized latent, and the
optional P term is a Fourier *phase* loss. Distortion mixes MSE with a perceptual
(1 - SSIM) component, since SSIM > 0.95 is the project's fidelity target. Rate can
come from the simple empirical-entropy estimate (`metrics.estimate_entropy_bits`)
or, preferably, the matched `quant.PolarRateModel`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ssim


class FourierPhaseLoss(nn.Module):
    """Penalize phase-spectrum mismatch between x and x_hat.

    Motivated by Oppenheim & Lim (1981): the Fourier *phase* carries an image's
    edges/structure. We compare the phase angles of the 2-D FFTs and use
    1 − cos(Δphase) (0 when phases agree, up to 2 when opposed). Optionally weight
    each frequency by the original's magnitude so dominant structures count most.
    """

    def __init__(self, magnitude_weighted: bool = True, eps: float = 1e-8):
        super().__init__()
        self.magnitude_weighted = magnitude_weighted
        self.eps = eps

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        fx = torch.fft.rfft2(x, norm="ortho")
        fy = torch.fft.rfft2(x_hat, norm="ortho")
        per_freq = 1.0 - torch.cos(torch.angle(fx) - torch.angle(fy))
        if self.magnitude_weighted:
            w = fx.abs()
            w = w / (w.mean() + self.eps)
            per_freq = per_freq * w
        return per_freq.mean()


class RateDistortionLoss(nn.Module):
    """L = distortion + lambda_rate * rate_bpp + phase_weight * phase_loss."""

    def __init__(
        self,
        lambda_rate: float = 0.01,
        ssim_weight: float = 0.2,
        phase_weight: float = 0.0,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.ssim_weight = ssim_weight
        self.phase_weight = phase_weight
        self.phase_loss = FourierPhaseLoss() if phase_weight > 0 else None

    def forward(self, x, x_hat, z_hat=None, rate_bits: torch.Tensor | None = None):
        mse = F.mse_loss(x_hat, x)
        ssim_term = 1.0 - ssim(x, x_hat)
        distortion = (1.0 - self.ssim_weight) * mse + self.ssim_weight * ssim_term

        if rate_bits is not None:
            num_px = x.shape[-1] * x.shape[-2]
            rate = rate_bits / (x.shape[0] * num_px)  # bits per pixel
        else:
            rate = torch.zeros((), device=x.device)

        total = distortion + self.lambda_rate * rate

        phase_val = torch.zeros((), device=x.device)
        if self.phase_loss is not None:
            phase_val = self.phase_loss(x, x_hat)
            total = total + self.phase_weight * phase_val

        return {
            "loss": total,
            "mse": mse.detach(),
            "ssim_term": ssim_term.detach(),
            "rate_bpp": rate.detach() if torch.is_tensor(rate) else torch.tensor(rate),
            "phase": phase_val.detach(),
        }
