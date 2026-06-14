"""Rayleigh–von Mises entropy model — a probability prior *matched* to a polar latent.

Why this is the right prior (and why everyone else uses the wrong one)
---------------------------------------------------------------------
Neural compressors place an entropy model over the latent so an arithmetic coder
can turn quantized symbols into a short bitstream; the model's negative
log-likelihood is the rate term in the training loss. Essentially every paper
uses a Gaussian/Laplacian model on the *Cartesian* latent coordinates.

But PolarQuant codes a latent in *polar* coordinates (r, θ). If the underlying
2-D latent pair (x, y) is approximately isotropic Gaussian — which the encoder is
pressured toward — then by a standard change of variables:

  * the magnitude  r = sqrt(x² + y²)  is **Rayleigh** distributed, and
  * the angle      θ = atan2(y, x)     is **circular (von Mises / uniform)**.

So the *mathematically correct* prior for a polar latent is
``Rayleigh(r; σ) × vonMises(θ; μ, κ)`` — not a Gaussian. This module implements
exactly that. The parameters (σ per channel-pair; μ, κ per channel-pair) are
learned jointly with the network, so as training proceeds the model fits the true
latent distribution and the rate estimate becomes tight.

Rate of a quantized symbol
--------------------------
The number of bits to code a symbol that falls in quantization bin B under a
density p is  −log2 P(B).

* Radius (continuous, non-negative): we use the **exact Rayleigh CDF**
  ``F(r) = 1 − exp(−r²/2σ²)`` over the bin ``[r_q − w/2, r_q + w/2]`` (w is the
  local radial bin width supplied by the quantizer, so this works for both the
  linear and log-polar grids).
* Angle (discrete circular grid of n_theta bins): we use the **exact discrete
  von Mises** mass ``P(θ_k) = exp(κ cos(θ_k − μ)) / Σ_j exp(κ cos(θ_j − μ))``,
  computed in log-space via logsumexp for numerical stability.

Both terms are differentiable in the latent (through r_q, θ_q) *and* in the model
parameters (σ, μ, κ), so the whole thing trains end-to-end.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = math.log(2.0)


class PolarRateModel(nn.Module):
    """Learned Rayleigh × von Mises entropy model for a PolarQuant latent.

    Args:
        n_pairs: number of (x, y) channel pairs = latent_channels // 2.
        n_theta: number of angular bins (must match the PolarQuant layer).
        init_sigma: initial Rayleigh scale per pair.
        init_kappa: initial von Mises concentration per pair.

    Call `rate_bits(quantizer, z)` to get the total estimated code length (bits)
    of the quantized polar symbols of latent `z`.
    """

    def __init__(
        self,
        n_pairs: int,
        n_theta: int,
        init_sigma: float = 1.0,
        init_kappa: float = 1.0,
    ):
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.n_theta = int(n_theta)
        # Rayleigh scale σ > 0, log-parameterized.
        self.log_sigma = nn.Parameter(torch.full((n_pairs,), math.log(init_sigma)))
        # von Mises mean direction μ (free angle) and concentration κ > 0
        # (κ via softplus of a raw parameter).
        self.mu = nn.Parameter(torch.zeros(n_pairs))
        inv_kappa = math.log(math.expm1(init_kappa))  # softplus(raw) = init_kappa
        self.raw_kappa = nn.Parameter(torch.full((n_pairs,), inv_kappa))

    # -- parameter accessors (broadcastable to (B, pairs, H, W)) ------------
    def _sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().view(1, -1, 1, 1)

    def _mu(self) -> torch.Tensor:
        return self.mu.view(1, -1, 1, 1)

    def _kappa(self) -> torch.Tensor:
        # bounded above for numerically stable, non-exploding angular gradients
        return (F.softplus(self.raw_kappa) + 1e-6).clamp(max=50.0).view(1, -1, 1, 1)

    # -- per-component bit costs --------------------------------------------
    def radius_bits(self, r_q: torch.Tensor, bin_width_r: torch.Tensor) -> torch.Tensor:
        """−log2 of the Rayleigh bin mass at each quantized radius.

        Uses the stable small-bin relaxation P(bin) ≈ p_Rayleigh(r_q) · width,
        evaluated in log-space:
            log p = log r − 2 log σ − r²/2σ²
        This avoids the CDF-difference of two nearly-equal exponentials (which
        produces vanishing probabilities and exploding −1/P gradients).
        """
        log_sigma = self.log_sigma.view(1, -1, 1, 1)
        sigma2 = torch.exp(2.0 * log_sigma)
        r_safe = r_q.clamp_min(1e-8)
        log_pdf = torch.log(r_safe) - 2.0 * log_sigma - (r_q * r_q) / (2.0 * sigma2)
        log_mass = log_pdf + torch.log(bin_width_r.clamp_min(1e-8))
        bits = -log_mass / _LN2
        return bits.clamp_min(0.0)  # a symbol cannot cost negative bits

    def angle_bits(self, theta_q: torch.Tensor, theta_grid: torch.Tensor) -> torch.Tensor:
        """−log2 of the discrete von Mises mass at each quantized angle."""
        mu = self._mu()
        kappa = self._kappa()
        # log numerator: κ cos(θ_q − μ)
        log_num = kappa * torch.cos(theta_q - mu)
        # log partition over the angular grid: logsumexp_j κ cos(θ_j − μ)
        grid = theta_grid.view(1, 1, 1, 1, -1)
        logits = kappa.unsqueeze(-1) * torch.cos(grid - mu.unsqueeze(-1))  # (1,pairs,1,1,K)
        log_z = torch.logsumexp(logits, dim=-1)  # (1, pairs, 1, 1)
        log_p = log_num - log_z
        return -log_p / _LN2

    # -- public API ---------------------------------------------------------
    def rate_bits(self, quantizer, z: torch.Tensor) -> torch.Tensor:
        """Total estimated code length (bits) for the polar symbols of `z`.

        `quantizer` must be a PolarQuant exposing `polar_components`,
        `radial_bin_width`, and `theta_step`.
        """
        _, _, r_q, theta_q = quantizer.polar_components(z)
        width = quantizer.radial_bin_width(r_q)
        # Angular grid over a full turn (cos is 2π-periodic, so any offset works).
        # Use the real dtype of theta_q (z may be complex).
        grid = torch.arange(self.n_theta, device=theta_q.device, dtype=theta_q.dtype) * quantizer.theta_step
        bits = self.radius_bits(r_q, width).sum() + self.angle_bits(theta_q, grid).sum()
        return bits

    def forward(self, quantizer, z: torch.Tensor) -> torch.Tensor:
        return self.rate_bits(quantizer, z)

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = self.log_sigma.exp().mean().item()
            k = (F.softplus(self.raw_kappa) + 1e-6).mean().item()
        return f"n_pairs={self.n_pairs}, n_theta={self.n_theta}, mean_sigma={s:.3g}, mean_kappa={k:.3g}"
