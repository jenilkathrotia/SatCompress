"""PolarQuant: projective (polar) warping of the latent space before quantization.

Motivation
----------
Standard neural compressors apply uniform scalar quantization on a Cartesian
grid: each latent value is independently snapped to round(z/delta)*delta. The
reconstruction cells are axis-aligned hypercubes. For natural and especially
satellite imagery, the dominant high-frequency structure (coastlines, field
boundaries, roads) produces latent activations that are better described by an
*orientation* (theta) and a *magnitude* (r) than by independent x/y coordinates.

PolarQuant pairs up latent channels into (x, y) couples, maps them to polar
coordinates, and quantizes the radius and angle on *separate* grids:

    r     = sqrt(x^2 + y^2)
    theta = atan2(y, x)
    r_hat     = round(r / r_step) * r_step
    theta_hat = round(theta / theta_step) * theta_step      (theta_step = 2*pi / n_theta)
    x_hat = r_hat * cos(theta_hat)
    y_hat = r_hat * sin(theta_hat)

The reconstruction cells become annular sectors. A fine angular grid (large
n_theta) preserves edge orientation precisely while a coarse radial grid keeps
the rate low — decoupling "which way the edge points" from "how strong it is".

Log-polar companding
--------------------
With ``radial_mode="log"`` the radius is quantized in the log domain
(``round(log r / r_step)``) instead of linearly. Edge magnitudes are heavy-
tailed (a few very strong edges, many weak ones), so a *multiplicative* radial
grid — fine near the origin, coarse for large magnitudes — matches the source
statistics better than a uniform grid. This is classic companding applied to the
polar radius; the local radial bin width grows like ``r * r_step``.

Differentiability
-----------------
Every operation above is differentiable except the two `round`s. We replace
those with `round_ste` (identity gradient), so the encoder receives a gradient
that flows through the genuine, differentiable cos/sin/atan2/sqrt warp while the
piecewise-constant rounding is short-circuited.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ste import round_ste


class UniformScalarQuant(nn.Module):
    """Baseline: uniform scalar quantization with STE (the standard NIC quantizer).

    Quantizes every latent coordinate independently to a grid of width `step`.
    This is the ablation control that PolarQuant is compared against.
    """

    def __init__(self, step: float = 1.0, learnable_step: bool = False):
        super().__init__()
        log_step = torch.log(torch.tensor(float(step)))
        if learnable_step:
            self.log_step = nn.Parameter(log_step)
        else:
            self.register_buffer("log_step", log_step)

    @property
    def step(self) -> torch.Tensor:
        return torch.exp(self.log_step)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.step
        return round_ste(z / s) * s

    def symbols(self, z: torch.Tensor) -> torch.Tensor:
        """Integer quantization indices (for entropy/rate estimation)."""
        return torch.round(z / self.step)


class PolarQuant(nn.Module):
    """Polar-coordinate quantization layer with straight-through gradients.

    Args:
        r_step:  radial grid width. In ``linear`` mode this is the step in r; in
                 ``log`` mode it is the step in log(r) (a multiplicative ratio).
        n_theta: number of angular bins over [-pi, pi).
        radial_mode: "linear" (uniform radial grid) or "log" (log-polar companding).
        learnable: if True, r_step is a learnable parameter (log-parameterized
                   to stay positive); n_theta stays fixed (it is an integer grid).
        eps:     numerical floor to keep gradients of sqrt/atan2/log well-behaved.

    Input/Output: tensors of shape (B, C, H, W). Channels are consumed in
    interleaved (x, y) pairs. If C is odd the final channel is quantized with a
    plain scalar STE round at radial resolution `r_step`.
    """

    def __init__(
        self,
        r_step: float = 1.0,
        n_theta: int = 16,
        radial_mode: str = "linear",
        learnable: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if radial_mode not in ("linear", "log"):
            raise ValueError(f"radial_mode must be 'linear' or 'log', got {radial_mode}")
        self.n_theta = int(n_theta)
        self.radial_mode = radial_mode
        self.eps = eps
        log_r = torch.log(torch.tensor(float(r_step)))
        if learnable:
            self.log_r_step = nn.Parameter(log_r)
        else:
            self.register_buffer("log_r_step", log_r)

    @property
    def r_step(self) -> torch.Tensor:
        return torch.exp(self.log_r_step)

    @property
    def theta_step(self) -> float:
        return (2.0 * torch.pi) / self.n_theta

    # -- core polar transform & quantization -------------------------------
    def _to_polar(self, x: torch.Tensor, y: torch.Tensor):
        r = torch.sqrt(x * x + y * y + self.eps)
        theta = torch.atan2(y, x)
        return r, theta

    def _quantize_radius(self, r: torch.Tensor) -> torch.Tensor:
        """STE-quantize the radius, linearly or in the log domain."""
        if self.radial_mode == "log":
            rho = torch.log(r.clamp_min(self.eps))
            rho_q = round_ste(rho / self.r_step) * self.r_step
            # clamp before exp() to avoid overflow if the latent magnitude grows
            return torch.exp(rho_q.clamp(-30.0, 30.0))
        return round_ste(r / self.r_step) * self.r_step

    def _quantize_angle(self, theta: torch.Tensor) -> torch.Tensor:
        return round_ste(theta / self.theta_step) * self.theta_step

    def polar_components(self, z: torch.Tensor):
        """Return differentiable (r, theta, r_q, theta_q) for the channel pairs.

        Computed in fp32 so it is safe under AMP/fp16: the log-polar exp() and the
        sqrt/atan2 warp overflow or lose precision in fp16 (fp16 max ≈ 65504, and
        exp(12) already exceeds it), which otherwise produces NaNs. Shapes:
        (B, pairs, H, W). Exposed so the entropy model scores the quantized symbols.
        """
        C = z.shape[1]
        pairs = C // 2
        x = z[:, 0 : 2 * pairs : 2].float()
        y = z[:, 1 : 2 * pairs : 2].float()
        r, theta = self._to_polar(x, y)
        return r, theta, self._quantize_radius(r), self._quantize_angle(theta)

    def radial_bin_width(self, r_q: torch.Tensor) -> torch.Tensor:
        """Local width (in r-space) of the radial quantization bin at r_q.

        Linear: constant r_step. Log: ~ r_q * r_step  (since dr = r * d(log r)).
        Used by the entropy model to convert a density into a bin probability.
        """
        if self.radial_mode == "log":
            return r_q * self.r_step
        return self.r_step * torch.ones_like(r_q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        C = z.shape[1]
        pairs = C // 2
        out = torch.empty_like(z)

        if pairs > 0:
            r, theta, r_q, theta_q = self.polar_components(z)
            # math done in fp32 (see polar_components); cast back to z's dtype.
            out[:, 0 : 2 * pairs : 2] = (r_q * torch.cos(theta_q)).to(z.dtype)
            out[:, 1 : 2 * pairs : 2] = (r_q * torch.sin(theta_q)).to(z.dtype)

        if C % 2 == 1:  # odd leftover channel -> scalar STE round
            last = z[:, -1:].float()
            out[:, -1:] = (round_ste(last / self.r_step) * self.r_step).to(z.dtype)

        return out

    @torch.no_grad()
    def symbols(self, z: torch.Tensor):
        """Return (r_index, theta_index) integer grids for rate estimation.

        These are the discrete symbols an entropy coder would compress. In log
        mode the radial index is round(log(r) / r_step).
        """
        C = z.shape[1]
        pairs = C // 2
        x = z[:, 0 : 2 * pairs : 2]
        y = z[:, 1 : 2 * pairs : 2]
        r, theta = self._to_polar(x, y)
        if self.radial_mode == "log":
            r_idx = torch.round(torch.log(r.clamp_min(self.eps)) / self.r_step)
        else:
            r_idx = torch.round(r / self.r_step)
        theta_idx = torch.round(theta / self.theta_step)
        return r_idx, theta_idx

    def extra_repr(self) -> str:
        return (
            f"r_step={float(self.r_step):.4g}, n_theta={self.n_theta}, "
            f"radial_mode={self.radial_mode}"
        )
