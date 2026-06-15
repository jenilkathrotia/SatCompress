"""Complex-valued autoencoder — the "phase is everything" framing.

Theoretical motivation
----------------------
A PolarQuant latent pair (x, y) is literally the real and imaginary parts of a
complex number z = x + iy, whose magnitude is r = |z| and phase is θ = ∠z. There
is a classic result — Oppenheim & Lim, "The Importance of Phase in Signals"
(Proc. IEEE, 1981) — that the **phase** spectrum carries the structural/edge
information of an image, while the magnitude carries comparatively little.
Satellite imagery is edge-dominated (coastlines, field boundaries, roads), so the
right thing to do is **spend bits on phase and save bits on magnitude**.

This module makes the latent *natively complex*:
  * `ComplexConv2d` / `ComplexConvTranspose2d` — complex convolutions implemented
    as two real convs (Gauss form): (Wr + iWi)(a + ib).
  * `modReLU` — a phase-preserving nonlinearity that rectifies magnitude only,
    leaving the phase untouched (exactly the inductive bias we want).
  * `ComplexPolarQuant` — quantize |z| and ∠z directly on the complex latent;
    interface-compatible with `PolarRateModel`.
  * `ComplexCompressionAutoencoder` — real image → complex latent → real image.

Spend-bits-on-phase is then controlled exactly as in the real model: a fine
angular grid (`n_theta` large) + a coarse radial grid (`r_step` large), plus the
optional Fourier phase loss in `satcompress.losses`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..quant.ste import round_ste


# ---------------------------------------------------------------------------
# Complex layers
# ---------------------------------------------------------------------------
class ComplexConv2d(nn.Module):
    """(Wr + iWi) * (a + ib) = (Wr*a − Wi*b) + i(Wr*b + Wi*a)."""

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv_r = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a, b = z.real, z.imag
        out_r = self.conv_r(a) - self.conv_i(b)
        out_i = self.conv_r(b) + self.conv_i(a)
        return torch.complex(out_r, out_i)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1):
        super().__init__()
        self.conv_r = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        self.conv_i = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a, b = z.real, z.imag
        out_r = self.conv_r(a) - self.conv_i(b)
        out_i = self.conv_r(b) + self.conv_i(a)
        return torch.complex(out_r, out_i)


class ModReLU(nn.Module):
    """Phase-preserving activation: relu(|z| + b) applied to the magnitude only.

    modReLU(z) = relu(|z| + b) * z / |z|.  The phase ∠z is left untouched — the
    inductive bias that matches "phase carries the structure".
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z.real * z.real + z.imag * z.imag + self.eps)
        scale = torch.relu(mag + self.bias.view(1, -1, 1, 1)) / mag
        return torch.complex(z.real * scale, z.imag * scale)


# ---------------------------------------------------------------------------
# Complex polar quantizer (interface-compatible with PolarRateModel)
# ---------------------------------------------------------------------------
class ComplexPolarQuant(nn.Module):
    """PolarQuant on a *complex* latent: quantize |z| and ∠z, rebuild z_hat.

    Each complex channel is one (x, y) pair, so `n_pairs == latent channels`.
    """

    def __init__(self, r_step=1.0, n_theta=16, radial_mode="linear", learnable=False, eps=1e-6):
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
    def r_step(self):
        return torch.exp(self.log_r_step)

    @property
    def theta_step(self) -> float:
        return (2.0 * torch.pi) / self.n_theta

    def _quantize_radius(self, r):
        if self.radial_mode == "log":
            rho_q = round_ste(torch.log(r.clamp_min(self.eps)) / self.r_step) * self.r_step
            return torch.exp(rho_q.clamp(-30.0, 30.0))
        return round_ste(r / self.r_step) * self.r_step

    def _quantize_angle(self, theta):
        return round_ste(theta / self.theta_step) * self.theta_step

    def polar_components(self, z: torch.Tensor):
        """Return (r, theta, r_q, theta_q); accepts complex z or interleaved real."""
        if torch.is_complex(z):
            a, b = z.real, z.imag
        else:  # interleaved real fallback
            a, b = z[:, 0::2], z[:, 1::2]
        r = torch.sqrt(a * a + b * b + self.eps)
        theta = torch.atan2(b, a)
        return r, theta, self._quantize_radius(r), self._quantize_angle(theta)

    def radial_bin_width(self, r_q):
        if self.radial_mode == "log":
            return r_q * self.r_step
        return self.r_step * torch.ones_like(r_q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, _, r_q, theta_q = self.polar_components(z)
        return torch.complex(r_q * torch.cos(theta_q), r_q * torch.sin(theta_q))

    @torch.no_grad()
    def symbols(self, z: torch.Tensor):
        a, b = (z.real, z.imag) if torch.is_complex(z) else (z[:, 0::2], z[:, 1::2])
        r = torch.sqrt(a * a + b * b + self.eps)
        theta = torch.atan2(b, a)
        if self.radial_mode == "log":
            r_idx = torch.round(torch.log(r.clamp_min(self.eps)) / self.r_step)
        else:
            r_idx = torch.round(r / self.r_step)
        return r_idx, torch.round(theta / self.theta_step)


# ---------------------------------------------------------------------------
# Complex autoencoder
# ---------------------------------------------------------------------------
class ComplexEncoder(nn.Module):
    """Real image -> complex latent. Input is lifted to complex (imag = 0)."""

    def __init__(self, in_channels=3, hidden=64, latent=96, num_down=4):
        super().__init__()
        blocks = []
        c_in = in_channels
        for i in range(num_down):
            c_out = hidden if i < num_down - 1 else latent
            blocks.append(ComplexConv2d(c_in, c_out, 5, stride=2, padding=2))
            if i < num_down - 1:
                blocks.append(ModReLU(c_out))
            c_in = c_out
        self.net = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x, torch.zeros_like(x))
        for layer in self.net:
            z = layer(z)
        return z


class ComplexDecoder(nn.Module):
    """Complex latent -> real image. Final real projection + sigmoid."""

    def __init__(self, out_channels=3, hidden=64, latent=96, num_up=4):
        super().__init__()
        blocks = []
        c_in = latent
        for i in range(num_up):
            c_out = hidden if i < num_up - 1 else out_channels
            blocks.append(ComplexConvTranspose2d(c_in, c_out, 5, 2, 2, 1))
            if i < num_up - 1:
                blocks.append(ModReLU(c_out))
            c_in = c_out
        self.net = nn.ModuleList(blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            z = layer(z)
        return torch.sigmoid(z.real)  # project complex -> real image in [0, 1]


class ComplexCompressionAutoencoder(nn.Module):
    """End-to-end complex compression model: real x -> complex z -> real x_hat."""

    def __init__(self, in_channels=3, hidden=64, latent=96, num_down=4, quantizer=None):
        super().__init__()
        self.encoder = ComplexEncoder(in_channels, hidden, latent, num_down)
        self.decoder = ComplexDecoder(in_channels, hidden, latent, num_down)
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        z_hat = self.quantizer(z) if self.quantizer is not None else z
        x_hat = self.decoder(z_hat)[..., : x.shape[-2], : x.shape[-1]]
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat}
