"""Convolutional autoencoder for learned image compression.

Architecture follows the now-canonical Ballé-style analysis/synthesis transform:
a stack of strided convolutions with GDN-like nonlinearities down to a compact
latent z, a pluggable quantizer, then a mirrored transposed-conv synthesis.

The quantizer is injected (dependency injection) so the *same* backbone can be
trained with:
  * no quantization        (Phase 2: prove the AE reconstructs cleanly)
  * UniformScalarQuant     (Phase 3 ablation control)
  * PolarQuant             (Phase 3 the proposed method)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GDN(nn.Module):
    """Generalized Divisive Normalization (Ballé et al. 2016), a normalization
    that is well matched to image statistics and outperforms BatchNorm/ReLU for
    compression transforms. Simplified, numerically-stable form."""

    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.eye(channels) * 0.1)
        self.beta_min = beta_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.beta, min=self.beta_min)
        gamma = torch.clamp(self.gamma, min=0.0)
        norm = torch.einsum("bchw,cd->bdhw", x * x, gamma) + beta[None, :, None, None]
        norm = torch.sqrt(norm)
        return x * norm if self.inverse else x / norm


class Encoder(nn.Module):
    """Analysis transform x -> z. Downsamples by 2^num_down (default 16x)."""

    def __init__(self, in_channels: int = 3, hidden: int = 128, latent: int = 192, num_down: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_down):
            c_out = hidden if i < num_down - 1 else latent
            layers += [nn.Conv2d(c_in, c_out, kernel_size=5, stride=2, padding=2)]
            if i < num_down - 1:
                layers += [GDN(c_out)]
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Synthesis transform z_hat -> x_hat. Mirrors the encoder."""

    def __init__(self, out_channels: int = 3, hidden: int = 128, latent: int = 192, num_up: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = latent
        for i in range(num_up):
            c_out = hidden if i < num_up - 1 else out_channels
            layers += [nn.ConvTranspose2d(c_in, c_out, kernel_size=5, stride=2, padding=2, output_padding=1)]
            if i < num_up - 1:
                layers += [GDN(c_out, inverse=True)]
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))  # outputs in [0, 1]


class CompressionAutoencoder(nn.Module):
    """End-to-end model: encode -> quantize -> decode.

    Args:
        quantizer: an nn.Module mapping z -> z_hat (e.g. PolarQuant). If None,
                   the latent is passed through unquantized (Phase 2 sanity).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 128,
        latent: int = 192,
        num_down: int = 4,
        quantizer: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, latent, num_down)
        self.decoder = Decoder(in_channels, hidden, latent, num_down)
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        z_hat = self.quantizer(z) if self.quantizer is not None else z
        x_hat = self.decoder(z_hat)
        # crop/pad to input size in case of odd dims from strided convs
        x_hat = x_hat[..., : x.shape[-2], : x.shape[-1]]
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat}
