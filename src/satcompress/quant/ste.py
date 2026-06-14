"""Straight-Through Estimators for non-differentiable quantization.

The rounding operator round(.) is piecewise-constant, so dround/dz = 0 almost
everywhere and exactly undefined at half-integers. Training an autoencoder
end-to-end through such an operator therefore kills the gradient to the encoder.

The Straight-Through Estimator (STE) resolves this: in the FORWARD pass we apply
the true (hard) quantizer; in the BACKWARD pass we substitute the identity
Jacobian, i.e. d(round(z))/dz := 1. This is the now-standard relaxation from
Bengio et al. (2013) and Theis et al. (2017) "Lossy Image Compression with
Compressive Autoencoders".

Two implementations are provided:

1. `round_ste` — the idiomatic one-liner using the detach trick. Preferred,
   because it lets autograd flow through any *differentiable* ops wrapped
   around the round (e.g. the Cartesian<->polar warp) while only short-
   circuiting the round itself.

2. `RoundSTE` / `PolarQuantSTE` — explicit `torch.autograd.Function`s, written
   out for clarity and for the portfolio requirement of a hand-rolled custom
   forward/backward. `PolarQuantSTE` treats the entire warp+quantize block as
   identity in the backward pass (global STE).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. Idiomatic STE round (detach trick)
# ---------------------------------------------------------------------------
def round_ste(z: torch.Tensor) -> torch.Tensor:
    """round(z) on the forward pass, identity gradient on the backward pass.

    forward:  z + (round(z) - z) = round(z)
    backward: d/dz [ z + (round(z) - z).detach() ] = 1
    """
    return z + (torch.round(z) - z).detach()


# ---------------------------------------------------------------------------
# 2a. Explicit custom Function for rounding (pedagogical / portfolio)
# ---------------------------------------------------------------------------
class RoundSTE(torch.autograd.Function):
    """Hand-written STE round: forward = round, backward = identity."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        return torch.round(z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through: pass the upstream gradient unchanged.
        return grad_output


# ---------------------------------------------------------------------------
# 2b. Whole-block PolarQuant Function (matches the project spec literally:
#     forward = Cartesian -> Polar -> Quantize -> Cartesian; backward = STE)
# ---------------------------------------------------------------------------
class PolarQuantSTE(torch.autograd.Function):
    """Polar-warp + quantize as a single op with a global straight-through grad.

    Channels are interpreted as interleaved (x, y) pairs; the last channel is
    passed through untouched if C is odd. Backward returns grad_output for the
    quantized pairs (identity), so the encoder sees gradients as if the whole
    block were the identity map.

    Use the modular `PolarQuant` layer (built on `round_ste`) for actual
    training; this Function exists to demonstrate an explicit STE backward.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, r_step: float, n_theta: int) -> torch.Tensor:
        B, C, H, W = z.shape
        pairs = C // 2
        out = z.clone()
        x = z[:, 0 : 2 * pairs : 2]
        y = z[:, 1 : 2 * pairs : 2]

        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(y, x)

        # Quantize radius to a uniform grid of width r_step; angle to n_theta bins.
        r_q = torch.round(r / r_step) * r_step
        theta_step = (2.0 * torch.pi) / n_theta
        theta_q = torch.round(theta / theta_step) * theta_step

        out[:, 0 : 2 * pairs : 2] = r_q * torch.cos(theta_q)
        out[:, 1 : 2 * pairs : 2] = r_q * torch.sin(theta_q)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Global STE: gradient flows straight through the warp+quantize block.
        # r_step and n_theta are non-tensor args -> return None for them.
        return grad_output, None, None
