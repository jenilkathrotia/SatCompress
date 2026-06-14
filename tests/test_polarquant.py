"""Tests for the STE and PolarQuant layers — the research core.

Run with: pytest -q
These tests verify the two properties that make the layer trainable:
  1. FORWARD is a true (hard) quantizer — outputs land on the polar grid.
  2. BACKWARD passes a non-zero, well-formed gradient to the encoder (STE).
"""

import math

import pytest

torch = pytest.importorskip("torch")

from satcompress.quant import (  # noqa: E402
    PolarQuant,
    PolarQuantSTE,
    RoundSTE,
    UniformScalarQuant,
    round_ste,
)


def test_round_ste_forward_is_hard_round():
    z = torch.tensor([-1.6, -0.4, 0.4, 1.6])
    assert torch.allclose(round_ste(z), torch.round(z))


def test_round_ste_backward_is_identity():
    z = torch.tensor([0.3, 0.7, 1.2], requires_grad=True)
    round_ste(z).sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))  # gradient flows straight through


def test_custom_function_round_ste_matches():
    z = torch.tensor([0.2, 0.9, -0.6], requires_grad=True)
    out = RoundSTE.apply(z)
    assert torch.allclose(out, torch.round(z))
    out.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_polarquant_outputs_on_grid():
    """After PolarQuant, each (x,y) pair's radius is a multiple of r_step and its
    angle is a multiple of theta_step (within float tolerance)."""
    q = PolarQuant(r_step=0.5, n_theta=8)
    z = torch.randn(2, 4, 8, 8)
    zq = q(z)
    x = zq[:, 0:4:2]
    y = zq[:, 1:4:2]
    r = torch.sqrt(x * x + y * y)
    theta = torch.atan2(y, x)
    # radius on the 0.5 grid
    assert torch.allclose(r, torch.round(r / 0.5) * 0.5, atol=1e-4)
    # angle on the 2pi/8 grid
    tstep = 2 * math.pi / 8
    snapped = torch.round(theta / tstep) * tstep
    assert torch.allclose(theta, snapped, atol=1e-4)


def test_polarquant_gradient_flows_to_input():
    q = PolarQuant(r_step=1.0, n_theta=16)
    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    q(z).pow(2).sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0  # STE delivered a real, non-zero gradient


def test_polarquant_handles_odd_channels():
    q = PolarQuant(r_step=1.0, n_theta=4)
    z = torch.randn(1, 3, 4, 4)  # odd channel count
    zq = q(z)
    assert zq.shape == z.shape
    # the leftover channel is scalar-rounded
    assert torch.allclose(zq[:, -1:], torch.round(z[:, -1:]))


def test_uniform_scalar_quant_grid():
    q = UniformScalarQuant(step=0.25)
    z = torch.randn(2, 3, 5, 5)
    zq = q(z)
    assert torch.allclose(zq, torch.round(z / 0.25) * 0.25)


def test_polarquant_ste_function_backward_identity():
    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    out = PolarQuantSTE.apply(z, 1.0, 16)
    out.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))  # global STE
