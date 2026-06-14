"""Tests for the three research extensions:
  1. log-polar companding
  2. Rayleigh-von Mises matched entropy model
  3. complex-valued network + Fourier phase loss
"""

import math

import pytest

torch = pytest.importorskip("torch")

from satcompress.losses import FourierPhaseLoss, RateDistortionLoss  # noqa: E402
from satcompress.models import ComplexCompressionAutoencoder, ComplexPolarQuant, ModReLU  # noqa: E402
from satcompress.quant import PolarQuant, PolarRateModel  # noqa: E402


# --------------------------------------------------------------------------
# 1. Log-polar companding
# --------------------------------------------------------------------------
def test_logpolar_radius_on_log_grid():
    q = PolarQuant(r_step=0.5, n_theta=8, radial_mode="log")
    z = torch.randn(2, 4, 8, 8)
    zq = q(z)
    x, y = zq[:, 0:4:2], zq[:, 1:4:2]
    r = torch.sqrt(x * x + y * y)
    log_r = torch.log(r.clamp_min(1e-8))
    # log(r) must lie on the 0.5 grid
    assert torch.allclose(log_r, torch.round(log_r / 0.5) * 0.5, atol=1e-3)


def test_logpolar_gradient_flows():
    q = PolarQuant(r_step=0.5, n_theta=8, radial_mode="log")
    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    q(z).pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_logpolar_bin_width_scales_with_radius():
    q = PolarQuant(r_step=0.5, n_theta=8, radial_mode="log")
    r_q = torch.tensor([[[[1.0, 4.0]]]])
    w = q.radial_bin_width(r_q)
    assert torch.allclose(w, r_q * 0.5)  # multiplicative width


# --------------------------------------------------------------------------
# 2. Rayleigh-von Mises entropy model
# --------------------------------------------------------------------------
def test_rate_model_returns_positive_finite_bits():
    q = PolarQuant(r_step=1.0, n_theta=16)
    rm = PolarRateModel(n_pairs=2, n_theta=16)  # latent=4 -> 2 pairs
    z = torch.randn(2, 4, 8, 8)
    bits = rm.rate_bits(q, z)
    assert bits.ndim == 0 and torch.isfinite(bits) and bits.item() > 0


def test_rate_model_is_differentiable_in_latent_and_params():
    q = PolarQuant(r_step=1.0, n_theta=16)
    rm = PolarRateModel(n_pairs=2, n_theta=16)
    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    bits = rm.rate_bits(q, z)
    bits.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert rm.log_sigma.grad is not None and torch.isfinite(rm.log_sigma.grad).all()
    assert rm.raw_kappa.grad is not None
    assert rm.mu.grad is not None


def test_angle_bits_lower_for_concentrated_distribution():
    """A symbol at the von Mises mean should cost fewer bits when kappa is high."""
    rm = PolarRateModel(n_pairs=1, n_theta=16)
    grid = torch.arange(16) * (2 * math.pi / 16)
    theta_at_mean = torch.zeros(1, 1, 1, 1)  # mu initialized to 0
    with torch.no_grad():
        rm.raw_kappa.fill_(5.0)  # high concentration
        bits_high = rm.angle_bits(theta_at_mean, grid).item()
        rm.raw_kappa.fill_(-5.0)  # ~uniform
        bits_low = rm.angle_bits(theta_at_mean, grid).item()
    assert bits_high < bits_low  # peak symbol is cheaper under a concentrated prior


def test_rate_model_works_with_log_polar():
    q = PolarQuant(r_step=0.5, n_theta=8, radial_mode="log")
    rm = PolarRateModel(n_pairs=2, n_theta=8)
    z = torch.randn(1, 4, 4, 4)
    assert torch.isfinite(rm.rate_bits(q, z))


# --------------------------------------------------------------------------
# 3. Complex network + phase loss
# --------------------------------------------------------------------------
def test_modrelu_preserves_phase():
    act = ModReLU(channels=3)
    with torch.no_grad():
        act.bias.fill_(0.5)  # positive bias keeps everything active
    z = torch.complex(torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))
    out = act(z)
    # where magnitude survives, the phase (direction) must be unchanged
    same = torch.cos(torch.angle(z) - torch.angle(out))
    assert torch.allclose(same, torch.ones_like(same), atol=1e-4)


def test_complex_autoencoder_forward_and_backward():
    q = ComplexPolarQuant(r_step=1.0, n_theta=16)
    model = ComplexCompressionAutoencoder(in_channels=3, latent=16, quantizer=q)
    x = torch.rand(1, 3, 32, 32)
    out = model(x)
    assert out["x_hat"].shape == x.shape
    assert torch.is_complex(out["z"]) and torch.is_complex(out["z_hat"])
    out["x_hat"].mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_complex_quant_compatible_with_rate_model():
    q = ComplexPolarQuant(r_step=1.0, n_theta=16)
    rm = PolarRateModel(n_pairs=8, n_theta=16)  # 8 complex channels = 8 pairs
    z = torch.complex(torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4))
    assert torch.isfinite(rm.rate_bits(q, z))


def test_fourier_phase_loss_zero_when_identical_and_positive_otherwise():
    loss = FourierPhaseLoss()
    x = torch.rand(2, 3, 16, 16)
    assert loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-5)
    assert loss(x, torch.rand(2, 3, 16, 16)).item() > 0


def test_phase_loss_integrates_into_rate_distortion():
    crit = RateDistortionLoss(lambda_rate=0.0, phase_weight=0.5)
    x = torch.rand(1, 3, 16, 16)
    x_hat = torch.rand(1, 3, 16, 16, requires_grad=True)
    out = crit(x, x_hat)
    assert "phase" in out and out["phase"].item() > 0
    out["loss"].backward()
    assert x_hat.grad is not None
