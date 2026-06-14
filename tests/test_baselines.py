"""Tests for the classical baselines (JPEG / JPEG2000), multi-band + rate control."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from satcompress.baselines import benchmark_jpeg, benchmark_jpeg2000  # noqa: E402


def _smooth_img(c=4, n=64):
    yy, xx = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    base = 0.5 + 0.5 * np.sin(8 * xx) * np.cos(6 * yy)
    stack = np.stack([(base * (1 + 0.1 * i)) for i in range(c)])
    return torch.from_numpy(np.clip(stack, 0, 1)).float()


def test_jpeg_4band_bpp_and_quality():
    r = benchmark_jpeg(_smooth_img(4), quality=75)
    assert r["bpp"] > 0
    assert np.isfinite(r["psnr"]) and 0.0 <= r["ssim"] <= 1.0


def test_jpeg_handles_3_and_1_band():
    assert benchmark_jpeg(_smooth_img(3), quality=50)["bpp"] > 0
    assert benchmark_jpeg(_smooth_img(1), quality=50)["bpp"] > 0


def test_jpeg2000_rate_is_monotonic():
    """The bug we fixed: higher compression ratio must yield lower bpp."""
    im = _smooth_img(4)
    bpp_low_ratio = benchmark_jpeg2000(im, compression_ratio=5)["bpp"]    # bigger file
    bpp_high_ratio = benchmark_jpeg2000(im, compression_ratio=80)["bpp"]  # smaller file
    assert bpp_low_ratio > bpp_high_ratio


def test_jpeg2000_quality_drops_with_compression():
    im = _smooth_img(4)
    psnr_light = benchmark_jpeg2000(im, compression_ratio=5)["psnr"]
    psnr_heavy = benchmark_jpeg2000(im, compression_ratio=80)["psnr"]
    assert psnr_light >= psnr_heavy  # more compression -> equal or worse fidelity


def test_jpeg2000_multiband_roundtrip_shape():
    # 4-band stays 4-band through encode/decode (PSNR computed over all bands)
    r = benchmark_jpeg2000(_smooth_img(4), compression_ratio=20)
    assert np.isfinite(r["psnr"]) and r["bpp"] > 0
