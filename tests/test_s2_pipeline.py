"""Tests for the raw Sentinel-2 data pipeline (no network required)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# make scripts/ importable for the SCL helper
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# --------------------------------------------------------------------------
# SCL bad-pixel-fraction helper (pure function)
# --------------------------------------------------------------------------
def test_scl_bad_fraction():
    from download_s2_tiles import scl_bad_fraction

    # classes: 4=veg, 5=bare (good); 9=cloud-high, 3=cloud-shadow, 0=nodata (bad)
    good = np.array([[4, 5], [4, 5]])
    assert scl_bad_fraction(good) == 0.0

    half = np.array([[4, 9], [5, 3]])  # 2 of 4 bad
    assert scl_bad_fraction(half) == pytest.approx(0.5)

    allbad = np.array([[0, 9], [10, 8]])
    assert scl_bad_fraction(allbad) == 1.0

    assert scl_bad_fraction(np.array([])) == 1.0  # empty -> treat as unusable


# --------------------------------------------------------------------------
# Dataset reads a 4-band GeoTIFF tile correctly
# --------------------------------------------------------------------------
def test_dataset_reads_4band_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from satcompress.data import Sentinel2PatchDataset

    ps, bands = 64, 4
    # write a synthetic 4-band uint16 reflectance tile (values up to ~10000)
    arr = (np.random.rand(bands, ps, ps) * 10000).astype(np.uint16)
    path = tmp_path / "scene_0_0.tif"
    profile = dict(
        driver="GTiff", height=ps, width=ps, count=bands, dtype="uint16",
        crs="EPSG:32633", transform=from_origin(0, 0, 10, 10),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)

    ds = Sentinel2PatchDataset(
        tmp_path, patch_size=ps, reflectance_scale=10000.0, patches_per_scene=1
    )
    assert len(ds) == 1  # patches_per_scene=1 -> one tile per file
    t = ds[0]
    assert tuple(t.shape) == (bands, ps, ps)
    assert t.dtype.is_floating_point
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0  # normalized to [0,1]


def test_patches_per_scene_multiplier(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from satcompress.data import Sentinel2PatchDataset

    arr = (np.random.rand(4, 128, 128) * 10000).astype(np.uint16)
    profile = dict(driver="GTiff", height=128, width=128, count=4, dtype="uint16",
                   crs="EPSG:32633", transform=from_origin(0, 0, 10, 10))
    with rasterio.open(tmp_path / "s.tif", "w", **profile) as dst:
        dst.write(arr)

    ds = Sentinel2PatchDataset(tmp_path, patch_size=64, reflectance_scale=10000.0,
                               patches_per_scene=8)
    assert len(ds) == 8  # one file x 8 crops
    assert tuple(ds[0].shape) == (4, 64, 64)
