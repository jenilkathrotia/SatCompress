"""Data pipeline for Sentinel-2 imagery.

Phase 1 deliverable: a robust PyTorch `Dataset`/`DataLoader` that
  * walks a directory of Sentinel-2 scenes (GeoTIFF / JP2 / PNG / common raster),
  * crops random 256x256 patches,
  * normalizes reflectance to [0, 1].

Sentinel-2 L2A surface reflectance is stored as uint16 with a nominal scale
factor of 10000. We expose `reflectance_scale` so RGB previews (8-bit PNG) and
true multispectral GeoTIFFs share one code path. `RandomPatchDataset` provides a
dependency-free synthetic source so the training loop and tests run before any
real data is downloaded.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_RASTER_EXT = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg"}


def _read_image(path: Path, bands: tuple[int, ...] | None) -> np.ndarray:
    """Read an image as float32 array of shape (C, H, W). Tries rasterio first
    (GeoTIFF/JP2 multispectral), falls back to PIL (8-bit RGB)."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".jp2"}:
        import rasterio  # local import: heavy GDAL dependency

        with rasterio.open(path) as src:
            idx = list(bands) if bands else list(range(1, src.count + 1))
            arr = src.read(idx).astype(np.float32)
        return arr
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


class Sentinel2PatchDataset(Dataset):
    """Random 256x256 patches drawn from a directory of raster scenes.

    Args:
        root: directory containing Sentinel-2 scenes (searched recursively).
        patch_size: square crop size in pixels.
        bands: 1-indexed band selection for multispectral rasters (None = all).
        reflectance_scale: divide raw values by this to reach [0, 1]
                           (10000 for S2 L2A reflectance; 255 for 8-bit PNG).
        patches_per_scene: virtual length multiplier — how many random crops to
                           expose per scene per epoch.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 256,
        bands: tuple[int, ...] | None = None,
        reflectance_scale: float = 10000.0,
        patches_per_scene: int = 16,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.patch_size = patch_size
        self.bands = bands
        self.reflectance_scale = reflectance_scale
        self.patches_per_scene = patches_per_scene
        self.files = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _RASTER_EXT
        )
        if not self.files:
            raise FileNotFoundError(
                f"No raster files {sorted(_RASTER_EXT)} found under {self.root}. "
                "Run scripts/download_sentinel.py or use RandomPatchDataset."
            )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files) * self.patches_per_scene

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx % len(self.files)]
        arr = _read_image(path, self.bands)  # (C, H, W)
        _, h, w = arr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            # pad small scenes by reflection
            arr = np.pad(
                arr,
                ((0, 0), (0, max(0, ps - h)), (0, max(0, ps - w))),
                mode="reflect",
            )
            _, h, w = arr.shape
        top = self._rng.randint(0, h - ps)
        left = self._rng.randint(0, w - ps)
        patch = arr[:, top : top + ps, left : left + ps]
        patch = np.clip(patch / self.reflectance_scale, 0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(patch)).float()


class RandomPatchDataset(Dataset):
    """Synthetic, dependency-free dataset of structured patches.

    Generates piecewise-smooth fields with sharp oriented edges (a crude proxy
    for coastlines/field boundaries) so the pipeline, training loop, and ablation
    harness are exercisable with zero external data. NOT for reporting results.
    """

    def __init__(self, length: int = 256, channels: int = 3, patch_size: int = 256, seed: int = 0):
        self.length = length
        self.channels = channels
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + idx)
        ps, c = self.patch_size, self.channels
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, ps), torch.linspace(0, 1, ps), indexing="ij"
        )
        out = []
        for _ in range(c):
            angle = torch.rand((), generator=g) * torch.pi
            freq = 2 + torch.rand((), generator=g) * 6
            edge = (torch.sin(2 * torch.pi * freq * (xx * torch.cos(angle) + yy * torch.sin(angle))) > 0).float()
            smooth = 0.5 + 0.5 * torch.sin(2 * torch.pi * (xx + yy))
            ch = 0.6 * edge + 0.4 * smooth
            out.append(ch)
        return torch.stack(out, 0).clamp(0, 1)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
