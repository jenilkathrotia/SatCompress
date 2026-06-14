"""Classical compression baselines: JPEG and JPEG 2000.

Phase 1 deliverable: measure the rate-distortion (bpp vs PSNR/SSIM) operating
points of the classical codecs that the neural model must beat. Each function
encodes an in-memory image at a given quality, measures the resulting bitstream
size, decodes, and computes distortion against the original.

These run on 8-bit RGB previews (3 channels). For full multispectral evaluation
extend `_to_uint8`/codec calls to per-band encoding.
"""

from __future__ import annotations

import io

import numpy as np
import torch

from ..metrics import bpp, psnr, ssim


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [0,1] -> (H, W, C) uint8."""
    arr = (img.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return np.transpose(arr, (1, 2, 0))


def _from_uint8(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 -> (C, H, W) float in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
    return t.permute(2, 0, 1)


def benchmark_jpeg(img: torch.Tensor, quality: int = 75) -> dict:
    """Encode/decode one image with JPEG at the given quality (1-95)."""
    from PIL import Image

    h, w = img.shape[-2:]
    pil = Image.fromarray(_to_uint8(img))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    num_bits = buf.tell() * 8
    dec = _from_uint8(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))

    x = img.unsqueeze(0)
    y = dec.unsqueeze(0)
    return {
        "codec": "jpeg",
        "quality": int(quality),
        "bpp": bpp(num_bits, (h, w)),
        "psnr": float(psnr(x, y)),
        "ssim": float(ssim(x, y)),
    }


def benchmark_jpeg2000(img: torch.Tensor, compression_ratio: float = 20.0) -> dict:
    """Encode/decode one image with JPEG 2000 at a target compression ratio.

    Uses `imagecodecs` (OpenJPEG). `compression_ratio` ~ uncompressed/compressed.
    """
    import imagecodecs

    h, w = img.shape[-2:]
    arr = _to_uint8(img)
    encoded = imagecodecs.jpeg2k_encode(arr, level=float(compression_ratio))
    num_bits = len(encoded) * 8
    dec = _from_uint8(imagecodecs.jpeg2k_decode(encoded))

    x = img.unsqueeze(0)
    y = dec.unsqueeze(0)
    return {
        "codec": "jpeg2000",
        "compression_ratio": float(compression_ratio),
        "bpp": bpp(num_bits, (h, w)),
        "psnr": float(psnr(x, y)),
        "ssim": float(ssim(x, y)),
    }


def rate_distortion_sweep(
    images: list[torch.Tensor],
    jpeg_qualities=(10, 25, 50, 75, 90),
    jp2k_ratios=(50, 30, 20, 10, 5),
) -> list[dict]:
    """Average bpp/PSNR/SSIM over a list of images for several operating points.

    Returns one record per (codec, setting), suitable for plotting an RD curve
    or dumping to results/baselines.csv.
    """
    records: list[dict] = []
    for q in jpeg_qualities:
        rows = [benchmark_jpeg(im, q) for im in images]
        records.append(_mean_record(rows, {"codec": "jpeg", "setting": q}))
    for r in jp2k_ratios:
        try:
            rows = [benchmark_jpeg2000(im, r) for im in images]
            records.append(_mean_record(rows, {"codec": "jpeg2000", "setting": r}))
        except Exception as e:  # imagecodecs/OpenJPEG not installed
            records.append({"codec": "jpeg2000", "setting": r, "error": str(e)})
    return records


def _mean_record(rows: list[dict], meta: dict) -> dict:
    keys = ("bpp", "psnr", "ssim")
    out = dict(meta)
    for k in keys:
        out[k] = float(np.mean([row[k] for row in rows]))
    return out
