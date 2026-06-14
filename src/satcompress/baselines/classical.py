"""Classical compression baselines: JPEG and JPEG 2000.

Measures the rate-distortion (bpp vs PSNR/SSIM) operating points of the classical
codecs the neural model must beat. Each function encodes an image, measures the
bitstream size, decodes, and computes distortion against the original.

Fairness: these compress **all** channels of the input (e.g. all 4 Sentinel-2
bands), and bpp is reported over H×W counting every band — directly comparable to
the neural model's bpp, which also covers all bands. JPEG (a 3-channel DCT codec)
handles >3 bands by encoding the first three as RGB and each remaining band as a
grayscale stream; JPEG 2000 (multi-component) encodes all bands together.

Both codecs use Pillow (OpenJPEG built in) — no extra dependency, and JPEG 2000
rate control via `quality_mode="rates"` is monotonic (unlike the previous
imagecodecs `level` usage, which produced a broken RD curve).
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from ..metrics import bpp, psnr, ssim

_J2K_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [0,1] -> (H, W, C) uint8."""
    arr = (img.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return np.transpose(arr, (1, 2, 0))


def _from_uint8(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 -> (C, H, W) float in [0,1]. Copy keeps the tensor writable."""
    t = torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True)).float() / 255.0
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    return t.permute(2, 0, 1)


def _encode_jpeg_stream(arr2d_or_3d: np.ndarray, mode: str, quality: int):
    """Encode one HxW (L) or HxWx3 (RGB) array as JPEG; return (bits, decoded)."""
    buf = io.BytesIO()
    Image.fromarray(arr2d_or_3d, mode).save(buf, format="JPEG", quality=int(quality))
    bits = buf.tell() * 8
    dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert(mode))
    return bits, dec


def benchmark_jpeg(img: torch.Tensor, quality: int = 75) -> dict:
    """JPEG at the given quality (1-95), over all C bands. bpp counts every band."""
    c, h, w = img.shape
    arr = _to_uint8(img)  # (H, W, C)
    rec = np.zeros_like(arr)
    total_bits = 0

    head = 3 if c >= 3 else 1  # first 3 bands as RGB (or 1 as grayscale)
    if head == 3:
        bits, dec = _encode_jpeg_stream(arr[..., :3], "RGB", quality)
        rec[..., :3] = dec
    else:
        bits, dec = _encode_jpeg_stream(arr[..., 0], "L", quality)
        rec[..., 0] = dec
    total_bits += bits
    for k in range(head, c):  # remaining bands as grayscale streams
        bits, dec = _encode_jpeg_stream(arr[..., k], "L", quality)
        rec[..., k] = dec
        total_bits += bits

    x = img.unsqueeze(0)
    y = _from_uint8(rec).unsqueeze(0)
    return {
        "codec": "jpeg", "quality": int(quality),
        "bpp": bpp(total_bits, (h, w)),
        "psnr": float(psnr(x, y)), "ssim": float(ssim(x, y)),
    }


def benchmark_jpeg2000(img: torch.Tensor, compression_ratio: float = 20.0) -> dict:
    """JPEG 2000 at a target compression ratio (uncompressed:compressed).

    Higher ratio = smaller file. Multi-component: all C bands in one stream.
    """
    c, h, w = img.shape
    arr = _to_uint8(img)  # (H, W, C)
    mode = _J2K_MODE.get(c)

    if mode is not None:
        src = arr if c > 1 else arr[..., 0]
        buf = io.BytesIO()
        Image.fromarray(src, mode).save(
            buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(compression_ratio)]
        )
        total_bits = buf.tell() * 8
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert(mode))
        rec = dec if c > 1 else dec[..., None]
    else:  # uncommon band count -> per-band grayscale JPEG2000
        rec = np.zeros_like(arr)
        total_bits = 0
        for k in range(c):
            buf = io.BytesIO()
            Image.fromarray(arr[..., k], "L").save(
                buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(compression_ratio)]
            )
            total_bits += buf.tell() * 8
            rec[..., k] = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"))

    x = img.unsqueeze(0)
    y = _from_uint8(rec).unsqueeze(0)
    return {
        "codec": "jpeg2000", "compression_ratio": float(compression_ratio),
        "bpp": bpp(total_bits, (h, w)),
        "psnr": float(psnr(x, y)), "ssim": float(ssim(x, y)),
    }


def rate_distortion_sweep(
    images: list[torch.Tensor],
    jpeg_qualities=(10, 25, 50, 75, 90),
    jp2k_ratios=(80, 40, 20, 10, 5),  # high->low compression (low->high bpp)
) -> list[dict]:
    """Average bpp/PSNR/SSIM over images for several operating points (one row each)."""
    records: list[dict] = []
    for q in jpeg_qualities:
        rows = [benchmark_jpeg(im, q) for im in images]
        records.append(_mean_record(rows, {"codec": "jpeg", "setting": q}))
    for r in jp2k_ratios:
        try:
            rows = [benchmark_jpeg2000(im, r) for im in images]
            records.append(_mean_record(rows, {"codec": "jpeg2000", "setting": r}))
        except Exception as e:  # Pillow built without OpenJPEG
            records.append({"codec": "jpeg2000", "setting": r, "error": str(e)})
    return records


def _mean_record(rows: list[dict], meta: dict) -> dict:
    out = dict(meta)
    for k in ("bpp", "psnr", "ssim"):
        out[k] = float(np.mean([row[k] for row in rows]))
    return out
