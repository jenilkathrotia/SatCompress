"""Phase 5: FastAPI inference endpoint.

Exposes the trained model as a compression service:
  POST /compress  (multipart image) -> reconstructed image + PSNR/SSIM/bpp

The encoder+quantizer produce the latent symbols (the "compressed" payload); the
decoder reconstructs. For a true production decoder-only endpoint, ship the
latent symbols from the client and call /decode.

Run:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from satcompress.metrics import estimate_entropy_bits, psnr, ssim
from satcompress.models import CompressionAutoencoder
from satcompress.quant import PolarQuant

app = FastAPI(title="SatCompress", version="0.1.0")

_MODEL: CompressionAutoencoder | None = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> CompressionAutoencoder:
    global _MODEL
    if _MODEL is None:
        ckpt_path = os.environ.get("SATCOMPRESS_CKPT", "checkpoints/satcompress_polar.pt")
        quant = PolarQuant(r_step=1.0, n_theta=16)
        model = CompressionAutoencoder(in_channels=3, latent=192, quantizer=quant)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=_DEVICE)
            model.load_state_dict(state["model"])
            print(f"[api] loaded {ckpt_path}")
        else:
            print(f"[api] WARNING: {ckpt_path} not found; serving untrained weights")
        _MODEL = model.eval().to(_DEVICE)
    return _MODEL


def _load_image(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    # pad to multiple of 16 for the 16x downsampling transform
    _, _, h, w = t.shape
    ph, pw = (-h) % 16, (-w) % 16
    t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t.to(_DEVICE), (h, w)


def _to_png(t: torch.Tensor) -> bytes:
    arr = (t.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return {"status": "ok", "device": _DEVICE}


@app.post("/compress")
async def compress(file: UploadFile = File(...)):
    model = get_model()
    x, (h, w) = _load_image(await file.read())
    with torch.no_grad():
        out = model(x)
        x_hat = out["x_hat"][..., :h, :w]
        x_crop = x[..., :h, :w]
        r_idx, th_idx = model.quantizer.symbols(out["z"])
        bits = estimate_entropy_bits(r_idx) + estimate_entropy_bits(th_idx)
        metrics = {
            "psnr": round(float(psnr(x_crop, x_hat)), 3),
            "ssim": round(float(ssim(x_crop, x_hat)), 4),
            "bpp": round(bits / (h * w), 4),
        }
    return JSONResponse(metrics)


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    """Return the reconstructed PNG (for side-by-side display)."""
    model = get_model()
    x, (h, w) = _load_image(await file.read())
    with torch.no_grad():
        x_hat = model(x)["x_hat"][..., :h, :w]
    return Response(content=_to_png(x_hat), media_type="image/png")
