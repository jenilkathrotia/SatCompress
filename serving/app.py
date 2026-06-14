"""Phase 5: Streamlit live demo.

Upload a satellite image, compress it with SatCompress, and view the original vs
reconstruction side by side with PSNR / SSIM / bpp.

Run:
    streamlit run serving/app.py
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from satcompress.metrics import estimate_entropy_bits, psnr, ssim
from satcompress.models import CompressionAutoencoder
from satcompress.quant import PolarQuant


@st.cache_resource
def load_model():
    quant = PolarQuant(r_step=1.0, n_theta=16)
    model = CompressionAutoencoder(in_channels=3, latent=192, quantizer=quant)
    try:
        state = torch.load("checkpoints/satcompress_polar.pt", map_location="cpu")
        model.load_state_dict(state["model"])
    except FileNotFoundError:
        st.warning("No trained checkpoint found — showing untrained reconstruction.")
    return model.eval()


def run(model, img: Image.Image):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    _, _, h, w = x.shape
    ph, pw = (-h) % 16, (-w) % 16
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        out = model(xp)
        x_hat = out["x_hat"][..., :h, :w]
        r_idx, th_idx = model.quantizer.symbols(out["z"])
        bits = estimate_entropy_bits(r_idx) + estimate_entropy_bits(th_idx)
        m = {
            "PSNR (dB)": round(float(psnr(x, x_hat)), 2),
            "SSIM": round(float(ssim(x, x_hat)), 4),
            "bpp": round(bits / (h * w), 4),
        }
    rec = (x_hat.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(rec), m


st.set_page_config(page_title="SatCompress", layout="wide")
st.title("🛰️ SatCompress — Neural Satellite Image Compression")
st.caption("PolarQuant latent quantization · upload an image to compress it live")

model = load_model()
uploaded = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    rec, metrics = run(model, img)
    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(img, use_container_width=True)
    c2.subheader("Reconstruction")
    c2.image(rec, use_container_width=True)
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)
