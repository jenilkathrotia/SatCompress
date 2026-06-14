# SatCompress — Neural Satellite-Image Compression with PolarQuant

> A learned image-compression system for multispectral satellite imagery whose
> central contribution is **PolarQuant**: a projective (polar-coordinate) warp of
> the latent space prior to quantization, trained end-to-end through a
> Straight-Through Estimator.

---

## Abstract

Learned image compression replaces hand-designed transforms (DCT, wavelet) with
a convolutional analysis/synthesis pair and a quantized latent bottleneck. The
quantizer is the crux: it is non-differentiable, and the shape of its
reconstruction cells determines what structure survives compression. Standard
neural compressors apply **uniform scalar quantization** on a Cartesian grid —
axis-aligned hypercube cells that treat every latent coordinate independently.

For satellite imagery, the dominant high-frequency content is *oriented* edges
(coastlines, field boundaries, roads). SatCompress introduces **PolarQuant**,
which pairs latent channels into `(x, y)` couples, maps them to polar
coordinates `(r, θ)`, and quantizes radius and angle on **separate grids**. The
resulting cells are annular sectors, decoupling *edge orientation* (θ) from
*edge magnitude* (r): a fine angular grid preserves orientation precisely while
a coarse radial grid keeps the bit-rate low. The entire warp is differentiable
except the rounding, which is handled by a Straight-Through Estimator so the
encoder trains end-to-end.

---

## Method

### 1. Architecture — latent-space convolutional autoencoder

```
x ──► Encoder E (analysis) ──► z ──► Quantizer Q ──► ẑ ──► Decoder D (synthesis) ──► x̂
       4× strided conv + GDN          PolarQuant            4× transposed conv + IGDN
       (16× downsample)                                     (16× upsample)
```

- **Encoder** `E`: maps `x ∈ ℝ^{C×H×W}` to a compact latent `z = E(x)`.
- **Quantizer** `Q`: `ẑ = Q(z)` — pluggable (`none` / `scalar` / `polar`).
- **Decoder** `D`: reconstructs `x̂ = D(ẑ)`.

Implemented in [src/satcompress/models/autoencoder.py](src/satcompress/models/autoencoder.py)
with GDN normalization (Ballé et al.), which is matched to natural-image
statistics and outperforms BatchNorm/ReLU for compression transforms.

### 2. PolarQuant + Straight-Through Estimator

Channels are consumed in interleaved `(x, y)` pairs:

$$ r = \sqrt{x^2 + y^2}, \qquad \theta = \operatorname{atan2}(y, x) $$

$$ \hat{r} = \Delta_r \cdot \mathrm{round}\!\left(\tfrac{r}{\Delta_r}\right), \qquad
   \hat{\theta} = \Delta_\theta \cdot \mathrm{round}\!\left(\tfrac{\theta}{\Delta_\theta}\right),\quad \Delta_\theta = \tfrac{2\pi}{n_\theta} $$

$$ \hat{x} = \hat{r}\cos\hat{\theta}, \qquad \hat{y} = \hat{r}\sin\hat{\theta} $$

**The discontinuity problem.** `round(·)` is piecewise-constant, so its
derivative is `0` almost everywhere and undefined at half-integers — the encoder
would receive no gradient. We apply the **Straight-Through Estimator**: the
forward pass quantizes; the backward pass substitutes the identity Jacobian,
`∂ẑ/∂z ≈ 1`. The idiomatic implementation (the *detach trick*) lets gradients
flow through the genuine, differentiable `cos/sin/atan2/sqrt` warp while only
the rounding is short-circuited:

```python
def round_ste(z):
    return z + (torch.round(z) - z).detach()   # forward: round(z); backward: 1
```

Both forms required by the project are provided in
[src/satcompress/quant/ste.py](src/satcompress/quant/ste.py): the `round_ste`
helper *and* a hand-written `torch.autograd.Function` (`RoundSTE`,
`PolarQuantSTE`) with explicit `forward`/`backward`. The layer itself is
[src/satcompress/quant/polarquant.py](src/satcompress/quant/polarquant.py);
`UniformScalarQuant` is the ablation control.

### 3. Objective — rate-distortion(-phase)

$$ \mathcal{L} = \underbrace{(1-w)\,\mathrm{MSE} + w\,(1 - \mathrm{SSIM})}_{\text{distortion}} \;+\; \lambda \cdot \underbrace{R(\hat{z})}_{\text{rate (bpp)}} \;+\; \gamma \cdot \underbrace{P(x,\hat{x})}_{\text{phase}} $$

Rate `R` is either the simple empirical Shannon entropy of the symbols (logging
only) or the **matched Rayleigh–von Mises model** below.
[src/satcompress/losses.py](src/satcompress/losses.py),
[src/satcompress/metrics.py](src/satcompress/metrics.py).

### 4. Research extensions

**4.1 Rayleigh–von Mises matched entropy model** —
[src/satcompress/quant/entropy.py](src/satcompress/quant/entropy.py).
If a latent pair `(x,y)` is isotropic Gaussian, then by change of variables its
magnitude `r` is **Rayleigh** and its angle `θ` is **circular (von Mises)**. So
the correct prior for a polar latent is `Rayleigh(r;σ) × vonMises(θ;μ,κ)` — not
the Gaussian everyone bolts on. The radius cost uses the Rayleigh log-density;
the angle cost uses the exact discrete von Mises mass via `logsumexp`. Parameters
`(σ, μ, κ)` are learned jointly, turning PolarQuant into a *complete* framework
(transform + quantizer + matched prior). Circular entropy models are unused in
image compression — a genuine gap.

**4.2 Phase-aware complex-valued network** —
[src/satcompress/models/complex_ae.py](src/satcompress/models/complex_ae.py).
A polar pair *is* the real/imag of a complex number. We provide a natively
complex encoder/decoder (`ComplexConv2d`), a phase-preserving activation
(`ModReLU`), and a `ComplexPolarQuant`. Motivated by Oppenheim & Lim (1981) —
phase carries an image's edges/structure — a `FourierPhaseLoss` lets the model
**spend bits on phase**.

**4.3 Log-polar companding** — `PolarQuant(radial_mode="log")` quantizes
`log(r)` instead of `r`, matching the heavy-tailed magnitude statistics of edges
with a multiplicative radial grid (fine near the origin, coarse far out).

---

## Repository layout

```
src/satcompress/
  quant/        PolarQuant (+ log-polar), UniformScalarQuant, STE,
                PolarRateModel (Rayleigh–von Mises entropy model)
  models/       Encoder/Decoder/CompressionAutoencoder (GDN backbone) +
                ComplexCompressionAutoencoder / ComplexPolarQuant / ModReLU
  data/         Sentinel-2 patch Dataset/DataLoader + synthetic source
  baselines/    JPEG / JPEG 2000 rate-distortion benchmarking
  metrics.py    PSNR, SSIM (differentiable), bpp, entropy estimate
  losses.py     RateDistortionLoss + FourierPhaseLoss
  train.py      training entry point (AMP + W&B, --device override)
scripts/        download_sentinel.py · run_baselines.py · ablation.py
serving/        api.py (FastAPI) · app.py (Streamlit live demo)
configs/        default.yaml (+ W&B sweep space)
tests/          test_polarquant.py + test_extensions.py  (20 tests)
Dockerfile      NVIDIA PyTorch base for Nebius H100
```

---

## Quickstart

> **Python version note.** PyTorch wheels lag the newest CPython. Use a 3.11/3.12
> virtualenv for reproducibility:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q                          # verify the STE + PolarQuant core
```

Smoke-test the full pipeline on synthetic data (no download needed):

```bash
python -m satcompress.train --quantizer polar --synthetic --epochs 2 --no-wandb
python scripts/ablation.py --epochs 3          # scalar vs polar, same backbone
python scripts/run_baselines.py --synthetic    # JPEG / JPEG 2000 RD points

# research extensions
python -m satcompress.train --quantizer polar --radial-mode log \
    --entropy-model rayleigh-vm --synthetic --no-wandb        # 4.1 + 4.3
python -m satcompress.train --complex --quantizer polar --phase-weight 0.1 \
    --entropy-model rayleigh-vm --synthetic --no-wandb        # 4.2
```

> On Apple Silicon, pass `--device cpu` for the research configs: MPS currently
> mishandles `atan2`/`logsumexp` and can produce NaNs (CUDA and CPU are fine).

Train on real Sentinel-2 data:

```bash
python scripts/download_sentinel.py --out data/sentinel     # EuroSAT (S2 RGB)
python -m satcompress.train --data-root data/sentinel --quantizer polar
```

Serve the trained model (Phase 5):

```bash
uvicorn serving.api:app --port 8000      # POST /compress, /reconstruct
streamlit run serving/app.py             # interactive side-by-side demo
```

---

## Roadmap (8-week research plan)

| Phase | Weeks | Deliverable | Status |
|------:|:-----:|-------------|:------:|
| 1 | 1–2 | Data pipeline + JPEG/JPEG2000 baselines (PSNR/SSIM/bpp) | ✅ scaffolded |
| 2 | 3–4 | Vanilla conv-autoencoder + W&B tracking | ✅ scaffolded |
| 3 | 5–6 | PolarQuant custom autograd layer + ablation vs scalar | ✅ scaffolded |
| 4 |  7  | Dockerize → Nebius H100, BF16/FP8, W&B sweeps | ✅ Dockerfile + sweep cfg |
| 5 |  8  | FastAPI + Streamlit demo + paper-style writeup | ✅ scaffolded |

"Scaffolded" = runnable code + smoke-tested on synthetic data; the research work
is filling in real Sentinel-2 training, tuning, and result tables.

### High-performance engineering (Phase 4)

- **Mixed precision** (`--amp bf16`) targets H100 Tensor Cores; FP8 path reserved
  for `transformer-engine` integration.
- **W&B Sweeps**: search `r_step`, `n_theta`, and `λ` for the best
  compression-ratio / fidelity trade-off (target **SSIM > 0.95**). Search space in
  [configs/default.yaml](configs/default.yaml).
- **Containerized** on the `nvcr.io/nvidia/pytorch` base so CUDA/cuDNN/Hopper
  kernels are pre-tuned.

---

## Results

Populate after training (`results/baselines.csv` is generated by
`scripts/run_baselines.py`). Report PolarQuant vs JPEG 2000 vs uniform-scalar on
the **rate-distortion plane** (bpp on x, PSNR/SSIM on y) and include edge-region
crops (coastlines) to show orientation preservation.

| Method | bpp ↓ | PSNR (dB) ↑ | SSIM ↑ |
|--------|:----:|:-----------:|:------:|
| JPEG (q=90) | _tbd_ | _tbd_ | _tbd_ |
| JPEG 2000 | _tbd_ | _tbd_ | _tbd_ |
| AE + uniform scalar | _tbd_ | _tbd_ | _tbd_ |
| **AE + PolarQuant (ours)** | _tbd_ | _tbd_ | _tbd_ |

---

## References

- Ballé, Laparra, Simoncelli. *End-to-end Optimized Image Compression.* ICLR 2017.
- Theis, Shi, Cunningham, Huszár. *Lossy Image Compression with Compressive Autoencoders.* ICLR 2017. (STE for quantization)
- Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons.* 2013. (Straight-Through Estimator)
- Ballé et al. *Variational Image Compression with a Scale Hyperprior.* ICLR 2018.
