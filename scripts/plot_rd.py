"""Plot rate-distortion curves from results/rd_results.csv.

Produces results/rd_psnr.png and results/rd_ssim.png — bits-per-pixel (x) vs
quality (y), one line per method. The winning method's curve is higher (more
quality at the same bitrate) and/or further left (fewer bits at the same quality).

Usage:
    python scripts/plot_rd.py --csv results/rd_results.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (Kaggle/CI)
import matplotlib.pyplot as plt  # noqa: E402

# Neural methods drawn as solid lines; classical codecs as dashed.
STYLE = {
    "polar-log-rvm": dict(marker="o", linestyle="-", label="PolarQuant + log + Rayleigh-vM (ours)"),
    "polar": dict(marker="s", linestyle="-", label="PolarQuant"),
    "scalar": dict(marker="^", linestyle="-", label="Scalar quant (control)"),
    "jpeg": dict(marker="x", linestyle="--", label="JPEG"),
    "jpeg2000": dict(marker="+", linestyle="--", label="JPEG 2000"),
}


def load(csv_path):
    pts = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pts[row["method"]].append((float(row["bpp"]), float(row["psnr"]), float(row["ssim"])))
    for m in pts:
        pts[m].sort(key=lambda t: t[0])  # sort by bpp for clean lines
    return pts


def _plot(pts, yidx, ylabel, title, out_path):
    plt.figure(figsize=(7, 5))
    for method, style in STYLE.items():
        if method not in pts:
            continue
        xs = [p[0] for p in pts[method]]
        ys = [p[yidx] for p in pts[method]]
        plt.plot(xs, ys, **style)
    plt.xlabel("bits per pixel (bpp)  ↓ smaller is better")
    plt.ylabel(f"{ylabel}  ↑ higher is better")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/rd_results.csv")
    args = ap.parse_args()
    pts = load(args.csv)
    if not pts:
        raise SystemExit(f"No data in {args.csv} — run scripts/rd_sweep.py first.")
    out_dir = Path(args.csv).parent
    _plot(pts, 1, "PSNR (dB)", "Rate-Distortion: PSNR vs bpp", out_dir / "rd_psnr.png")
    _plot(pts, 2, "SSIM", "Rate-Distortion: SSIM vs bpp", out_dir / "rd_ssim.png")


if __name__ == "__main__":
    main()
