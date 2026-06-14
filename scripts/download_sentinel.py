"""Phase 1: fetch a small Sentinel-2 subset for development.

Sentinel-2 L2A is freely available from several mirrors. The cleanest
no-account path for a *subset* is the Hugging Face mirrors of EuroSAT
(Sentinel-2 RGB/multispectral 64x64 tiles) or BigEarthNet patches. This script
pulls EuroSAT (RGB) by default — small, fast, and license-clean for prototyping
— and lays it out under data/sentinel/ for the dataset class.

For full-resolution scenes (true 256x256 patch cropping), use the Copernicus
Data Space Ecosystem (https://dataspace.copernicus.eu) with an account and the
`sentinelsat`/`openeo` APIs; this script intentionally avoids credentials so the
pipeline is runnable out of the box.

Usage:
    python scripts/download_sentinel.py --out data/sentinel
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/sentinel")
    ap.add_argument("--dataset", choices=["eurosat"], default="eurosat")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "eurosat":
        _download_eurosat(out)


def _download_eurosat(out: Path):
    """Download EuroSAT RGB (Sentinel-2) via torchvision into `out`."""
    try:
        from torchvision.datasets import EuroSAT
    except Exception as e:
        raise SystemExit(
            f"torchvision required: {e}\n"
            "pip install torchvision, or download EuroSAT manually from "
            "https://github.com/phelber/EuroSAT and unzip into the --out dir."
        )

    print(f"[download] EuroSAT (Sentinel-2 RGB) -> {out}")
    EuroSAT(root=str(out), download=True)
    print(
        "[download] done. Point the dataset at the extracted image folder, e.g.\n"
        f"    Sentinel2PatchDataset('{out}', reflectance_scale=255, patch_size=64)"
    )


if __name__ == "__main__":
    main()
