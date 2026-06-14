"""Phase 1: compute classical JPEG / JPEG 2000 rate-distortion baselines.

Usage:
    python scripts/run_baselines.py --data-root data/sentinel --limit 50
    python scripts/run_baselines.py --synthetic            # no data needed

Writes results/baselines.csv with one row per (codec, setting).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from satcompress.baselines import rate_distortion_sweep
from satcompress.data import RandomPatchDataset, Sentinel2PatchDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--reflectance-scale", type=float, default=10000.0,
                    help="255 for 8-bit EuroSAT RGB; 10000 for S2 L2A reflectance")
    ap.add_argument("--out", type=str, default="results/baselines.csv")
    args = ap.parse_args()

    if args.synthetic or not args.data_root:
        ds = RandomPatchDataset(length=args.limit, channels=3, patch_size=args.patch_size)
    else:
        ds = Sentinel2PatchDataset(
            args.data_root, patch_size=args.patch_size,
            reflectance_scale=args.reflectance_scale,
        )

    # Use ALL bands so bpp is comparable to the neural model (which compresses all
    # bands). JPEG handles >3 bands via RGB + grayscale; JPEG2000 is multi-component.
    images = [ds[i] for i in range(min(args.limit, len(ds)))]
    records = rate_distortion_sweep(images)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in records for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    for r in records:
        print(r)
    print(f"\n[baselines] wrote {out}")


if __name__ == "__main__":
    main()
