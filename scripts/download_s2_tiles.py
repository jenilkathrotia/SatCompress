"""Download raw Sentinel-2 L2A 256x256 tiles (4 bands: RGB+NIR) for compression.

Source: Earth Search v1 STAC API (https://earth-search.aws.element84.com/v1),
collection `sentinel-2-l2a` -> Cloud-Optimized GeoTIFFs on AWS open data. This is
free and needs NO account / NO AWS credentials.

Why this is light despite being "raw": COGs support HTTP range reads, so we pull
only the 256x256 windows we keep, never the ~1 GB full scenes.

Each output tile is a (4, 256, 256) uint16 GeoTIFF with band order
[B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)] -> NDVI-ready. Cloudy/no-data
windows are dropped using the scene-classification (SCL) band.

Usage:
    python scripts/download_s2_tiles.py --out data/s2 --num-tiles 15000
    python scripts/download_s2_tiles.py --out data/s2_smoke --num-tiles 8   # smoke
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# GDAL tuning for anonymous, efficient remote COG reads (must precede rasterio).
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np  # noqa: E402

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Earth Search v1 uses common-name asset keys. Band order of the output tile:
BAND_ASSETS = ["red", "green", "blue", "nir"]  # B04, B03, B02, B08
BAND_NAMES = ["B04", "B03", "B02", "B08"]
SCL_ASSET = "scl"

# Sentinel-2 SCL classes considered unusable: nodata, saturated, cloud shadow,
# cloud (medium/high), thin cirrus.
SCL_BAD = frozenset({0, 1, 3, 8, 9, 10})

# Diverse areas of interest (lon_min, lat_min, lon_max, lat_max) chosen to span
# coastlines, farmland, rivers, mountains, desert, forest, and urban texture.
AOI_PRESETS = {
    "ca_coast": (-122.6, 36.5, -121.8, 37.3),
    "nl_farmland": (4.0, 51.8, 5.2, 52.6),
    "nile_delta": (30.5, 30.5, 31.7, 31.4),
    "alps": (6.8, 45.8, 8.0, 46.6),
    "sahara_edge": (2.0, 16.0, 3.2, 17.0),
    "amazon": (-60.5, -3.5, -59.3, -2.7),
    "tokyo_urban": (139.4, 35.4, 140.2, 36.0),
    "perth_coast": (115.5, -32.3, 116.5, -31.5),
}


def scl_bad_fraction(scl_window: np.ndarray, bad_classes=SCL_BAD) -> float:
    """Fraction of pixels in `scl_window` whose class is in `bad_classes`.

    Pure function (no I/O) so it is unit-testable without network access.
    """
    if scl_window.size == 0:
        return 1.0
    bad = np.isin(scl_window, list(bad_classes))
    return float(bad.mean())


def parse_args():
    ap = argparse.ArgumentParser(description="Download Sentinel-2 256x256 RGB+NIR tiles")
    ap.add_argument("--out", type=str, default="data/s2")
    ap.add_argument("--num-tiles", type=int, default=15000)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--max-cloud", type=float, default=10.0, help="scene cloud_cover %% cap")
    ap.add_argument("--max-bad-frac", type=float, default=0.1, help="per-tile SCL bad-pixel cap")
    ap.add_argument("--max-per-scene", type=int, default=400, help="cap tiles per scene for variety")
    ap.add_argument("--date-range", type=str, default="2024-04-01/2024-09-30")
    ap.add_argument("--scenes-per-aoi", type=int, default=12)
    ap.add_argument("--preset", nargs="*", default=list(AOI_PRESETS), choices=list(AOI_PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def find_scenes(aois, date_range, max_cloud, scenes_per_aoi):
    """Query Earth Search for low-cloud scenes across the requested AOIs."""
    from pystac_client import Client

    client = Client.open(EARTH_SEARCH_URL)
    items = []
    seen = set()
    for name in aois:
        bbox = AOI_PRESETS[name]
        search = client.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            max_items=scenes_per_aoi,
        )
        n = 0
        for it in search.items():
            if it.id in seen:
                continue
            seen.add(it.id)
            items.append(it)
            n += 1
        print(f"[s2] AOI {name}: {n} scenes (cloud < {max_cloud}%)")
    random.shuffle(items)
    return items


def _asset_href(item, key):
    asset = item.assets.get(key)
    if asset is None:
        raise KeyError(f"asset '{key}' missing from item {item.id}")
    return asset.href


def extract_tiles_from_scene(item, out_dir, ps, max_bad_frac, max_per_scene, remaining, rng):
    """Read 256x256 windows from one scene's COGs; write the clean ones. Returns count."""
    import rasterio
    from rasterio.windows import Window, transform as window_transform

    band_hrefs = [_asset_href(item, k) for k in BAND_ASSETS]
    scl_href = _asset_href(item, SCL_ASSET)

    written = 0
    try:
        srcs = [rasterio.open(h) for h in band_hrefs]
    except Exception as e:
        print(f"[s2]   skip {item.id}: open failed ({e})")
        return 0
    try:
        scl_src = rasterio.open(scl_href)
        H, W = srcs[0].height, srcs[0].width
        # SCL is 20 m (half the 10 m grid); scale factor for window coords.
        scl_scale = scl_src.width / W
        # candidate top-left corners on a non-overlapping grid, shuffled
        corners = [(r, c) for r in range(0, H - ps + 1, ps) for c in range(0, W - ps + 1, ps)]
        rng.shuffle(corners)
        for (row, col) in corners:
            if written >= max_per_scene or remaining - written <= 0:
                break
            win = Window(col, row, ps, ps)
            try:
                stack = np.stack([s.read(1, window=win) for s in srcs]).astype(np.uint16)
            except Exception:
                continue
            # drop near-empty (nodata is 0 in S2 L2A COGs)
            if (stack == 0).mean() > max_bad_frac:
                continue
            # SCL cloud/shadow/nodata filter
            sps = max(1, int(round(ps * scl_scale)))
            scl_win = Window(int(col * scl_scale), int(row * scl_scale), sps, sps)
            try:
                scl = scl_src.read(1, window=scl_win)
            except Exception:
                scl = np.array([])
            if scl_bad_fraction(scl) > max_bad_frac:
                continue
            transform = window_transform(win, srcs[0].transform)
            out_path = out_dir / f"{item.id}_{row}_{col}.tif"
            _write_tile(rasterio, out_path, stack, srcs[0].crs, transform)
            written += 1
    finally:
        for s in srcs:
            s.close()
        try:
            scl_src.close()
        except Exception:
            pass
    return written


def _write_tile(rasterio, path, stack, crs, transform):
    c, h, w = stack.shape
    profile = dict(
        driver="GTiff", height=h, width=w, count=c, dtype="uint16",
        crs=crs, transform=transform, compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[s2] searching Earth Search across {len(args.preset)} AOIs ...")
    scenes = find_scenes(args.preset, args.date_range, args.max_cloud, args.scenes_per_aoi)
    print(f"[s2] {len(scenes)} candidate scenes; extracting up to {args.num_tiles} tiles -> {out}")

    try:
        from tqdm import tqdm
        scene_iter = tqdm(scenes, desc="scenes")
    except Exception:
        scene_iter = scenes

    total = 0
    for item in scene_iter:
        if total >= args.num_tiles:
            break
        got = extract_tiles_from_scene(
            item, out, args.patch_size, args.max_bad_frac,
            args.max_per_scene, args.num_tiles - total, rng,
        )
        total += got
    print(f"[s2] done: wrote {total} tiles to {out}")
    if total == 0:
        print("[s2] WARNING: 0 tiles — try a wider --date-range or higher --max-cloud.")


if __name__ == "__main__":
    main()
