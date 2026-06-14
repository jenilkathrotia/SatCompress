#!/usr/bin/env bash
# SatCompress — core experiment suite on raw Sentinel-2 256x256 tiles (4-band RGB+NIR).
#
# Assumes tiles already pulled:  python scripts/download_s2_tiles.py --out data/s2
# Produces the ablation that fills the results table:
#   1. classical JPEG / JPEG2000 baselines (on the RGB bands)
#   2. autoencoder + uniform scalar quantization   (control)
#   3. autoencoder + PolarQuant                     (proposed)
#   4. autoencoder + PolarQuant + log-polar + Rayleigh-von Mises entropy model
#
# Tunables (override via env): EPOCHS=50 BATCH=24 bash scripts/train_s2.sh
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-16}"           # 256x256 is heavy; 16 fits a 16 GB T4 (fp16). AWS can go higher.
LATENT="${LATENT:-192}"        # even -> clean PolarQuant pairing
DATA="${DATA:-data/s2}"
AMP="${AMP:-fp16}"             # T4: fp16. P100/older: set AMP=off. H100: bf16.
PS=256
SCALE=10000                    # Sentinel-2 L2A reflectance
CH=4                           # B04,B03,B02,B08
COMMON="--data-root $DATA --patch-size $PS --reflectance-scale $SCALE --channels $CH \
  --patches-per-scene 1 --latent $LATENT --batch-size $BATCH --epochs $EPOCHS \
  --amp $AMP --num-workers 4"
set -euo pipefail

echo "==> [1/4] Classical baselines (JPEG / JPEG2000, RGB bands)"
python scripts/run_baselines.py --data-root "$DATA" --patch-size $PS \
  --reflectance-scale $SCALE --limit 200

echo "==> [2/4] Uniform scalar quantization (control)"
python -m satcompress.train $COMMON --quantizer scalar

echo "==> [3/4] PolarQuant (proposed)"
python -m satcompress.train $COMMON --quantizer polar

echo "==> [4/4] PolarQuant + log-polar + Rayleigh-von Mises entropy model"
python -m satcompress.train $COMMON --quantizer polar \
  --radial-mode log --entropy-model rayleigh-vm

echo
echo "✅ All runs done. Checkpoints in ./checkpoints, baselines in results/baselines.csv"
echo "   STOP the instance now if you're finished."
