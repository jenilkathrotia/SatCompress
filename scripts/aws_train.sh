#!/usr/bin/env bash
# SatCompress — run the core experiment suite on an AWS GPU instance (EuroSAT).
#
# Produces the ablation that fills your results table:
#   1. classical JPEG / JPEG2000 baselines
#   2. autoencoder + uniform scalar quantization   (control)
#   3. autoencoder + PolarQuant                     (proposed)
#   4. autoencoder + PolarQuant + log-polar + Rayleigh-von Mises entropy model
#
# Tunables (override via env), e.g.:  EPOCHS=50 BATCH=128 bash scripts/aws_train.sh
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"          # g5/g6 (24 GB) handle this easily at 64x64
LATENT="${LATENT:-128}"        # even -> clean PolarQuant pairing
DATA="${DATA:-data/sentinel}"  # EuroSAT extracted here by aws_setup.sh
PS=64                          # EuroSAT tiles are 64x64
SCALE=255                      # 8-bit RGB
COMMON="--data-root $DATA --patch-size $PS --reflectance-scale $SCALE \
  --channels 3 --latent $LATENT --batch-size $BATCH --epochs $EPOCHS \
  --amp bf16 --num-workers 8"
set -euo pipefail

echo "==> [1/4] Classical baselines (JPEG / JPEG2000)"
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
echo "   Optional W&B: run 'wandb login' before this script to log curves;"
echo "   otherwise add --no-wandb to each command. STOP the instance now if finished."
