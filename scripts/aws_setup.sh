#!/usr/bin/env bash
# SatCompress — one-shot setup on an AWS GPU instance.
#
# Assumes you are INSIDE the repo directory on the instance (cloned or scp'd) and
# on an "AWS Deep Learning AMI (Ubuntu)" where CUDA + PyTorch are preinstalled.
# Run:   bash scripts/aws_setup.sh
#
# It: checks the GPU, installs the (non-torch) Python deps, installs the package,
# downloads the EuroSAT (Sentinel-2 RGB) subset, and runs the test suite.
set -euo pipefail

echo "==> GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! No NVIDIA GPU visible. Are you on a g4dn/g5/g6 instance?"; exit 1; }

echo "==> PyTorch check"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "   torch+CUDA not found in this env; installing CUDA wheels..."
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print('   torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> Core Python deps"
pip install --quiet numpy pillow scikit-image wandb pyyaml tqdm pytest \
    fastapi "uvicorn[standard]" streamlit python-multipart

echo "==> Geo/codec deps (rasterio + pystac-client needed for raw Sentinel-2 tiles)"
pip install --quiet glymur imagecodecs rasterio pystac-client || \
  echo "   (some geo libs failed — fine for the EuroSAT RGB path, but raw S2 needs rasterio+pystac-client)"

echo "==> Install satcompress (package only)"
pip install --quiet -e . --no-deps

echo "==> Download EuroSAT (Sentinel-2 RGB) -> data/sentinel"
python scripts/download_sentinel.py --out data/sentinel

echo "==> Sanity: run the test suite"
python -m pytest -q

echo
echo "✅ Setup complete. Next:  bash scripts/aws_train.sh"
echo "   (Remember to STOP the instance when training finishes.)"
