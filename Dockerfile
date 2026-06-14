# Phase 4: containerized training/serving image for Nebius H100.
# Built on the NVIDIA PyTorch image so CUDA, cuDNN, and Tensor-Core BF16/FP8
# kernels are already tuned for Hopper.
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace

# System libs for rasterio/imagecodecs (GDAL + OpenJPEG)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch/torchvision already in base image; install the rest.
RUN pip install --no-cache-dir \
        numpy pillow scikit-image glymur imagecodecs rasterio \
        wandb pyyaml tqdm fastapi "uvicorn[standard]" streamlit python-multipart pytest

COPY . .
RUN pip install --no-cache-dir -e .

# Default: train PolarQuant. Override CMD on `docker run` for serving:
#   docker run ... uvicorn serving.api:app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "satcompress.train", "--quantizer", "polar", "--amp", "bf16"]
