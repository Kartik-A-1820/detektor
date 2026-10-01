# syntax=docker/dockerfile:1.6
#
# Detektor inference image.
#   CPU (default, small):   docker build -t detektor:latest .
#   NVIDIA GPU:             docker build -t detektor:gpu \
#                             --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
#
# Weights are NOT baked in: mount them at /artifacts and set DETEKTOR_WEIGHTS.

ARG PYTHON_IMAGE=python:3.11-slim

# ---- builder: resolve and install dependencies into an isolated venv ---------------------
FROM ${PYTHON_IMAGE} AS builder
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt ./
# Install torch first from the selected index so the CPU image does not pull ~2 GB of CUDA libs.
RUN pip install --index-url ${TORCH_INDEX_URL} "torch>=2.1" "torchvision>=0.16" \
 && pip install -r requirements.txt

# ---- runtime ------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="Detektor" \
      org.opencontainers.image.description="Object detection and instance segmentation inference service" \
      org.opencontainers.image.source="https://github.com/Kartik-A-1820/detektor" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DETEKTOR_HOST=0.0.0.0 \
    DETEKTOR_PORT=8000 \
    DETEKTOR_WEIGHTS=/artifacts/model.pt \
    DETEKTOR_DEVICE=cpu \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 detektor \
 && useradd --uid 10001 --gid detektor --create-home --shell /usr/sbin/nologin detektor

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=detektor:detektor . .

USER detektor
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${DETEKTOR_PORT}/ready || exit 1

ENTRYPOINT ["python", "serve.py"]
CMD []
