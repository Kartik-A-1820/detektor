# syntax=docker/dockerfile:1.6
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DETEKTOR_HOST=0.0.0.0 \
    DETEKTOR_PORT=8000 \
    DETEKTOR_WEIGHTS=/artifacts/model.pt \
    DETEKTOR_DEVICE=cpu

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

ENTRYPOINT ["python", "serve.py"]
CMD []
