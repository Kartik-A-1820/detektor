# Deployment guide

Detektor serves a trained checkpoint over HTTP (FastAPI) and, optionally, a web console (Gradio). This guide covers
running it as a container, behind a reverse proxy, under systemd and on Kubernetes, plus a production checklist.

> **Verification status.** The service, auth, limits and metrics are covered by the test suite and were exercised
> end-to-end in CI-style runs. The container, systemd and Kubernetes snippets below follow standard practice and the
> CI workflow builds the image on every push, but they have not been load-tested against every orchestrator —
> treat them as a vetted starting point.

## 1. Prepare an artifact

```bash
# Train (or reuse) a checkpoint, then package it with metadata and a SHA-256 checksum
python -m scripts.package_model --weights runs/chimera/chimera_best.pt --output-dir artifacts --name my_model
# → artifacts/my_model/{model.pt, package_manifest.json, class_names.json, environment.txt, …}
sha256sum artifacts/my_model/model.pt     # compare with package_manifest.json → checksums.model_pt
```

Serve only artifacts you trained or verified — checkpoints are pickles ([SECURITY.md](../../SECURITY.md)).

## 2. Docker

```bash
# CPU image (small: CPU-only PyTorch wheels)
docker build -t detektor:latest .

# NVIDIA GPU image
docker build -t detektor:gpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .

docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts/my_model:/artifacts:ro" \
  -e DETEKTOR_WEIGHTS=/artifacts/model.pt \
  -e DETEKTOR_API_KEY="$(openssl rand -hex 32)" \
  detektor:latest

# GPU
docker run --rm --gpus all -p 8000:8000 -v "$PWD/artifacts/my_model:/artifacts:ro" \
  -e DETEKTOR_WEIGHTS=/artifacts/model.pt -e DETEKTOR_DEVICE=cuda detektor:gpu
```

The image runs as a **non-root user (uid 10001)**, has a `HEALTHCHECK` on `/ready`, and never bakes in weights.

### Docker Compose

```bash
cp .env.example .env            # set DETEKTOR_API_KEY, DETEKTOR_WEIGHTS, …
mkdir -p artifacts && cp artifacts/my_model/* artifacts/
docker compose up --build -d                       # CPU
docker compose --profile gpu up --build -d detektor-gpu   # GPU (needs the NVIDIA container toolkit)
docker compose logs -f detektor
```

The compose service is hardened by default: read-only root filesystem (`/tmp` is a tmpfs), `no-new-privileges`, all
Linux capabilities dropped, weights mounted read-only.

## 3. Behind a reverse proxy (TLS)

Bind Detektor to localhost/an internal network and terminate TLS in front of it. Example nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name detect.example.com;
    ssl_certificate     /etc/ssl/certs/detect.pem;
    ssl_certificate_key /etc/ssl/private/detect.key;

    client_max_body_size 170m;           # ≥ max_batch_size × max_upload_size_mb (+ overhead)
    proxy_read_timeout   120s;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Request-ID $request_id;   # propagated to logs and responses
        proxy_set_header   X-Forwarded-For $remote_addr;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
    # Gradio console needs websocket upgrades
    location /ui/ {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
    }
}
```

Add rate limiting (`limit_req`) at the proxy — Detektor does not implement it.

## 4. systemd

```ini
# /etc/systemd/system/detektor.service
[Unit]
Description=Detektor inference service
After=network.target

[Service]
User=detektor
WorkingDirectory=/opt/detektor
EnvironmentFile=/etc/detektor.env          # DETEKTOR_WEIGHTS=…, DETEKTOR_API_KEY=… (chmod 600)
ExecStart=/opt/detektor/.venv/bin/python serve.py
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadOnlyPaths=/opt/detektor /srv/models

[Install]
WantedBy=multi-user.target
```

## 5. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: detektor }
spec:
  replicas: 2
  selector: { matchLabels: { app: detektor } }
  template:
    metadata:
      labels: { app: detektor }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: /metrics/prometheus
        prometheus.io/port: "8000"
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001 }
      containers:
        - name: detektor
          image: ghcr.io/your-org/detektor:latest        # build & push your own image
          ports: [{ containerPort: 8000 }]
          env:
            - { name: DETEKTOR_WEIGHTS, value: /models/model.pt }
            - { name: DETEKTOR_DEVICE, value: cpu }
            - name: DETEKTOR_API_KEY
              valueFrom: { secretKeyRef: { name: detektor, key: api-key } }
          readinessProbe: { httpGet: { path: /ready, port: 8000 }, periodSeconds: 5 }
          livenessProbe:  { httpGet: { path: /health, port: 8000 }, periodSeconds: 20, failureThreshold: 3 }
          startupProbe:   { httpGet: { path: /ready, port: 8000 }, failureThreshold: 30, periodSeconds: 2 }
          resources:
            requests: { cpu: "1", memory: 1Gi }
            limits:   { memory: 2Gi }
          securityContext: { readOnlyRootFilesystem: true, allowPrivilegeEscalation: false, capabilities: { drop: [ALL] } }
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
            - { name: tmp, mountPath: /tmp }
      volumes:
        - { name: models, persistentVolumeClaim: { claimName: detektor-models } }
        - { name: tmp, emptyDir: {} }
```

Scale horizontally with replicas; each replica holds one model copy. Use `DETEKTOR_MAX_CONCURRENCY=1` per CPU-bound
replica and let the Service load-balance.

## 6. Capacity planning

Throughput is bounded by one model execution at a time per process (`--max-concurrency`). Use the benchmarks to size it:

```bash
python -m benchmarks run --suites latency,throughput,memory,api --profiles <your-profile> --img-sizes <your-size>
```

* **Per-request latency** ≈ decode + resize + forward + postprocess (the `latency` suite breaks these down).
* **Requests/s** ≈ `1000 / predict_p50_ms` per replica (see the `api` suite for measured concurrency scaling).
* **Memory** — the `memory` suite reports peak RSS/VRAM for inference; keep ≥ 1.5× headroom.
* Reference numbers for a 4-vCPU CPU host are in [BENCHMARKS.md](../BENCHMARKS.md).
* ONNX Runtime can be several times faster than eager PyTorch on CPU (see the `onnx` suite) — export with
  `python export.py …` if CPU throughput matters.

## 7. Production checklist

- [ ] `DETEKTOR_API_KEY` set; secrets come from a secret store, not the image or repo
- [ ] TLS terminated at a proxy; service reachable only via the proxy / private network
- [ ] `--ui` disabled, or protected with `DETEKTOR_UI_AUTH` and network restrictions
- [ ] CORS limited to explicit origins (or left off)
- [ ] Upload / batch limits tuned to the workload
- [ ] Weights mounted read-only from a trusted source; checksum verified
- [ ] Readiness/liveness probes wired to `/ready` and `/health`
- [ ] Prometheus scraping `/metrics/prometheus`; alerts on error rate and p95 latency
- [ ] Logs collected from stdout (structured, with request IDs)
- [ ] Load-tested with the `api` benchmark suite on production-like hardware
