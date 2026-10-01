# REST API reference

Detektor ships a FastAPI service (`serve.py`) with interactive OpenAPI docs at `/docs` (Swagger UI) and `/redoc`.

> **Security at a glance.** Out of the box the service binds to `127.0.0.1` with no authentication, which is appropriate
> for local use. Before exposing it on a network, set an API key (`--api-key` / `DETEKTOR_API_KEY`), put it behind a TLS
> reverse proxy, and protect the web console with `--ui-auth`. See [Security & operations](#security--operations) and
> [`SECURITY.md`](../../SECURITY.md).


## Starting the Server

**Basic:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt
```

**With CUDA:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --device cuda
```

**Custom host/port:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --host 0.0.0.0 --port 8080
```

## API Endpoints

> ✅ All endpoints emit `X-Request-ID` + `X-Response-Time` headers for traceability.

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `GET /health` | Liveness probe | Returns device + model load state |
| `GET /ready` | Readiness probe | Returns `ready=true` only after model + warmup |
| `GET /version` | Build info | Includes API + model metadata |
| `GET /metrics` | Lightweight stats | Latency percentiles, totals, error counts (🔒 when an API key is set) |
| `GET /metrics/prometheus` | Prometheus scrape target | Counters + latency histogram + build info (🔒) |
| `GET /runtime` | Active run metadata | Checkpoints, dataset, training summary (🔒) |
| `POST /runtime/select_model?model_key=best` | Hot-swap checkpoint | No restart needed (🔒) |
| `POST /v1/predict` | Single image inference | Structured JSON response with detections (🔒) |
| `POST /v1/predict_batch` | Multi-image inference | Validates batch size + per-image errors (🔒) |
| `POST /predict` | Legacy alias | Maps to `/v1/predict` (deprecated) (🔒) |

🔒 = requires the API key when one is configured. `/health`, `/ready` and `/version` are always open so orchestrator probes keep working.

**Health checks:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/version
curl http://localhost:8000/metrics
```

**Single prediction:**
```bash
# Inference options are *query* parameters; the image is the multipart body
curl -X POST "http://localhost:8000/v1/predict?include_masks=true&conf_thresh=0.35&iou_thresh=0.5" \
  -H "Accept: application/json" \
  -F "image=@image.jpg"
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/v1/predict_batch" \
  -F "images=@frame1.png" \
  -F "images=@frame2.png"
```

Responses now include:
```json
{
  "request_id": "7c6f...",
  "num_detections": 2,
  "detections": [
    {"box": [x1, y1, x2, y2], "score": 0.93, "label": 0, "mask": "..."},
    {"box": [x1, y1, x2, y2], "score": 0.88, "label": 3}
  ],
  "boxes": [[...]],         // legacy fields
  "scores": [0.93, 0.88],
  "labels": [0, 3],
  "image_width": 1024,
  "image_height": 768,
  "inference_time_ms": 42.3
}
```

Uploads are validated for MIME type, corrupt data, and size (`--max-upload-size-mb`).

## Interactive Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Configuration & Environment Variables

Every CLI flag has an `DETEKTOR_*` env var override so you can run via `uvicorn` or container orchestrators without editing the command line:

| Flag | Env Var | Default |
| --- | --- | --- |
| `--weights` | `DETEKTOR_WEIGHTS` | **required** |
| `--host` | `DETEKTOR_HOST` | `127.0.0.1` |
| `--port` | `DETEKTOR_PORT` | `8000` |
| `--device` | `DETEKTOR_DEVICE` | `auto` |
| `--num-classes` | `DETEKTOR_NUM_CLASSES` | auto-detect |
| `--proto-k` | `DETEKTOR_PROTO_K` | `24` |
| `--img-size` | `DETEKTOR_IMG_SIZE` | training size of the checkpoint (else `512`) |
| `--conf-thresh` | `DETEKTOR_CONF_THRESH` | `0.25` |
| `--iou-thresh` | `DETEKTOR_IOU_THRESH` | `0.6` |
| `--max-det` | `DETEKTOR_MAX_DET` | `100` |
| `--topk-pre-nms` | `DETEKTOR_TOPK_PRE_NMS` | `300` |
| `--mask-thresh` | `DETEKTOR_MASK_THRESH` | `0.5` |
| `--include-masks` | `DETEKTOR_INCLUDE_MASKS` | `false` |
| `--max-upload-size-mb` | `DETEKTOR_MAX_UPLOAD_SIZE_MB` | `10` |
| `--max-batch-size` | `DETEKTOR_MAX_BATCH_SIZE` | `16` |
| `--no-warmup` | `DETEKTOR_NO_WARMUP` | `false` |
| `--warmup-iterations` | `DETEKTOR_WARMUP_ITERATIONS` | `3` |
| `--log-level` | `DETEKTOR_LOG_LEVEL` | `INFO` |
| `--api-key` | `DETEKTOR_API_KEY` | unset (no auth) |
| `--cors-origins` | `DETEKTOR_CORS_ORIGINS` | unset (CORS off) |
| `--max-concurrency` | `DETEKTOR_MAX_CONCURRENCY` | `1` |
| `--ui` | `DETEKTOR_UI` | `false` |
| `--ui-path` | `DETEKTOR_UI_PATH` | `/ui` |
| `--ui-auth` | `DETEKTOR_UI_AUTH` | unset (open) |

Warmup is enabled by default and runs a few dummy passes to remove first-request latency spikes. Disable it with `--no-warmup` if you need instant start.

## Features

✅ **Auto num_classes detection** from checkpoint  
✅ **Modern FastAPI lifespan** handlers (no deprecation warnings)  
✅ **Async processing** for concurrent requests  
✅ **Structured logging** with request IDs  
✅ **Readiness + metrics endpoints** for orchestrators  
✅ **Optional mask output** via query parameter  
✅ **CUDA support** with automatic fallback


## Security & operations

### Authentication

```bash
export DETEKTOR_API_KEY=$(openssl rand -hex 32)
python serve.py --weights runs/chimera/chimera_best.pt --host 0.0.0.0

curl -H "X-API-Key: $DETEKTOR_API_KEY" -F "image=@photo.jpg" http://localhost:8000/v1/predict
curl -H "Authorization: Bearer $DETEKTOR_API_KEY" http://localhost:8000/metrics
```

Missing or wrong credentials return `401` with `WWW-Authenticate: Bearer`. Keys are compared in constant time. The key
protects the HTTP API; the Gradio console (`--ui`) has its own login via `--ui-auth user:password`.

### Request tracing

Every response carries `X-Request-ID` and `X-Response-Time`. A client-supplied `X-Request-ID` (≤ 64 chars of
`A-Z a-z 0-9 . _ -`) is propagated into the logs and the response; anything else is replaced by a generated UUID.

### Limits and validation

| Guard | Behaviour |
| --- | --- |
| Request body | `413 PayloadTooLarge` when `Content-Length` exceeds `max_batch_size × max_upload_size_mb` (+1 MB) |
| Per-image size | `--max-upload-size-mb` (default 10) → `400 ValidationError` |
| Content type | Missing/`application/octet-stream` accepted (content is verified by decoding); explicit non-image types rejected |
| Decompression bombs | Header-declared dimensions above 8192 × 8192 px are rejected *before* decoding |
| Dimensions | 32 – 8192 px per side |
| Batch size | `--max-batch-size` (default 16) |
| Concurrency | Inference runs in a worker thread; `--max-concurrency` model executions run at once, the rest queue |

### CORS

Disabled by default. Enable for browser clients with `--cors-origins https://app.example.com,https://admin.example.com`.
Only `GET`, `POST` and `OPTIONS` are allowed.

### Monitoring

`GET /metrics/prometheus` exposes (text format 0.0.4):

```
detektor_requests_total, detektor_errors_total, detektor_predictions_total
detektor_inference_latency_ms_{bucket,sum,count}   # histogram, buckets 5 ms … 5 s
detektor_uptime_seconds
detektor_build_info{version="…",device="…",checkpoint="…"} 1
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: detektor
    metrics_path: /metrics/prometheus
    authorization: { type: Bearer, credentials: "<DETEKTOR_API_KEY>" }
    static_configs: [{ targets: ["detektor:8000"] }]
```

### Error format

All errors share one JSON shape:

```json
{ "error": "ValidationError", "message": "Could not decode uploaded image", "request_id": "…", "details": null }
```
