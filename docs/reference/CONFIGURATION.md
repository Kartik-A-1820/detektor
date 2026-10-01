# Configuration reference

Every `serve.py` option can be set by CLI flag **or** environment variable (flags win). Booleans accept `true`/`false`.

## Model & device

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--weights` | `DETEKTOR_WEIGHTS` | *required* | Checkpoint (`.pt`). Sibling `chimera_best.pt` / `chimera_last.pt` become switchable checkpoints |
| `--device` | `DETEKTOR_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `--num-classes` | `DETEKTOR_NUM_CLASSES` | auto | Override the class count inferred from the checkpoint |
| `--proto-k` | `DETEKTOR_PROTO_K` | `24` | Legacy fallback for checkpoints without embedded model metadata (ignored otherwise) |
| `--img-size` | `DETEKTOR_IMG_SIZE` | checkpoint's training size (else `512`) | Square network input size. Leave unset so the model is served at the resolution it was trained at |

## Inference defaults (overridable per request)

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--conf-thresh` | `DETEKTOR_CONF_THRESH` | `0.25` | Minimum score |
| `--iou-thresh` | `DETEKTOR_IOU_THRESH` | `0.6` | NMS IoU |
| `--max-det` | `DETEKTOR_MAX_DET` | `100` | Max detections per image |
| `--topk-pre-nms` | `DETEKTOR_TOPK_PRE_NMS` | `300` | Candidates kept before NMS |
| `--mask-thresh` | `DETEKTOR_MASK_THRESH` | `0.5` | Mask binarisation threshold |
| `--include-masks` | `DETEKTOR_INCLUDE_MASKS` | `false` | Return base64 PNG masks by default |

## Server

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--host` | `DETEKTOR_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` in containers) |
| `--port` | `DETEKTOR_PORT` | `8000` | Port |
| `--max-upload-size-mb` | `DETEKTOR_MAX_UPLOAD_SIZE_MB` | `10` | Per-image limit |
| `--max-batch-size` | `DETEKTOR_MAX_BATCH_SIZE` | `16` | Images per batch request |
| `--max-concurrency` | `DETEKTOR_MAX_CONCURRENCY` | `1` | Simultaneous model executions |
| `--no-warmup` | `DETEKTOR_NO_WARMUP` | `false` | Skip start-up warm-up passes |
| `--warmup-iterations` | `DETEKTOR_WARMUP_ITERATIONS` | `3` | Warm-up passes |
| `--log-level` | `DETEKTOR_LOG_LEVEL` | `INFO` | `DEBUG`…`CRITICAL` |

## Security

| Flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--api-key` | `DETEKTOR_API_KEY` | unset | Require `X-API-Key` / `Bearer` on inference, runtime and metrics endpoints |
| `--cors-origins` | `DETEKTOR_CORS_ORIGINS` | unset | Comma-separated allowed origins |
| `--ui` | `DETEKTOR_UI` | `false` | Mount the web console |
| `--ui-path` | `DETEKTOR_UI_PATH` | `/ui` | Console mount path |
| `--ui-auth` | `DETEKTOR_UI_AUTH` | unset | `user:password` login for the console |

## Remote console (`python -m ui.app`)

| Variable | Default | Description |
| --- | --- | --- |
| `DETEKTOR_UI_BACKEND` | `http://localhost:8000` | API the console talks to |
| `DETEKTOR_UI_HOST` / `DETEKTOR_UI_PORT` | `127.0.0.1` / `7860` | Bind address |

## Docker Compose helpers

`DETEKTOR_PUBLISH_PORT` (host port, default `8000`) is read by `docker-compose.yml` only.
