# Security policy

## Supported versions

Security fixes are applied to the latest release on `main`. Detektor is pre-1.0 in spirit even where version numbers say
otherwise — pin a commit or tag for production use.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/Kartik-A-1820/detektor/security/advisories/new) for this repository.
Include a description, reproduction steps and the impact you foresee. You can expect an initial response within a few
days; we will coordinate a fix and credit you in the release notes unless you prefer otherwise.

## Threat model and hardening guidance

Detektor is designed for single-machine and small-team deployments. The defaults favour local use:

| Area | Default | Production recommendation |
| --- | --- | --- |
| Network bind | `127.0.0.1` | Keep it, and expose only through a TLS-terminating reverse proxy |
| API authentication | **off** | Set `DETEKTOR_API_KEY` (≥ 32 random bytes); rotate by redeploying |
| Web console (`--ui`) | off; no login | If exposed, set `DETEKTOR_UI_AUTH=user:strong-password` **and** restrict by network |
| CORS | disabled | Allow-list exact origins with `DETEKTOR_CORS_ORIGINS` |
| Upload limits | 10 MB/image, 16 images/batch, 8192² px | Lower them to what your workload needs |
| Container | non-root (uid 10001), read-only FS, no capabilities (compose) | Keep these settings |

### Model files are code

Checkpoints (`.pt`) are Python pickles and can execute arbitrary code when loaded. **Never load a checkpoint from an
untrusted source.** Serve only artifacts you trained or verified; mount them read-only. If you distribute models,
publish a checksum (`scripts/package_model.py` records one in the artifact manifest).

### Input handling

Uploads are size-limited, MIME-checked (when declared), header-checked for decompression bombs and fully decoded before
inference. Errors never echo file contents. Request IDs are sanitised before logging.

### Known limitations

* No built-in rate limiting or TLS — use your proxy/gateway for both.
* The API key is a single shared secret; there is no per-user identity or audit trail beyond request IDs in the logs.
* `POST /runtime/select_model` can only switch among checkpoints discovered next to the configured weights.
