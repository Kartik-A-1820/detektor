"""HTTP load test against a real, in-process Detektor server (uvicorn + FastAPI)."""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests
import uvicorn

from benchmarks.common import BenchContext, save_profile_checkpoint, synthetic_jpeg
from benchmarks.timing import summarize

DESCRIPTION = "Requests/second and latency percentiles of /v1/predict under concurrent clients"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(url: str, jpeg: bytes, n: int, params: Dict[str, Any], headers: Dict[str, str]) -> Tuple[List[float], int]:
    latencies: List[float] = []
    errors = 0
    with requests.Session() as session:
        for _ in range(n):
            start = time.perf_counter()
            try:
                resp = session.post(url, files={"image": ("bench.jpg", jpeg, "image/jpeg")}, params=params,
                                    headers=headers, timeout=120)
                ok = resp.status_code == 200
            except requests.RequestException:
                ok = False
            elapsed = (time.perf_counter() - start) * 1000.0
            if ok:
                latencies.append(elapsed)
            else:
                errors += 1
    return latencies, errors


def run(ctx: BenchContext) -> Dict[str, Any]:
    from serve import MODEL_STORE, ServiceConfig, create_app

    profile = ctx.profiles[0]
    img_size = min(ctx.img_size, 320) if ctx.quick else ctx.img_size
    ckpt = save_profile_checkpoint(profile, ctx.num_classes, ctx.scratch() / f"api_{profile}.pt", seed=ctx.seed)
    api_key = "bench-key"
    config = ServiceConfig(
        weights=str(ckpt),
        device=ctx.device,
        image_size=img_size,
        enable_warmup=True,
        warmup_iterations=2,
        log_level="WARNING",
        api_key=api_key,
        max_concurrency=1,
        conf_thresh=0.25,
    )
    app = create_app(config)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):  # up to ~30 s for model load + warmup
            try:
                if requests.get(f"{base}/ready", timeout=1).json().get("ready"):
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)
        else:
            return {"description": DESCRIPTION, "error": "server did not become ready"}

        jpeg = synthetic_jpeg(1280, 720, seed=ctx.seed)
        headers = {"X-API-Key": api_key}
        # Auth check doubles as a correctness probe.
        unauth = requests.post(f"{base}/v1/predict", files={"image": ("a.jpg", jpeg, "image/jpeg")}, timeout=30).status_code
        total_requests = 24 if ctx.quick else 80
        levels = [1, 2, 4] if ctx.quick else [1, 2, 4, 8, 16]
        rows = []
        for conc in levels:
            per_worker = max(1, total_requests // conc)
            _worker(f"{base}/v1/predict", jpeg, 2, {}, headers)  # settle
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as pool:
                futures = [pool.submit(_worker, f"{base}/v1/predict", jpeg, per_worker, {}, headers) for _ in range(conc)]
                results = [f.result() for f in futures]
            wall = time.perf_counter() - started
            lats = [lat for r, _ in results for lat in r]
            errs = sum(e for _, e in results)
            ok = len(lats)
            rows.append(
                {
                    "concurrency": conc,
                    "requests": ok + errs,
                    "errors": errs,
                    "rps": round(ok / wall, 2) if wall > 0 else 0.0,
                    "latency": summarize(lats),
                }
            )
        return {
            "description": DESCRIPTION,
            "profile": profile,
            "img_size": img_size,
            "device": ctx.device,
            "payload": "synthetic 1280x720 JPEG",
            "server_max_concurrency": config.max_concurrency,
            "unauthenticated_status": unauth,
            "rows": rows,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        MODEL_STORE.inference_service = None
        ckpt.unlink(missing_ok=True)
