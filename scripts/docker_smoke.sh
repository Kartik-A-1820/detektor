#!/usr/bin/env bash
# Functional smoke test for a built Detektor image: serves a throw-away model and exercises the API.
#   scripts/docker_smoke.sh detektor:latest
set -euo pipefail

IMAGE="${1:?usage: scripts/docker_smoke.sh IMAGE}"
PORT="${PORT:-18000}"
KEY="smoke-key"
WORK="$(mktemp -d)"
chmod 777 "$WORK"   # the container runs as uid 10001
CID=""
cleanup() {
  if [ -n "$CID" ]; then
    echo "--- container logs (tail) ---"; docker logs "$CID" 2>&1 | tail -30 || true
    docker rm -f "$CID" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== image runs as non-root"
test "$(docker run --rm --entrypoint id "$IMAGE" -u)" = "10001"

echo "== create a throw-away checkpoint and test image inside the image"
docker run --rm -v "$WORK:/out" --entrypoint python "$IMAGE" -c "
from pathlib import Path
from benchmarks.common import save_profile_checkpoint, synthetic_jpeg
save_profile_checkpoint('firefly', 2, Path('/out/model.pt'))
Path('/out/test.jpg').write_bytes(synthetic_jpeg(320, 240))"

echo "== start the service (API key set, read-only weights)"
CID="$(docker run -d -p "$PORT:8000" -v "$WORK:/artifacts:ro" -e DETEKTOR_API_KEY="$KEY" "$IMAGE")"
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/ready"; echo
curl -fsS "http://127.0.0.1:$PORT/version"; echo

echo "== auth is enforced and inference works"
code="$(curl -s -o /dev/null -w '%{http_code}' -F "image=@$WORK/test.jpg" "http://127.0.0.1:$PORT/v1/predict")"
test "$code" = "401" || { echo "expected 401 without key, got $code"; exit 1; }
curl -fsS -H "X-API-Key: $KEY" -F "image=@$WORK/test.jpg" "http://127.0.0.1:$PORT/v1/predict" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'num_detections' in d and d['image_width']==320, d; print('predict ok:', d['num_detections'], 'detections')"
curl -fsS -H "X-API-Key: $KEY" "http://127.0.0.1:$PORT/metrics/prometheus" | grep -q '^detektor_requests_total'

echo "== container healthcheck reports healthy"
for _ in $(seq 1 30); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$CID")"
  [ "$status" = "healthy" ] && break
  sleep 2
done
test "$status" = "healthy" || { echo "healthcheck status: $status"; exit 1; }
echo "SMOKE TEST PASSED"
