#!/usr/bin/env bash
# Boot-status smoke: the server answers /v1/status from the moment it binds.
# Runs the built binary with a BOGUS model id, so it exercises the model-missing
# path with no weights: status must flip to state=error (with "not downloaded"),
# inference must 503 with the reason, and the process must STAY ALIVE (no crash
# loop under launchd). `pixi run boot-smoke` (builds first; needs Metal only to
# build, not to run — the missing-model path never touches the GPU).
set -euo pipefail
PORT="${1:-8123}"
BIN="${BIN:-build/server}"

MILLFOLIO_PORT="$PORT" HF_HOME="$(mktemp -d)" "$BIN" totally/bogus-model-id &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

fail() { echo "[FAIL] $1"; exit 1; }

# status must answer within ~2s of process start and report the missing model
for i in $(seq 1 20); do
  STATUS="$(curl -s -m 1 "http://127.0.0.1:$PORT/v1/status" || true)"
  [ -n "$STATUS" ] && break
  sleep 0.1
done
echo "  status: $STATUS"
echo "$STATUS" | grep -q '"state":"error"' || fail "status is not state=error"
echo "$STATUS" | grep -q 'not downloaded'  || fail "error does not name the missing model"

# inference must 503 with the reason, not hang or crash
CODE="$(curl -s -m 2 -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/v1/chat/completions" -d '{"messages":[]}')"
[ "$CODE" = "503" ] || fail "chat returned $CODE, want 503"

# /v1/version must NOT return 200 before ready (old CLIs treat 200 as ready)
VCODE="$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/version")"
[ "$VCODE" = "503" ] || fail "/v1/version returned $VCODE during error state, want 503"

# and the process is still alive (a missing model must not be fatal)
kill -0 "$PID" 2>/dev/null || fail "server process died on a missing model"

echo "ALL CHECKS PASSED"
