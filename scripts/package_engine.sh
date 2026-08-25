#!/usr/bin/env bash
#
# Build runner.zip — the engine ("runner") source bundle the Millfolio menu app
# downloads, then `mojo build`s on-device against a separately-fetched Mojo
# compiler (see millfolio/app Bootstrapper). The bundle unzips to three siblings:
#
#   inference-server/   src + assets +
#                   build/{libflare_tls.so + libssl.3 + libcrypto.3, rpath-fixed}
#   pkgs/jinja2.mojoc   compiled jinja2 package (from the pixi env's jinja2-mojo
#                       source dependency — tied to the pinned mojo nightly)
#   flare/flare/    vendored flare package (HTTP server)
#
# so the app can run:
#   (cd inference-server && mojo build src/server.mojo -I ../pkgs -I ../flare -o build/server)
#
# We ship the prebuilt libflare_tls.so (building it needs clang + OpenSSL) and
# its OpenSSL dylibs, made relocatable via @loader_path so the server finds them
# at runtime with no pixi. Run via pixi (needs CONDA_PREFIX) AFTER `pixi run
# flare-tls`. Usage: scripts/package_engine.sh [out.zip]
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLARE="${FLARE:-$ROOT/../flare}"
OUT="${1:-$ROOT/runner.zip}"
PREFIX="${CONDA_PREFIX:?run via pixi — need CONDA_PREFIX for libflare_tls.so + OpenSSL}"
LIB="$PREFIX/lib/libflare_tls.so"
[[ -f "$LIB" ]] || { echo "error: $LIB missing — run 'pixi run flare-tls' first" >&2; exit 1; }

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/inference-server"

echo "==> staging inference-server source" >&2
# src + assets (chat template). The tokenizer + model weights are NOT bundled —
# they're generated/large and ride with the separate model download (the runner
# fetches them at runtime), so the engine bundle stays small and model-agnostic.
mkdir -p "$B/build"
cp -R "$ROOT/src" "$B/src"
cp -R "$ROOT/assets" "$B/assets"

echo "==> bundling libflare_tls.so + OpenSSL (relocatable)" >&2
cp "$LIB" "$PREFIX/lib/libssl.3.dylib" "$PREFIX/lib/libcrypto.3.dylib" "$B/build/"
# OpenSSL dylibs are already @rpath-id'd with an @loader_path rpath, so copying
# is enough. Only libflare_tls.so needs fixing: find OpenSSL beside it
# (@loader_path) and take libc++ from the OS instead of the (unshipped) conda one.
install_name_tool -delete_rpath "$PREFIX/lib" "$B/build/libflare_tls.so" 2>/dev/null || true
install_name_tool \
    -id "@rpath/libflare_tls.so" \
    -add_rpath "@loader_path" \
    -change "@rpath/libc++.1.dylib" "/usr/lib/libc++.1.dylib" \
    "$B/build/libflare_tls.so"
codesign --force --sign - "$B/build/libflare_tls.so" 2>/dev/null || true

echo "==> staging jinja2 package (from the pixi env) + flare" >&2
mkdir -p "$STAGE/pkgs" "$STAGE/flare"
if [[ -f "$PREFIX/lib/mojo/jinja2.mojoc" ]]; then
    cp "$PREFIX/lib/mojo/jinja2.mojoc" "$STAGE/pkgs/jinja2.mojoc"
else
    cp "$PREFIX/lib/mojo/jinja2.mojopkg" "$STAGE/pkgs/jinja2.mojoc"
fi
cp -R "$FLARE/flare" "$STAGE/flare/flare"

echo "==> zipping -> $OUT" >&2
rm -f "$OUT"
( cd "$STAGE" && zip -qr -X "$OUT" inference-server pkgs flare )
echo "==> done" >&2
ls -lh "$OUT" >&2
