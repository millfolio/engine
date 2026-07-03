#!/usr/bin/env bash
#
# Build runner.zip — the engine ("runner") bundle the Millfolio app downloads.
# The engine binaries are now shipped PREBUILT (built here in CI, not `mojo build`d
# on-device — mirrors vault/core/scripts/package_millfolio.sh). The bundle unzips
# to one dir:
#
#   inference-server/   assets +
#                   build/{server, download}                (prebuilt engine binaries)
#                   build/{libflare_tls.so + libssl.3 + libcrypto.3, rpath-fixed}
#
# so the app runs build/server directly — no on-device source build, no `.mojo`
# source shipped for the engine.
#
# The two binaries are built here with the SAME `mojo build` invocations the
# installer used, their rpath relocated to a device-relative @loader_path (the CI
# machine's $CONDA_PREFIX/lib doesn't exist on the user's box), and ad-hoc signed.
# The engine runs with CONDA_PREFIX UNSET and dlopens build/libflare_tls.so (a
# relative path — see flare.utils.dylib), so the shim needs no rpath; only the
# Mojo runtime dylibs (linked via @rpath) resolve from the toolchain's mojo/lib.
#
# We ship the prebuilt libflare_tls.so (building it needs clang + OpenSSL) and
# its OpenSSL dylibs, made relocatable via @loader_path so the server finds them
# at runtime with no pixi. Run via pixi (needs CONDA_PREFIX for the toolchain +
# libflare_tls.so) AFTER `pixi run flare-tls`. Usage: scripts/package_engine.sh [out.zip]
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JINJA2="${JINJA2:-$ROOT/../jinja2.mojo}"
FLARE="${FLARE:-$ROOT/../flare}"
OUT="${1:-$ROOT/runner.zip}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac   # zip runs from a temp dir — need absolute
PREFIX="${CONDA_PREFIX:?run via pixi — need CONDA_PREFIX for libflare_tls.so + OpenSSL}"
MOJO="${MOJO:-mojo}"
LIB="$PREFIX/lib/libflare_tls.so"
[[ -f "$LIB" ]] || { echo "error: $LIB missing — run 'pixi run flare-tls' first" >&2; exit 1; }

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/inference-server"

echo "==> staging inference-server assets" >&2
# assets (chat template) only. The tokenizer + model weights are NOT bundled —
# they're generated/large and ride with the separate model download (the runner
# fetches them at runtime), so the engine bundle stays small and model-agnostic.
# No `.mojo` source ships: the binaries below are built here, not on-device.
mkdir -p "$B/build"
cp -R "$ROOT/assets" "$B/assets"

echo "==> building prebuilt engine binaries (server + download)" >&2
# The SAME invocations the on-device installer used (Bootstrapper.installServer).
"$MOJO" build "$ROOT/src/server.mojo"   -I "$JINJA2/src" -I "$FLARE" -o "$B/build/server"
"$MOJO" build "$ROOT/src/download.mojo" -I "$FLARE"                  -o "$B/build/download"

# Relocate each binary's rpath. `mojo build` bakes in the BUILD machine's
# $CONDA_PREFIX/lib (the CI runner's), which is absent on the user's box — so the
# Mojo runtime dylibs it links via @rpath (libKGENCompilerRTShared.dylib, …) fail
# to load. The on-device layout is fixed: the binary lands at
# <support>/bundle/runner/inference-server/build/<bin> and the toolchain at
# <support>/mojo/lib, so a @loader_path-relative rpath (4 dirs up to <support>,
# then mojo/lib) resolves on any machine. The engine runs with CONDA_PREFIX unset
# and dlopens build/libflare_tls.so by relative path, so no rpath is needed for
# the shim — only the Mojo runtime libs. Ad-hoc sign (matches package_millfolio).
# The server keeps the "millfolio" signing identifier so the LaunchAgent's
# background-app notification / Login Items entry read "millfolio" (what the old
# on-device signServerIdentity did).
install_name_tool -delete_rpath "$PREFIX/lib" "$B/build/server"   2>/dev/null || true
install_name_tool -add_rpath "@loader_path/../../../../mojo/lib" "$B/build/server"   2>/dev/null || true
codesign --force --sign - --identifier millfolio "$B/build/server" 2>/dev/null || true
install_name_tool -delete_rpath "$PREFIX/lib" "$B/build/download" 2>/dev/null || true
install_name_tool -add_rpath "@loader_path/../../../../mojo/lib" "$B/build/download" 2>/dev/null || true
codesign --force --sign - "$B/build/download" 2>/dev/null || true

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

# jinja2.mojo + flare are no longer shipped: they were only on the include path
# for the on-device build, which now happens here. Nothing references them at
# runtime (flare's FFI is the prebuilt libflare_tls.so, dlopen'd by path).

echo "==> zipping -> $OUT" >&2
rm -f "$OUT"
( cd "$STAGE" && zip -qr -X "$OUT" inference-server )
echo "==> done — bundle contains ONLY assets + prebuilt binaries + FFI shims (no .mojo)" >&2
ls -lh "$OUT" >&2
