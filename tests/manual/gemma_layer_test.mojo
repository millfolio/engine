"""Per-layer numeric validation gate for the Gemma 4 12B-it dense decoder.

For every layer L that has a captured reference fixture (tests/fixtures/gemma/
layer<L>_in.bin / _out.bin, produced by `pixi run gemma-ref`), load ONLY that
layer's bf16 weights, upload the reference input hidden state ([1,S,3840]), run
`gemma_layer(L)` (prefill Tq=S, q_offset=0 — seq S < window 1024 so sliding ==
full-causal), and compare to the reference output. Loading one layer at a time
keeps this tiny in memory: the full bf16 12B is ~24 GB, but the HF reference is
captured SEPARATELY (gemma4_layer_ref.py), so the two 12B-sized things are never
co-resident — this is the whole point of the dump-and-compare split.

Feeding each layer the REFERENCE input (not the previous layer's Mojo output)
means errors don't accumulate, so a mismatch pinpoints the exact broken layer.

Target max|Δ| < 1e-2 (bf16 weights vs the f32 transformers reference).

Build/run:  pixi run gemma-layer-test
"""

from std.sys import has_accelerator
from std.os.path import exists
from max.gpu.host import DeviceContext

from models.gemma import (
    load_gemma_weights,
    gemma_layer,
    G_HIDDEN,
    G_NLAYERS,
    SL_NKV,
    FU_NKV,
    _is_full_layer,
)
from testio import read_f32, max_abs

comptime CKPT = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"
comptime FIX = "tests/fixtures/gemma/"
comptime S = 12
comptime TOL = Float32(1.0e-2)


def _check_layer(
    ctx: DeviceContext, L: Int, mut fails: Int, mut tested: Int
) raises:
    """Validate one decoder layer against its captured reference in isolation."""
    var full = _is_full_layer(L)
    var nkv = FU_NKV if full else SL_NKV
    # Load ONLY this layer's bf16 weights (the rest get size-1 placeholders).
    var w = load_gemma_weights(ctx, CKPT, [L])
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)

    # Upload the reference input hidden [1, S, 3840].
    var inp = read_f32(FIX + "layer" + String(L) + "_in.bin")
    var h = ctx.enqueue_create_buffer[DType.float32](S * G_HIDDEN)
    with h.map_to_host() as m:
        for i in range(S * G_HIDDEN):
            m[i] = inp[i]

    var cache_len = S * nkv
    var kc = ctx.enqueue_create_buffer[DType.float32](cache_len)
    var vc = ctx.enqueue_create_buffer[DType.float32](cache_len)

    var out = gemma_layer(ctx, w, L, h, kc, vc, S, 0, cache_len, dummy)
    ctx.synchronize()

    var expected = read_f32(FIX + "layer" + String(L) + "_out.bin")
    var ma = max_abs(out, expected)
    var tag = String("full   ") if full else String("sliding")
    tested += 1
    var status = String("ok") if ma <= TOL else String("FAIL <-- >1e-2")
    print("  L", L, " (", tag, ")  max|Δ|=", ma, "  ", status, sep="")
    if ma > TOL:
        fails += 1


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var ctx = DeviceContext()
    var fails = 0
    var tested = 0
    print("Gemma 12B per-layer validation (bf16 weights, one layer at a time)…")
    for L in range(G_NLAYERS):
        if not exists(FIX + "layer" + String(L) + "_in.bin"):
            continue
        _check_layer(ctx, L, fails, tested)

    if tested == 0:
        raise Error(
            "no fixtures found under " + FIX + " — run `pixi run gemma-ref` first"
        )
    if fails > 0:
        raise Error(
            String(fails)
            + "/"
            + String(tested)
            + " layers FAILED (max|Δ| > 1e-2)"
        )
    print("OK —", tested, "layers match the reference within 1e-2")
