"""Per-layer validation gate for the Gemma 4 12B-it text decoder.

For layer L in {0 (sliding), 5 (full-attention)}: load just that layer's bf16
weights from the model, upload the committed f32 reference input hidden state
([1,12,3840]), run `gemma_layer(L)` (prefill Tq=12, q_offset=0 — seq 12 < window
1024 so sliding == full-causal), and compare to the committed reference output.
Target max|Δ| < 1e-2 (bf16 weights vs the f32 transformers reference).

Build/run self-contained:
  pixi run mojo build tests/manual/gemma_layer_test.mojo -I src -o build/gemma_layer_test
  ./build/gemma_layer_test
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer

from models.gemma import (
    load_gemma_weights,
    gemma_layer,
    G_HIDDEN,
    SL_NKV,
    FU_NKV,
    _is_full_layer,
)
from testio import read_f32, max_abs

comptime DevBuf = DeviceBuffer[DType.float32]
comptime CKPT = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"
comptime FIX = "tests/fixtures/gemma/"
comptime S = 12


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var ctx = DeviceContext()
    var want = List[Int]()
    want.append(0)
    want.append(5)
    print("loading Gemma layers 0 + 5 (bf16)…")
    var w = load_gemma_weights(ctx, CKPT, want)
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)

    var all_ok = True
    for li in range(2):
        var L = 0 if li == 0 else 5
        var full = _is_full_layer(L)
        var nkv = FU_NKV if full else SL_NKV

        # upload the reference input [1,12,3840]
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
        var tag = String("full") if full else String("sliding")
        print("  layer ", L, " (", tag, ")  max|Δ|=", ma, sep="")
        if ma > 1.0e-2:
            all_ok = False
            print("    ^-- FAIL (>1e-2)")

    if not all_ok:
        raise Error("Gemma per-layer validation FAILED")
    print("OK — Gemma layers 0 + 5 match the reference within 1e-2")
