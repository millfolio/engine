"""Gate: full forward pass vs HF/CPU, per-layer (ARCHITECTURE.md §6 Phase 3, §11 #7).

Drives the model library through prefill and compares the residual-stream hidden
state after the embedding and each layer, the final norm, and the last-position
argmax against HF (`pixi run forward-capture`). `pixi run test-forward`.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer

from model import (
    Weights,
    load_weights,
    embed_tokens,
    layer_cached,
    rmsnorm,
    mm,
    upload_ids,
)
from testio import (
    read_text,
    read_f32,
    read_i32,
    max_abs,
    argmax_row,
    argmax_list,
)

comptime H = 896
comptime NKV = 128
comptime VOCAB = 151936
comptime NLAYERS = 24
comptime DevBuf = DeviceBuffer[DType.float32]


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var dir = "tests/fixtures/forward/"
    var meta = read_text(dir + "meta.txt").split("\n")
    var T = Int(atol(String(meta[0]).strip()))
    var ckpt = String(String(meta[1]).strip())
    print("forward gate — T=", T, " loading weights…", sep="")

    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)

    var cache_len = T * NKV
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(NLAYERS):
        kcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
        vcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))

    var prompt = List[Int]()
    var ids32 = read_i32(dir + "ids.bin")
    for i in range(len(ids32)):
        prompt.append(Int(ids32[i]))
    var ids_dev = upload_ids(ctx, prompt)

    var h = embed_tokens(ctx, ids_dev, w.embed, T, w.hidden, w.vocab)
    ctx.synchronize()
    var all_ok = True
    var worst = max_abs(h, read_f32(dir + "embed.bin"))
    print("  embed        max_abs=", worst)

    for l in range(NLAYERS):
        h = layer_cached(ctx, w, l, h, kcs[l], vcs[l], T, 0, cache_len, dummy)
        ctx.synchronize()
        var ma = max_abs(h, read_f32(dir + "layer_" + String(l) + ".bin"))
        if ma > worst:
            worst = ma
        if ma > 5.0e-2:
            print("  layer ", l, " max_abs=", ma, "  <-- large", sep="")
            all_ok = False

    var hn = rmsnorm(ctx, h, w.final_norm, T, H)
    ctx.synchronize()
    print(
        "  final_norm   max_abs=", max_abs(hn, read_f32(dir + "final_norm.bin"))
    )

    var logits = mm(ctx, hn, w.embed, dummy, T, H, VOCAB, 0)
    ctx.synchronize()
    var gpu_am = argmax_row(logits, T - 1, VOCAB)
    var ref_am = argmax_list(read_f32(dir + "logits_last.bin"))
    print("  worst per-layer max_abs=", worst)
    print("  argmax  gpu=", gpu_am, "  ref=", ref_am, sep="")
    if gpu_am != ref_am:
        all_ok = False

    if not all_ok:
        raise Error("forward pass mismatch — gate FAILED")
    print(
        "OK — GPU forward matches HF per layer; greedy argmax agrees (",
        gpu_am,
        ")",
        sep="",
    )
