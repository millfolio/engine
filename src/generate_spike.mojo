"""Phase-4 greedy decode loop + KV cache (ARCHITECTURE.md §5.4–5.5, §6).

Turns the verified forward pass into generated text: prefill the prompt (filling
a per-layer KV cache), then decode one token at a time — each step embeds the
last token, runs the 24 layers attending over the growing cache, takes the
argmax, and appends — until EOS or max_tokens. Checked **token-for-token** vs HF
greedy generation (`generate-capture`).

The cache stores raw (pre-RoPE) K/V at row = absolute position; attn_cached_kernel
applies RoPE using the row index, so a decode step is O(positions), not O(T²).
Reuses the weights loader and op helpers from forward_spike. Build+run via
`pixi run generate-spike`.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import embed_kernel, attn_cached_kernel, copy_kernel
from forward_spike import Weights, load_weights, mm, rms, add, silumul, read_text, read_i32

comptime H = 896
comptime NKV = 128
comptime INTER = 4864
comptime VOCAB = 151936
comptime NLAYERS = 24
comptime BLOCK = 256
comptime EOS1 = 151645
comptime EOS2 = 151643

comptime DevBuf = DeviceBuffer[DType.float32]


def embed_tokens(ctx: DeviceContext, mut ids_dev: DeviceBuffer[DType.int32], mut emb: DevBuf, T: Int) raises -> DevBuf:
    var h = ctx.enqueue_create_buffer[DType.float32](T * H)
    var lay = row_major(T * H)
    comptime k = embed_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(ids_dev, row_major(T)), TileTensor(emb, row_major(VOCAB * H)),
        TileTensor(h, lay), T, H,
        grid_dim=ceildiv(T * H, BLOCK), block_dim=BLOCK,
    )
    return h^


def copy_into(ctx: DeviceContext, mut src: DevBuf, mut dst: DevBuf, dst_offset: Int, n: Int, dst_len: Int) raises:
    var lay = row_major(n)
    comptime k = copy_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, lay), TileTensor(dst, row_major(dst_len)), dst_offset, n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )


def attn_cached(ctx: DeviceContext, mut q: DevBuf, mut kc: DevBuf, mut vc: DevBuf,
                Tq: Int, q_offset: Int, cache_len: Int) raises -> DevBuf:
    var o = ctx.enqueue_create_buffer[DType.float32](Tq * H)
    var lay = row_major(Tq * H)
    comptime k = attn_cached_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(q, row_major(Tq * H)), TileTensor(kc, row_major(cache_len)),
        TileTensor(vc, row_major(cache_len)), TileTensor(o, lay), Tq, q_offset,
        grid_dim=ceildiv(Tq * 14, 14), block_dim=14,
    )
    return o^


def layer_cached(ctx: DeviceContext, mut w: Weights, l: Int, mut h: DevBuf,
                 mut kc: DevBuf, mut vc: DevBuf, Tq: Int, q_offset: Int,
                 cache_len: Int, mut dummy: DevBuf) raises -> DevBuf:
    var ln1 = rms(ctx, h, w.ln1[l], Tq)
    var q = mm(ctx, ln1, w.qw[l], w.qb[l], Tq, H, H, 1)
    var kk = mm(ctx, ln1, w.kw[l], w.kb[l], Tq, H, NKV, 1)
    var vv = mm(ctx, ln1, w.vw[l], w.vb[l], Tq, H, NKV, 1)
    copy_into(ctx, kk, kc, q_offset * NKV, Tq * NKV, cache_len)
    copy_into(ctx, vv, vc, q_offset * NKV, Tq * NKV, cache_len)
    var o = attn_cached(ctx, q, kc, vc, Tq, q_offset, cache_len)
    var o2 = mm(ctx, o, w.ow[l], dummy, Tq, H, H, 0)
    var h2 = add(ctx, h, o2, Tq * H)
    var ln2 = rms(ctx, h2, w.ln2[l], Tq)
    var g = mm(ctx, ln2, w.gate[l], dummy, Tq, H, INTER, 0)
    var u = mm(ctx, ln2, w.up[l], dummy, Tq, H, INTER, 0)
    var gu = silumul(ctx, g, u, Tq * INTER)
    var dn = mm(ctx, gu, w.down[l], dummy, Tq, INTER, H, 0)
    return add(ctx, h2, dn, Tq * H)


def argmax_last(ctx: DeviceContext, mut w: Weights, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> Int:
    var hn = rms(ctx, h, w.final_norm, T)
    var logits = mm(ctx, hn, w.embed, dummy, T, H, VOCAB, 0)
    ctx.synchronize()
    var base = (T - 1) * VOCAB
    var best = -1
    var best_v = Float32(-1.0e30)
    with logits.map_to_host() as m:
        var mt = TileTensor(m, row_major(T * VOCAB))
        for i in range(VOCAB):
            var v = rebind[Scalar[DType.float32]](mt[base + i])
            if v > best_v:
                best_v = v
                best = i
    return best


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var dir = "tests/fixtures/generate/"
    var lines = read_text(dir + "expected.txt").split("\n")
    var P = Int(atol(String(lines[0]).strip()))
    var ckpt = String(String(lines[1]).strip())
    var expected = List[Int]()
    for t in String(lines[2]).split(" "):
        var ts = String(t).strip()
        if ts.byte_length() > 0:
            expected.append(Int(atol(ts)))
    var max_new = len(expected)
    print("generate spike — P=", P, " max_new=", max_new, "; loading weights…", sep="")

    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)

    var max_seq = P + max_new + 2
    var cache_len = max_seq * NKV
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(NLAYERS):
        var kc = ctx.enqueue_create_buffer[DType.float32](cache_len)
        var vc = ctx.enqueue_create_buffer[DType.float32](cache_len)
        kcs.append(kc^)
        vcs.append(vc^)

    # prompt ids -> device int32
    var prompt = read_i32(dir + "prompt_ids.bin")
    var ids_dev = ctx.enqueue_create_buffer[DType.int32](P)
    with ids_dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(P))
        for i in range(P):
            mt[i] = rebind[mt.ElementType](prompt[i])

    # prefill
    var h = embed_tokens(ctx, ids_dev, w.embed, P)
    for l in range(NLAYERS):
        h = layer_cached(ctx, w, l, h, kcs[l], vcs[l], P, 0, cache_len, dummy)
    var nxt = argmax_last(ctx, w, h, P, dummy)

    var gen = List[Int]()
    gen.append(nxt)
    var pos = P

    while len(gen) < max_new and nxt != EOS1 and nxt != EOS2:
        var one = ctx.enqueue_create_buffer[DType.int32](1)
        with one.map_to_host() as m:
            var mt = TileTensor(m, row_major(1))
            mt[0] = rebind[mt.ElementType](Int32(nxt))
        var h1 = embed_tokens(ctx, one, w.embed, 1)
        for l in range(NLAYERS):
            h1 = layer_cached(ctx, w, l, h1, kcs[l], vcs[l], 1, pos, cache_len, dummy)
        nxt = argmax_last(ctx, w, h1, 1, dummy)
        pos += 1
        gen.append(nxt)

    # compare
    var ok = len(gen) == len(expected)
    var n = len(gen) if len(gen) < len(expected) else len(expected)
    for i in range(n):
        if gen[i] != expected[i]:
            ok = False
    var gs = String("")
    var es = String("")
    for i in range(len(gen)):
        gs += String(gen[i]) + " "
    for i in range(len(expected)):
        es += String(expected[i]) + " "
    print("  gpu gen: ", gs, sep="")
    print("  hf  ref: ", es, sep="")

    if not ok:
        raise Error("greedy decode does NOT match HF token-for-token — Phase 4 FAILED")
    print("OK — Mojo greedy decode matches HF token-for-token (", len(gen), " tokens)", sep="")
