"""Phase-3 full Qwen2.5-0.5B forward pass on the GPU + verification gate.

Assembles the verified kernels (src/kernels.mojo) into the whole model — embed →
24 decoder layers (RMSNorm → QKV → RoPE+attention → o_proj → residual; RMSNorm →
SwiGLU → residual) → final RMSNorm → tied LM head — running entirely on the M4
GPU in float32, with weights loaded from the real safetensors checkpoint.

Verifies against HF (CPU/f32, via `forward-capture`): the residual-stream hidden
state after the embedding and after every layer (per-layer comparison =
layer-bisection, max-backend §8 #2 rung 6), the final-norm output, and the
last-position logits — requiring **greedy-argmax agreement** (the next-token
decision matches). Build + run via `pixi run forward-spike`.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy
from layout import TileTensor, row_major

from kernels import (
    cvt_kernel, embed_kernel, add_kernel, rmsnorm_kernel, matmul_kernel,
    silu_mul_kernel, attn_kernel,
)

comptime H = 896
comptime NKV = 128
comptime INTER = 4864
comptime VOCAB = 151936
comptime NLAYERS = 24
comptime BLOCK = 256

comptime DevBuf = DeviceBuffer[DType.float32]


# ── safetensors header parsing (JSON subset) ──────────────────────────────────

comptime QUOTE = 34
comptime LBRACE = 123
comptime RBRACE = 125
comptime LBRACK = 91
comptime RBRACK = 93
comptime COLON = 58
comptime COMMA = 44


@fieldwise_init
struct TensorEntry(Copyable, Movable):
    var name: String
    var begin: Int
    var end: Int


def is_ws(c: Int) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13

def skip_ws(buf: List[UInt8], mut pos: Int):
    while pos < len(buf) and is_ws(Int(buf[pos])):
        pos += 1

def expect(buf: List[UInt8], mut pos: Int, ch: Int) raises:
    if pos >= len(buf) or Int(buf[pos]) != ch:
        raise Error("parse error at byte " + String(pos))
    pos += 1

def parse_string(buf: List[UInt8], mut pos: Int) raises -> String:
    expect(buf, pos, QUOTE)
    var s = String("")
    while pos < len(buf) and Int(buf[pos]) != QUOTE:
        s += chr(Int(buf[pos]))
        pos += 1
    expect(buf, pos, QUOTE)
    return s^

def parse_uint(buf: List[UInt8], mut pos: Int) raises -> Int:
    var v = 0
    var start = pos
    while pos < len(buf) and Int(buf[pos]) >= 48 and Int(buf[pos]) <= 57:
        v = v * 10 + (Int(buf[pos]) - 48)
        pos += 1
    if pos == start:
        raise Error("expected int at " + String(pos))
    return v

def parse_int_array(buf: List[UInt8], mut pos: Int) raises -> List[Int]:
    var out = List[Int]()
    expect(buf, pos, LBRACK)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACK:
        pos += 1
        return out^
    while True:
        skip_ws(buf, pos)
        out.append(parse_uint(buf, pos))
        skip_ws(buf, pos)
        if Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    expect(buf, pos, RBRACK)
    return out^

def skip_value(buf: List[UInt8], mut pos: Int) raises:
    skip_ws(buf, pos)
    var c = Int(buf[pos])
    if c == QUOTE:
        _ = parse_string(buf, pos)
    elif c == LBRACE:
        skip_object(buf, pos)
    elif c == LBRACK:
        expect(buf, pos, LBRACK)
        skip_ws(buf, pos)
        if Int(buf[pos]) == RBRACK:
            pos += 1
            return
        while True:
            skip_value(buf, pos)
            skip_ws(buf, pos)
            if Int(buf[pos]) == COMMA:
                pos += 1
                continue
            break
        expect(buf, pos, RBRACK)
    else:
        while pos < len(buf):
            var d = Int(buf[pos])
            if d == COMMA or d == RBRACE or d == RBRACK or is_ws(d):
                break
            pos += 1

def skip_object(buf: List[UInt8], mut pos: Int) raises:
    expect(buf, pos, LBRACE)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACE:
        pos += 1
        return
    while True:
        skip_ws(buf, pos)
        _ = parse_string(buf, pos)
        skip_ws(buf, pos)
        expect(buf, pos, COLON)
        skip_value(buf, pos)
        skip_ws(buf, pos)
        if Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    expect(buf, pos, RBRACE)

def parse_header(buf: List[UInt8]) raises -> List[TensorEntry]:
    var entries = List[TensorEntry]()
    var pos = 0
    skip_ws(buf, pos)
    expect(buf, pos, LBRACE)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACE:
        return entries^
    while True:
        skip_ws(buf, pos)
        var name = parse_string(buf, pos)
        skip_ws(buf, pos)
        expect(buf, pos, COLON)
        skip_ws(buf, pos)
        if name == "__metadata__":
            skip_object(buf, pos)
        else:
            expect(buf, pos, LBRACE)
            var begin = 0
            var end = 0
            skip_ws(buf, pos)
            if Int(buf[pos]) != RBRACE:
                while True:
                    skip_ws(buf, pos)
                    var fkey = parse_string(buf, pos)
                    skip_ws(buf, pos)
                    expect(buf, pos, COLON)
                    skip_ws(buf, pos)
                    if fkey == "data_offsets":
                        var offs = parse_int_array(buf, pos)
                        begin = offs[0]
                        end = offs[1]
                    else:
                        skip_value(buf, pos)
                    skip_ws(buf, pos)
                    if Int(buf[pos]) == COMMA:
                        pos += 1
                        continue
                    break
            expect(buf, pos, RBRACE)
            entries.append(TensorEntry(name, begin, end))
        skip_ws(buf, pos)
        if pos < len(buf) and Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    return entries^


# ── weights ────────────────────────────────────────────────────────────────────

@fieldwise_init
struct Weights(Movable):
    var embed: DevBuf
    var final_norm: DevBuf
    var ln1: List[DevBuf]
    var qw: List[DevBuf]
    var qb: List[DevBuf]
    var kw: List[DevBuf]
    var kb: List[DevBuf]
    var vw: List[DevBuf]
    var vb: List[DevBuf]
    var ow: List[DevBuf]
    var ln2: List[DevBuf]
    var gate: List[DevBuf]
    var up: List[DevBuf]
    var down: List[DevBuf]


def read_header(path: String) raises -> List[TensorEntry]:
    """Parse the header; returns entries with begin/end as ABSOLUTE file offsets."""
    with open(path, "r") as f:
        var lenb = f.read_bytes(8)
        var hlen: UInt64 = 0
        for i in range(8):
            hlen |= UInt64(Int(lenb[i])) << UInt64(8 * i)
        var hdr = f.read_bytes(Int(hlen)).copy()
        var entries = parse_header(hdr)
        var ds = 8 + Int(hlen)
        for i in range(len(entries)):
            entries[i].begin += ds
            entries[i].end += ds
        return entries^


def load_one(ctx: DeviceContext, path: String, begin: Int, end: Int) raises -> DevBuf:
    var nbytes = end - begin
    var count = nbytes // 2
    var dev_f32 = ctx.enqueue_create_buffer[DType.float32](count)
    with open(path, "r") as f:
        _ = f.seek(UInt64(begin))
        var raw = f.read_bytes(nbytes)
        var host = ctx.enqueue_create_host_buffer[DType.uint16](count)
        ctx.synchronize()
        memcpy(dest=host.unsafe_ptr().bitcast[UInt8](), src=raw.unsafe_ptr(), count=nbytes)
        var dev_u16 = ctx.enqueue_create_buffer[DType.uint16](count)
        ctx.enqueue_copy(dev_u16, host)
        var lay = row_major(count)
        comptime k = cvt_kernel[type_of(lay)]
        ctx.enqueue_function[k](
            TileTensor(dev_u16, lay), TileTensor(dev_f32, lay), count,
            grid_dim=ceildiv(count, BLOCK), block_dim=BLOCK,
        )
        ctx.synchronize()
    return dev_f32^


def load_named(ctx: DeviceContext, path: String,
               entries: List[TensorEntry], name2idx: Dict[String, Int], name: String) raises -> DevBuf:
    var idx = name2idx[name]
    return load_one(ctx, path, entries[idx].begin, entries[idx].end)


def load_weights(ctx: DeviceContext, path: String) raises -> Weights:
    var entries = read_header(path)
    var name2idx = Dict[String, Int]()
    for e in range(len(entries)):
        name2idx[entries[e].name] = e

    var embed = load_named(ctx, path, entries, name2idx, "model.embed_tokens.weight")
    var final_norm = load_named(ctx, path, entries, name2idx, "model.norm.weight")
    var ln1 = List[DevBuf]()
    var qw = List[DevBuf]()
    var qb = List[DevBuf]()
    var kw = List[DevBuf]()
    var kb = List[DevBuf]()
    var vw = List[DevBuf]()
    var vb = List[DevBuf]()
    var ow = List[DevBuf]()
    var ln2 = List[DevBuf]()
    var gate = List[DevBuf]()
    var up = List[DevBuf]()
    var down = List[DevBuf]()
    for l in range(NLAYERS):
        var p = "model.layers." + String(l) + "."
        ln1.append(load_named(ctx, path, entries, name2idx, p + "input_layernorm.weight"))
        qw.append(load_named(ctx, path, entries, name2idx, p + "self_attn.q_proj.weight"))
        qb.append(load_named(ctx, path, entries, name2idx, p + "self_attn.q_proj.bias"))
        kw.append(load_named(ctx, path, entries, name2idx, p + "self_attn.k_proj.weight"))
        kb.append(load_named(ctx, path, entries, name2idx, p + "self_attn.k_proj.bias"))
        vw.append(load_named(ctx, path, entries, name2idx, p + "self_attn.v_proj.weight"))
        vb.append(load_named(ctx, path, entries, name2idx, p + "self_attn.v_proj.bias"))
        ow.append(load_named(ctx, path, entries, name2idx, p + "self_attn.o_proj.weight"))
        ln2.append(load_named(ctx, path, entries, name2idx, p + "post_attention_layernorm.weight"))
        gate.append(load_named(ctx, path, entries, name2idx, p + "mlp.gate_proj.weight"))
        up.append(load_named(ctx, path, entries, name2idx, p + "mlp.up_proj.weight"))
        down.append(load_named(ctx, path, entries, name2idx, p + "mlp.down_proj.weight"))
    return Weights(
        embed^, final_norm^, ln1^, qw^, qb^, kw^, kb^, vw^, vb^, ow^, ln2^,
        gate^, up^, down^,
    )


# ── op helpers (each launches one kernel, returns a new device buffer) ─────────

def mm(ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, mut b: DevBuf,
       M: Int, K: Int, N: Int, use_bias: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    comptime k = matmul_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
        TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
        M, K, N, use_bias,
        grid_dim=ceildiv(M * N, BLOCK), block_dim=BLOCK,
    )
    return y^

def rms(ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, T: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](T * H)
    var lay = row_major(T * H)
    comptime k = rmsnorm_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, lay), TileTensor(w, row_major(H)), TileTensor(y, lay), T, H,
        grid_dim=ceildiv(T, 64), block_dim=64,
    )
    return y^

def add(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = add_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def silumul(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = silu_mul_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def attention(ctx: DeviceContext, mut q: DevBuf, mut k: DevBuf, mut v: DevBuf, T: Int) raises -> DevBuf:
    var o = ctx.enqueue_create_buffer[DType.float32](T * H)
    var lay = row_major(T * H)
    comptime kk = attn_kernel[type_of(lay)]
    ctx.enqueue_function[kk](
        TileTensor(q, row_major(T * H)), TileTensor(k, row_major(T * NKV)),
        TileTensor(v, row_major(T * NKV)), TileTensor(o, lay), T,
        grid_dim=ceildiv(T * 14, 14), block_dim=14,
    )
    return o^

def layer(ctx: DeviceContext, mut w: Weights, l: Int, mut h: DevBuf, mut dummy: DevBuf, T: Int) raises -> DevBuf:
    var ln1 = rms(ctx, h, w.ln1[l], T)
    var q = mm(ctx, ln1, w.qw[l], w.qb[l], T, H, H, 1)
    var k = mm(ctx, ln1, w.kw[l], w.kb[l], T, H, NKV, 1)
    var v = mm(ctx, ln1, w.vw[l], w.vb[l], T, H, NKV, 1)
    var a = attention(ctx, q, k, v, T)
    var o = mm(ctx, a, w.ow[l], dummy, T, H, H, 0)
    var h2 = add(ctx, h, o, T * H)
    var ln2 = rms(ctx, h2, w.ln2[l], T)
    var g = mm(ctx, ln2, w.gate[l], dummy, T, H, INTER, 0)
    var u = mm(ctx, ln2, w.up[l], dummy, T, H, INTER, 0)
    var gu = silumul(ctx, g, u, T * INTER)
    var dn = mm(ctx, gu, w.down[l], dummy, T, INTER, H, 0)
    var h3 = add(ctx, h2, dn, T * H)
    return h3^


# ── fixtures + comparison ──────────────────────────────────────────────────────

def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

def read_f32(path: String) raises -> List[Float32]:
    var out = List[Float32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var p = raw.unsafe_ptr().bitcast[Float32]()
        for i in range(len(raw) // 4):
            out.append(p[i])
    return out^

def read_i32(path: String) raises -> List[Int32]:
    var out = List[Int32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var p = raw.unsafe_ptr().bitcast[Int32]()
        for i in range(len(raw) // 4):
            out.append(p[i])
    return out^

def max_abs(mut dev: DevBuf, expected: List[Float32]) raises -> Float32:
    var n = len(expected)
    var worst = Float32(0.0)
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(len(expected)))
        for i in range(n):
            var d = abs(rebind[Scalar[DType.float32]](mt[i]) - expected[i])
            if d > worst:
                worst = d
    return worst

def argmax_dev_lastrow(mut dev: DevBuf, T: Int) raises -> Int:
    var best = -1
    var best_v = Float32(-1.0e30)
    var base = (T - 1) * VOCAB
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(T * VOCAB))
        for i in range(VOCAB):
            var v = rebind[Scalar[DType.float32]](mt[base + i])
            if v > best_v:
                best_v = v
                best = i
    return best

def argmax_list(a: List[Float32]) -> Int:
    var best = -1
    var best_v = Float32(-1.0e30)
    for i in range(len(a)):
        if a[i] > best_v:
            best_v = a[i]
            best = i
    return best


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var dir = "tests/fixtures/forward/"
    var meta = read_text(dir + "meta.txt").split("\n")
    var T = Int(atol(String(meta[0]).strip()))
    var ckpt = String(String(meta[1]).strip())
    print("forward spike — T=", T, " loading weights…", sep="")

    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)

    # ids -> device int32
    var ids_host = read_i32(dir + "ids.bin")
    var ids_dev = ctx.enqueue_create_buffer[DType.int32](T)
    with ids_dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(T))
        for i in range(T):
            mt[i] = rebind[mt.ElementType](ids_host[i])

    # embed
    var h = ctx.enqueue_create_buffer[DType.float32](T * H)
    var elay = row_major(T * H)
    comptime ek = embed_kernel[type_of(elay)]
    ctx.enqueue_function[ek](
        TileTensor(ids_dev, row_major(T)), TileTensor(w.embed, row_major(VOCAB * H)),
        TileTensor(h, elay), T, H,
        grid_dim=ceildiv(T * H, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()

    var all_ok = True
    var worst_layer = max_abs(h, read_f32(dir + "embed.bin"))
    print("  embed        max_abs=", worst_layer)

    for l in range(NLAYERS):
        h = layer(ctx, w, l, h, dummy, T)
        ctx.synchronize()
        var ma = max_abs(h, read_f32(dir + "layer_" + String(l) + ".bin"))
        if ma > worst_layer:
            worst_layer = ma
        if ma > 5.0e-2:
            print("  layer ", l, "   max_abs=", ma, "  <-- large", sep="")
            all_ok = False

    var hn = rms(ctx, h, w.final_norm, T)
    ctx.synchronize()
    var ma_norm = max_abs(hn, read_f32(dir + "final_norm.bin"))
    print("  final_norm   max_abs=", ma_norm)

    var logits = mm(ctx, hn, w.embed, dummy, T, H, VOCAB, 0)
    ctx.synchronize()
    var ref_last = read_f32(dir + "logits_last.bin")
    var gpu_am = argmax_dev_lastrow(logits, T)
    var ref_am = argmax_list(ref_last)

    print("  worst per-layer hidden max_abs=", worst_layer)
    print("  argmax  gpu=", gpu_am, "  ref=", ref_am, sep="")
    var argmax_ok = gpu_am == ref_am
    if not argmax_ok:
        all_ok = False

    if not all_ok:
        raise Error("forward pass mismatch — Phase 3 FAILED")
    print("OK — GPU forward matches HF/CPU per layer; greedy next-token argmax agrees (", gpu_am, ")", sep="")
