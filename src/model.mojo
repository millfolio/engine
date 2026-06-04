"""Native-Mojo Qwen2.5-0.5B-Instruct inference library (ARCHITECTURE.md §3, §5).

Everything needed to run the model on the GPU, independent of any test harness:
  - safetensors header parsing + bf16→f32 weight loading to device buffers
  - the op launchers (matmul+bias, RMSNorm, residual add, SwiGLU's silu·mul,
    embedding gather, KV-cached attention) over src/kernels.mojo
  - one decoder layer (`layer_cached`, used for both prefill at q_offset=0 and
    single-token decode), `argmax_last`, and greedy `generate`

Hardcoded to Qwen2.5-0.5B (ARCHITECTURE.md §2). Verified by the test_*.mojo gates.
"""

from std.math import ceildiv, exp
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from kernels import (
    cvt_kernel, embed_kernel, add_kernel, rmsnorm_kernel, matmul_kernel,
    matmul_tiled_kernel, matmul_simd_kernel, SG_BM, SG_BN, SG_TPB, slice_row_kernel,
    silu_mul_kernel, attn_cached_kernel, flash_attn_kernel, FLASH_PW,
    copy_kernel, rope_k_kernel, rope_q_kernel,
)

# Above this context length (keys = q_offset + Tq) the f32 KV working set spills
# the M4 system cache and attn_cached_kernel super-cliffs; flash_attn_kernel
# (shared-memory K/V staging, bit-identical output) wins past the ~20K crossover.
comptime FLASH_THRESHOLD = 20480
comptime HQ = 14

comptime H = 896
comptime NKV = 128
comptime HKV = 2
comptime INTER = 4864
comptime VOCAB = 151936
comptime NLAYERS = 24
comptime BLOCK = 256
comptime EOS1 = 151645
comptime EOS2 = 151643

comptime DevBuf = DeviceBuffer[DType.float32]
comptime WBuf = DeviceBuffer[DType.uint16]   # bf16 weights kept on-device as raw u16


# ── safetensors header (JSON subset) ──────────────────────────────────────────

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

def read_header(path: String) raises -> List[TensorEntry]:
    """Parse the header; entries' begin/end are ABSOLUTE file offsets."""
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


# ── weight loading (bf16 → f32 on device) ─────────────────────────────────────

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

def load_named(ctx: DeviceContext, paths: List[String], entries: List[TensorEntry],
               name2idx: Dict[String, Int], name: String) raises -> DevBuf:
    var idx = name2idx[name]
    return load_one(ctx, paths[idx], entries[idx].begin, entries[idx].end)


def load_one_bf16(ctx: DeviceContext, path: String, begin: Int, end: Int) raises -> WBuf:
    """Load a bf16 tensor to device *without* widening to f32 — the matmul/embed
    kernels widen per element (bf16_widen), halving weight read traffic (§11 #12).
    The raw safetensors bytes are already bf16, so this is a plain upload."""
    var nbytes = end - begin
    var count = nbytes // 2
    var dev_u16 = ctx.enqueue_create_buffer[DType.uint16](count)
    with open(path, "r") as f:
        _ = f.seek(UInt64(begin))
        var raw = f.read_bytes(nbytes)
        var host = ctx.enqueue_create_host_buffer[DType.uint16](count)
        ctx.synchronize()
        memcpy(dest=host.unsafe_ptr().bitcast[UInt8](), src=raw.unsafe_ptr(), count=nbytes)
        ctx.enqueue_copy(dev_u16, host)
        ctx.synchronize()
    return dev_u16^

def load_named_bf16(ctx: DeviceContext, paths: List[String], entries: List[TensorEntry],
                    name2idx: Dict[String, Int], name: String) raises -> WBuf:
    var idx = name2idx[name]
    return load_one_bf16(ctx, paths[idx], entries[idx].begin, entries[idx].end)


def _str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _parse_shard_names(buf: List[UInt8]) raises -> List[String]:
    """Distinct shard filenames from a safetensors `model.safetensors.index.json`
    weight_map. Reuses the tiny safetensors-header JSON helpers (no minja2 dep, so
    model.mojo still builds without the -I include the tests use)."""
    var names = List[String]()
    var pos = 0
    skip_ws(buf, pos)
    expect(buf, pos, LBRACE)
    skip_ws(buf, pos)
    if Int(buf[pos]) == RBRACE:
        return names^
    while True:
        skip_ws(buf, pos)
        var key = parse_string(buf, pos)
        skip_ws(buf, pos)
        expect(buf, pos, COLON)
        skip_ws(buf, pos)
        if key == "weight_map":
            expect(buf, pos, LBRACE)
            skip_ws(buf, pos)
            if Int(buf[pos]) != RBRACE:
                while True:
                    skip_ws(buf, pos)
                    _ = parse_string(buf, pos)          # tensor name (ignored)
                    skip_ws(buf, pos)
                    expect(buf, pos, COLON)
                    skip_ws(buf, pos)
                    var shard = parse_string(buf, pos)  # shard filename
                    var seen = False
                    for i in range(len(names)):
                        if names[i] == shard:
                            seen = True
                            break
                    if not seen:
                        names.append(shard)
                    skip_ws(buf, pos)
                    if Int(buf[pos]) == COMMA:
                        pos += 1
                        continue
                    break
            expect(buf, pos, RBRACE)
        else:
            skip_value(buf, pos)
        skip_ws(buf, pos)
        if pos < len(buf) and Int(buf[pos]) == COMMA:
            pos += 1
            continue
        break
    return names^


def gather_tensors(path: String) raises -> Tuple[List[TensorEntry], List[String]]:
    """Resolve a checkpoint into (entries, per-entry file path). `path` is either a
    single .safetensors file (0.5B in the HF cache is one blob) or a directory
    holding sharded shards + model.safetensors.index.json (3B). Detection: try to
    open the index inside `path`-as-dir; absent → treat `path` as a single file."""
    var entries = List[TensorEntry]()
    var paths = List[String]()
    var shards = List[String]()
    var sharded = False
    try:
        with open(path + "/model.safetensors.index.json", "r") as f:
            shards = _parse_shard_names(_str_bytes(f.read()))
        sharded = True
    except:
        pass
    if sharded:
        for si in range(len(shards)):
            var sp = path + "/" + shards[si]
            var se = read_header(sp)
            for e in range(len(se)):
                entries.append(se[e].copy())
                paths.append(sp)
    else:
        var se = read_header(path)
        for e in range(len(se)):
            entries.append(se[e].copy())
            paths.append(path)
    return (entries^, paths^)


@fieldwise_init
struct Weights(Movable):
    var embed: WBuf            # bf16 — used as both embedding table and (tied) lm-head
    var final_norm: DevBuf
    var ln1: List[DevBuf]
    var qw: List[WBuf]
    var qb: List[DevBuf]
    var kw: List[WBuf]
    var kb: List[DevBuf]
    var vw: List[WBuf]
    var vb: List[DevBuf]
    var ow: List[WBuf]
    var ln2: List[DevBuf]
    var gate: List[WBuf]
    var up: List[WBuf]
    var down: List[WBuf]
    # Architecture dims, auto-detected from the checkpoint (see load_weights).
    # `arch` selects the comptime head-kernel instantiation: 0 = 0.5B, 1 = 3B.
    var arch: Int
    var nlayers: Int
    var hidden: Int       # H
    var inter: Int        # INTER (MLP)
    var nkv: Int          # HKV * HEAD_DIM (K/V row width)
    var hq: Int           # query heads
    var hkv: Int          # kv heads
    var head_dim: Int
    var vocab: Int
    # Set once at startup by probe_simd_gemm: use the simdgroup-matrix GEMM for
    # prefill if this toolchain accepts the AIR intrinsics, else the scalar path.
    var simd_ok: Bool


def _hidden_size(entries: List[TensorEntry], name2idx: Dict[String, Int]) raises -> Int:
    """Hidden size = the width of an RMSNorm weight ([hidden], bf16 → /2 bytes)."""
    var idx = name2idx["model.layers.0.input_layernorm.weight"]
    return (entries[idx].end - entries[idx].begin) // 2


def load_weights(ctx: DeviceContext, path: String) raises -> Weights:
    var gathered = gather_tensors(path)
    var entries = gathered[0].copy()
    var paths = gathered[1].copy()
    var name2idx = Dict[String, Int]()
    for e in range(len(entries)):
        name2idx[entries[e].name] = e

    # Auto-detect which Qwen2.5 this checkpoint is from its hidden size, and pick
    # the matching dims. Both sizes share vocab/θ/ε/tokenizer/template and tie the
    # LM head, so only these dims differ. arch selects the comptime head kernels.
    var hidden = _hidden_size(entries, name2idx)
    var arch = 0
    var hq = HQ
    var hkv = HKV
    var head_dim = 64
    var inter = INTER
    var nlayers = NLAYERS
    if hidden == 2048:           # Qwen2.5-3B
        arch = 1
        hq = 16
        hkv = 2
        head_dim = 128
        inter = 11008
        nlayers = 36
    elif hidden != 896:          # Qwen2.5-0.5B is the default preset above
        raise Error(
            "unsupported hidden size " + String(hidden)
            + " (expected 896 = Qwen2.5-0.5B or 2048 = Qwen2.5-3B)"
        )
    var nkv = hkv * head_dim

    var embed = load_named_bf16(ctx, paths, entries, name2idx, "model.embed_tokens.weight")
    var final_norm = load_named(ctx, paths, entries, name2idx, "model.norm.weight")
    var ln1 = List[DevBuf]()
    var qw = List[WBuf]()
    var qb = List[DevBuf]()
    var kw = List[WBuf]()
    var kb = List[DevBuf]()
    var vw = List[WBuf]()
    var vb = List[DevBuf]()
    var ow = List[WBuf]()
    var ln2 = List[DevBuf]()
    var gate = List[WBuf]()
    var up = List[WBuf]()
    var down = List[WBuf]()
    for l in range(nlayers):
        var p = "model.layers." + String(l) + "."
        ln1.append(load_named(ctx, paths, entries, name2idx, p + "input_layernorm.weight"))
        qw.append(load_named_bf16(ctx, paths, entries, name2idx, p + "self_attn.q_proj.weight"))
        qb.append(load_named(ctx, paths, entries, name2idx, p + "self_attn.q_proj.bias"))
        kw.append(load_named_bf16(ctx, paths, entries, name2idx, p + "self_attn.k_proj.weight"))
        kb.append(load_named(ctx, paths, entries, name2idx, p + "self_attn.k_proj.bias"))
        vw.append(load_named_bf16(ctx, paths, entries, name2idx, p + "self_attn.v_proj.weight"))
        vb.append(load_named(ctx, paths, entries, name2idx, p + "self_attn.v_proj.bias"))
        ow.append(load_named_bf16(ctx, paths, entries, name2idx, p + "self_attn.o_proj.weight"))
        ln2.append(load_named(ctx, paths, entries, name2idx, p + "post_attention_layernorm.weight"))
        gate.append(load_named_bf16(ctx, paths, entries, name2idx, p + "mlp.gate_proj.weight"))
        up.append(load_named_bf16(ctx, paths, entries, name2idx, p + "mlp.up_proj.weight"))
        down.append(load_named_bf16(ctx, paths, entries, name2idx, p + "mlp.down_proj.weight"))
    return Weights(embed^, final_norm^, ln1^, qw^, qb^, kw^, kb^, vw^, vb^, ow^, ln2^, gate^, up^, down^,
                   arch, nlayers, hidden, inter, nkv, hq, hkv, head_dim, VOCAB, False)


# ── op launchers (each runs one kernel, returns a new device buffer) ───────────

def mm(ctx: DeviceContext, mut x: DevBuf, mut w: WBuf, mut b: DevBuf,
       M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    if M == 1:
        # decode: memory-bound GEMV, one warp per output element (M*N warps). The
        # simdgroup-matrix path is for prefill only — at M=1 its 8-row tiles waste
        # 7/8 of every fragment, so decode always uses the GEMV.
        comptime k = matmul_kernel[type_of(lay)]
        ctx.enqueue_function[k](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    elif simd_ok:
        # prefill, fast path: simdgroup-matrix GEMM (~4.5× the scalar tiled kernel
        # on the M4). Gated by the startup probe; the scalar path below is the
        # fallback if this toolchain rejects the AIR intrinsics.
        comptime ks = matmul_simd_kernel[type_of(lay)]
        ctx.enqueue_function[ks](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)), block_dim=SG_TPB,
        )
    else:
        # prefill, scalar fallback: 2D register-tiled GEMM, one warp per (CN-column,
        # TM-token) block, so each weight is reused across TM tokens and each X value
        # across CN columns — cutting the dominant X traffic CN-fold (§11 #12). TM=CN=8
        # measured ~2× a token-only tiling (~210 GFLOP/s) on the M4.
        comptime TM = 8
        comptime CN = 8
        comptime kt = matmul_tiled_kernel[type_of(lay), TM, CN]
        ctx.enqueue_function[kt](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=ceildiv(ceildiv(N, CN) * ceildiv(M, TM) * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    return y^

def probe_simd_gemm(ctx: DeviceContext) raises -> Bool:
    """Runtime capability gate for the simdgroup-matrix GEMM. Runs a tiny
    matmul_simd_kernel and checks it against a CPU reference. Returns False — so
    mm() uses the scalar fallback — if this Metal toolchain rejects the AIR
    intrinsics: that surfaces as a catchable pipeline-state error (not a crash),
    and the DeviceContext stays usable afterward."""
    try:
        var M = 8
        var K = 16
        var N = 8
        var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
        var wb = ctx.enqueue_create_buffer[DType.uint16](N * K)
        var bb = ctx.enqueue_create_buffer[DType.float32](1)
        var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
        var hx = List[Float32]()
        for i in range(M * K):
            hx.append(Float32((i * 3) % 7) * 0.25 - 0.75)
        with xb.map_to_host() as h:
            for i in range(M * K):
                h[i] = hx[i]
        var hw = List[Float32]()        # bf16-truncated weight values (host ref)
        with wb.map_to_host() as h:
            for i in range(N * K):
                var f = Float32((i * 2) % 5) * 0.5 - 1.0
                var bits = UnsafePointer(to=f).bitcast[UInt32]()[0]
                var top = UInt16(bits >> 16)
                h[i] = top
                var re: UInt32 = UInt32(top) << 16
                hw.append(UnsafePointer(to=re).bitcast[Float32]()[0])
        var lay = row_major(M * N)
        var xt = TileTensor(xb, row_major(M * K))
        var wt = TileTensor(wb, row_major(N * K))
        var bt = TileTensor(bb, row_major(1))
        var yt = TileTensor(yb, lay)
        comptime ks = matmul_simd_kernel[type_of(lay)]
        ctx.enqueue_function[ks](
            xt, wt, bt, yt, M, K, N, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)), block_dim=SG_TPB,
        )
        ctx.synchronize()
        var ok = True
        with yb.map_to_host() as h:
            for m in range(M):
                for n in range(N):
                    var acc = Float32(0.0)
                    for k in range(K):
                        acc += hx[m * K + k] * hw[n * K + k]
                    var e = h[m * N + n] - acc
                    if e < 0:
                        e = -e
                    if e > 1.0e-3:
                        ok = False
        return ok
    except:
        return False


def rmsnorm(ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, T: Int, dim: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](T * dim)
    var lay = row_major(T * dim)
    comptime k = rmsnorm_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, lay), TileTensor(w, row_major(dim)), TileTensor(y, lay), T, dim,
        grid_dim=ceildiv(T * WARP_SIZE, BLOCK), block_dim=BLOCK,   # one warp per row
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

def silu_mul(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = silu_mul_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def embed_tokens(ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], mut emb: WBuf, T: Int,
                 hidden: Int = H, vocab: Int = VOCAB) raises -> DevBuf:
    var h = ctx.enqueue_create_buffer[DType.float32](T * hidden)
    var lay = row_major(T * hidden)
    comptime k = embed_kernel[type_of(lay)]   # dimension-agnostic (runtime T, hidden)
    ctx.enqueue_function[k](
        TileTensor(ids, row_major(T)), TileTensor(emb, row_major(vocab * hidden)),
        TileTensor(h, lay), T, hidden,
        grid_dim=ceildiv(T * hidden, BLOCK), block_dim=BLOCK,
    )
    return h^

def last_row(ctx: DeviceContext, mut src: DevBuf, T: Int, dim: Int) raises -> DevBuf:
    """Lift row T-1 (dim elements) of src[T,dim] into a fresh 1×dim buffer."""
    var y = ctx.enqueue_create_buffer[DType.float32](dim)
    var lay = row_major(dim)
    comptime k = slice_row_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, row_major(T * dim)), TileTensor(y, lay), (T - 1) * dim, dim,
        grid_dim=ceildiv(dim, BLOCK), block_dim=BLOCK,
    )
    return y^

def copy_into(ctx: DeviceContext, mut src: DevBuf, mut dst: DevBuf, dst_offset: Int, n: Int, dst_len: Int) raises:
    var lay = row_major(n)
    comptime k = copy_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, lay), TileTensor(dst, row_major(dst_len)), dst_offset, n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )

def rope_k(ctx: DeviceContext, mut kin: DevBuf, mut kc: DevBuf,
           Tq: Int, q_offset: Int, cache_len: Int,
           hkv: Int = HKV, head_dim: Int = 64, arch: Int = 0) raises:
    """Apply RoPE to projected K and store it (rotated) at its cache rows —
    replaces the plain K copy so attention reads pre-rotated K (§11 #12).
    arch selects the comptime head-dim instantiation (0 = 0.5B, 1 = 3B)."""
    var lay = row_major(Tq * hkv * head_dim)
    if arch == 1:
        comptime k = rope_k_kernel[type_of(lay), 2, 128]
        ctx.enqueue_function[k](
            TileTensor(kin, lay), TileTensor(kc, row_major(cache_len)), Tq, q_offset,
            grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK,
        )
    else:
        comptime k = rope_k_kernel[type_of(lay), 2, 64]
        ctx.enqueue_function[k](
            TileTensor(kin, lay), TileTensor(kc, row_major(cache_len)), Tq, q_offset,
            grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK,
        )

def attn_cached(ctx: DeviceContext, mut q: DevBuf, mut kc: DevBuf, mut vc: DevBuf,
                Tq: Int, q_offset: Int, cache_len: Int,
                hidden: Int = H, hq: Int = HQ, hkv: Int = HKV, head_dim: Int = 64,
                arch: Int = 0) raises -> DevBuf:
    # Rotate Q first (rope_q), then attend; K is already rotated in the cache.
    var qr = ctx.enqueue_create_buffer[DType.float32](Tq * hidden)
    var qlay = row_major(Tq * hidden)
    if arch == 1:
        comptime kq = rope_q_kernel[type_of(qlay), 16, 128]
        ctx.enqueue_function[kq](TileTensor(q, qlay), TileTensor(qr, qlay), Tq, q_offset,
            grid_dim=ceildiv(Tq * hq, BLOCK), block_dim=BLOCK)
    else:
        comptime kq = rope_q_kernel[type_of(qlay), 14, 64]
        ctx.enqueue_function[kq](TileTensor(q, qlay), TileTensor(qr, qlay), Tq, q_offset,
            grid_dim=ceildiv(Tq * hq, BLOCK), block_dim=BLOCK)

    var o = ctx.enqueue_create_buffer[DType.float32](Tq * hidden)
    var lay = row_major(Tq * hidden)
    if arch == 0 and q_offset + Tq > FLASH_THRESHOLD:
        # 0.5B long context: stream K/V through shared memory (bit-identical, no cliff).
        # Flash is 0.5B-only: HEAD_DIM=128 would double the staged tile to ~32 KB
        # (Metal threadgroup limit), so 3B always uses attn_cached_kernel.
        comptime kf = flash_attn_kernel[type_of(lay), 14, 2, 64, FLASH_PW]
        comptime nwarp = FLASH_PW * 7    # GROUP = 14/2
        ctx.enqueue_function[kf](
            TileTensor(qr, row_major(Tq * hidden)), TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)), TileTensor(o, lay), Tq, q_offset,
            grid_dim=ceildiv(Tq, FLASH_PW) * hkv, block_dim=nwarp * WARP_SIZE,
        )
    elif arch == 1:
        comptime k = attn_cached_kernel[type_of(lay), 16, 2, 128]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * hidden)), TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)), TileTensor(o, lay), Tq, q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    else:
        comptime k = attn_cached_kernel[type_of(lay), 14, 2, 64]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * hidden)), TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)), TileTensor(o, lay), Tq, q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    return o^


# ── model assembly ─────────────────────────────────────────────────────────────

def layer_cached(ctx: DeviceContext, mut w: Weights, l: Int, mut h: DevBuf,
                 mut kc: DevBuf, mut vc: DevBuf, Tq: Int, q_offset: Int,
                 cache_len: Int, mut dummy: DevBuf) raises -> DevBuf:
    """One decoder layer. Prefill = (Tq=P, q_offset=0); decode = (Tq=1, q_offset=pos).
    Dims come from the loaded Weights, so this serves any supported arch."""
    var hd = w.hidden
    var nkv = w.nkv
    var ln1 = rmsnorm(ctx, h, w.ln1[l], Tq, hd)
    var q = mm(ctx, ln1, w.qw[l], w.qb[l], Tq, hd, hd, 1, w.simd_ok)
    var kk = mm(ctx, ln1, w.kw[l], w.kb[l], Tq, hd, nkv, 1, w.simd_ok)
    var vv = mm(ctx, ln1, w.vw[l], w.vb[l], Tq, hd, nkv, 1, w.simd_ok)
    rope_k(ctx, kk, kc, Tq, q_offset, cache_len, w.hkv, w.head_dim, w.arch)   # store K RoPE-rotated
    copy_into(ctx, vv, vc, q_offset * nkv, Tq * nkv, cache_len)               # V is not rotated
    var o = attn_cached(ctx, q, kc, vc, Tq, q_offset, cache_len,
                        w.hidden, w.hq, w.hkv, w.head_dim, w.arch)
    var o2 = mm(ctx, o, w.ow[l], dummy, Tq, hd, hd, 0, w.simd_ok)
    var h2 = add(ctx, h, o2, Tq * hd)
    var ln2 = rmsnorm(ctx, h2, w.ln2[l], Tq, hd)
    var g = mm(ctx, ln2, w.gate[l], dummy, Tq, hd, w.inter, 0, w.simd_ok)
    var u = mm(ctx, ln2, w.up[l], dummy, Tq, hd, w.inter, 0, w.simd_ok)
    var gu = silu_mul(ctx, g, u, Tq * w.inter)
    var dn = mm(ctx, gu, w.down[l], dummy, Tq, w.inter, hd, 0, w.simd_ok)
    return add(ctx, h2, dn, Tq * hd)

def argmax_last(ctx: DeviceContext, mut w: Weights, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> Int:
    """Final RMSNorm + tied LM head; argmax over the last position's logits.

    Only row T-1 feeds the LM head: the VOCAB-wide (151936) head is the largest
    matmul in the net, so at prefill running it on all T rows and keeping one was
    the dominant cost (§11 #12). Slice the last hidden row first → one GEMV."""
    var hl = last_row(ctx, h, T, w.hidden)
    var hn = rmsnorm(ctx, hl, w.final_norm, 1, w.hidden)
    var logits = mm(ctx, hn, w.embed, dummy, 1, w.hidden, w.vocab, 0)
    ctx.synchronize()
    var best = -1
    var best_v = Float32(-1.0e30)
    with logits.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.vocab))
        for i in range(w.vocab):
            var v = rebind[Scalar[DType.float32]](mt[i])
            if v > best_v:
                best_v = v
                best = i
    return best

def logits_last(ctx: DeviceContext, mut w: Weights, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
    """Final RMSNorm + tied LM head; returns the last position's logits on host.
    Slices row T-1 before the head so prefill runs it once, not T times (§11 #12)."""
    var hl = last_row(ctx, h, T, w.hidden)
    var hn = rmsnorm(ctx, hl, w.final_norm, 1, w.hidden)
    var logits = mm(ctx, hn, w.embed, dummy, 1, w.hidden, w.vocab, 0)
    ctx.synchronize()
    var out = List[Float32]()
    with logits.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.vocab))
        for i in range(w.vocab):
            out.append(rebind[Scalar[DType.float32]](mt[i]))
    return out^


# ── sampling (ARCHITECTURE.md §5.6; generation_config.json defaults) ───────────

@fieldwise_init
struct Dist(Movable):
    var ids: List[Int]
    var probs: List[Float32]


def process_logits(logits: List[Float32], context: List[Int], temp: Float32,
                   top_k: Int, top_p: Float32, rep_pen: Float32) raises -> Dist:
    """HF order: repetition_penalty → temperature → top_k → top_p → softmax.
    Returns the kept token ids and their (renormalized) probabilities."""
    var v = logits.copy()

    # repetition penalty over the unique tokens seen so far
    var seen = List[Bool]()
    for _ in range(len(v)):
        seen.append(False)
    for c in context:
        var id = Int(c)
        if 0 <= id and id < len(v) and not seen[id]:
            seen[id] = True
            v[id] = v[id] / rep_pen if v[id] > 0 else v[id] * rep_pen

    # temperature
    for i in range(len(v)):
        v[i] = v[i] / temp

    # top-k: pull the k largest (k is small; selection beats a full sort)
    var k = top_k if top_k < len(v) else len(v)
    var used = List[Bool]()
    for _ in range(len(v)):
        used.append(False)
    var ids = List[Int]()
    var logs = List[Float32]()
    for _ in range(k):
        var bi = -1
        var bv = Float32(-1.0e30)
        for i in range(len(v)):
            if not used[i] and v[i] > bv:
                bv = v[i]
                bi = i
        used[bi] = True
        ids.append(bi)
        logs.append(v[bi])

    # softmax over the top-k (descending order already)
    var maxl = logs[0]
    var ps = List[Float32]()
    var z = Float32(0.0)
    for i in range(len(logs)):
        var e = exp(logs[i] - maxl)
        ps.append(e)
        z += e
    for i in range(len(ps)):
        ps[i] = ps[i] / z

    # top-p: keep the smallest prefix with cumulative prob >= top_p
    var keep = 0
    var cum = Float32(0.0)
    for i in range(len(ps)):
        keep = i + 1
        cum += ps[i]
        if cum >= top_p:
            break

    var out_ids = List[Int]()
    var out_probs = List[Float32]()
    var s = Float32(0.0)
    for i in range(keep):
        s += ps[i]
    for i in range(keep):
        out_ids.append(ids[i])
        out_probs.append(ps[i] / s)
    return Dist(out_ids^, out_probs^)


def next_rand(mut state: UInt64) -> UInt64:
    state ^= state << UInt64(13)
    state ^= state >> UInt64(7)
    state ^= state << UInt64(17)
    return state

def sample(dist: Dist, mut rng: UInt64) -> Int:
    var r = Float32(Int(next_rand(rng) >> UInt64(40))) / Float32(1 << 24)  # [0,1)
    var cum = Float32(0.0)
    for i in range(len(dist.ids)):
        cum += dist.probs[i]
        if r < cum:
            return dist.ids[i]
    return dist.ids[len(dist.ids) - 1]


def upload_ids(ctx: DeviceContext, vals: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var n = len(vals)
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    with d.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(vals[i]))
    return d^

def argmax_f(logits: List[Float32]) -> Int:
    var best = -1
    var best_v = Float32(-1.0e30)
    for i in range(len(logits)):
        if logits[i] > best_v:
            best_v = logits[i]
            best = i
    return best


# ── decode session: KV caches + position, the per-step primitive ──────────────

@fieldwise_init
struct Session(Movable):
    """Holds the per-layer KV caches and the current position. `prefill` runs the
    prompt and returns the last-position logits; `step` advances one token. Shared
    by greedy/sampled generate and the server's streaming loop."""
    var kcs: List[DevBuf]
    var vcs: List[DevBuf]
    var dummy: DevBuf
    var cache_len: Int
    var pos: Int


def new_session(ctx: DeviceContext, max_seq: Int, nlayers: Int = NLAYERS, nkv: Int = NKV) raises -> Session:
    var cache_len = max_seq * nkv
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(nlayers):
        kcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
        vcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
    return Session(kcs^, vcs^, ctx.enqueue_create_buffer[DType.float32](1), cache_len, 0)


def sess_prefill(ctx: DeviceContext, mut w: Weights, mut s: Session, prompt: List[Int]) raises -> List[Float32]:
    var P = len(prompt)
    var ids_dev = upload_ids(ctx, prompt)
    var h = embed_tokens(ctx, ids_dev, w.embed, P, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = layer_cached(ctx, w, l, h, s.kcs[l], s.vcs[l], P, 0, s.cache_len, s.dummy)
    s.pos = P
    return logits_last(ctx, w, h, P, s.dummy)


# Below this suffix length, prefill is sub-second and frequent (the prefix-cache
# common case), so progress reporting — and its per-layer synchronize — is skipped
# entirely, leaving that hot path byte-for-byte as before.
comptime PROGRESS_MIN_TOK = 2048
comptime PROGRESS_EVERY_NS = 5_000_000_000   # ~5 s between progress lines


def _ktok(n: Int) -> String:
    """Compact token count: 9562 -> '9.5k', 800 -> '800'."""
    if n < 1000:
        return String(n)
    return String(n // 1000) + "." + String((n % 1000) // 100) + "k"

def _dur(secs: Float64) -> String:
    var s = Int(secs + 0.5)
    if s < 90:
        return String(s) + "s"
    return String(s // 60) + "m" + String(s % 60) + "s"


def sess_prefill_suffix(ctx: DeviceContext, mut w: Weights, mut s: Session,
                        suffix: List[Int], offset: Int, progress: Bool = False) raises -> List[Float32]:
    """Prefill `suffix` tokens at cache position `offset`, reusing the K/V already
    stored in rows [0, offset). Returns the last-row logits. This is the engine
    behind the server's cross-request prefix cache; `sess_prefill` is just the
    offset==0 / whole-prompt special case. RoPE positions come from `offset`, so
    the rotated K and the attention mask stay correct for the reused prefix.

    With `progress` (and a large enough suffix), prints a throttled stdout line
    with percent done + ETA. Layers are uniform cost, so elapsed/layers-done
    extrapolates accurately. It synchronizes per layer to time real GPU progress;
    that adds no throughput cost (layers are a sequential dependency chain anyway)
    but is gated off below PROGRESS_MIN_TOK so the frequent tiny prefills are
    untouched. Granularity is per-layer — a kernel already running can't be
    interrupted — so very long contexts tick per layer rather than exactly 5 s."""
    var Q = len(suffix)
    var ids_dev = upload_ids(ctx, suffix)
    var h = embed_tokens(ctx, ids_dev, w.embed, Q, w.hidden, w.vocab)
    var report = progress and Q >= PROGRESS_MIN_TOK
    var t0 = perf_counter_ns()
    var last = t0
    for l in range(w.nlayers):
        h = layer_cached(ctx, w, l, h, s.kcs[l], s.vcs[l], Q, offset, s.cache_len, s.dummy)
        if report:
            ctx.synchronize()
            var now = perf_counter_ns()
            if Float64(now - last) >= Float64(PROGRESS_EVERY_NS):
                var done = l + 1
                var elapsed = Float64(now - t0) / 1.0e9
                var eta = elapsed * Float64(w.nlayers - done) / Float64(done)
                print("  prefill ", _ktok(Q), "tok: ", (done * 100) // w.nlayers,
                      "% (layer ", done, "/", w.nlayers, "), ~", _dur(eta), " left", sep="")
                last = now
    s.pos = offset + Q
    return logits_last(ctx, w, h, Q, s.dummy)


def sess_step(ctx: DeviceContext, mut w: Weights, mut s: Session, token: Int) raises -> List[Float32]:
    var one = upload_ids(ctx, [token])
    var h = embed_tokens(ctx, one, w.embed, 1, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = layer_cached(ctx, w, l, h, s.kcs[l], s.vcs[l], 1, s.pos, s.cache_len, s.dummy)
    s.pos += 1
    return logits_last(ctx, w, h, 1, s.dummy)


def generate(ctx: DeviceContext, mut w: Weights, prompt: List[Int], max_new: Int) raises -> List[Int]:
    """Greedy decode: prefill the prompt then emit tokens until EOS or max_new."""
    var s = new_session(ctx, len(prompt) + max_new + 2, w.nlayers, w.nkv)
    var nxt = argmax_f(sess_prefill(ctx, w, s, prompt))
    var gen = List[Int]()
    gen.append(nxt)
    while len(gen) < max_new and nxt != EOS1 and nxt != EOS2:
        nxt = argmax_f(sess_step(ctx, w, s, nxt))
        gen.append(nxt)
    return gen^


def generate_sample(ctx: DeviceContext, mut w: Weights, prompt: List[Int], max_new: Int,
                    temp: Float32, top_k: Int, top_p: Float32, rep_pen: Float32,
                    seed: UInt64) raises -> List[Int]:
    """Greedy-structure decode but draw each token from the processed distribution."""
    var s = new_session(ctx, len(prompt) + max_new + 2, w.nlayers, w.nkv)
    var rng = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)
    var context = prompt.copy()
    var nxt = sample(process_logits(sess_prefill(ctx, w, s, prompt), context, temp, top_k, top_p, rep_pen), rng)
    var gen = List[Int]()
    gen.append(nxt)
    context.append(nxt)
    while len(gen) < max_new and nxt != EOS1 and nxt != EOS2:
        var dist = process_logits(sess_step(ctx, w, s, nxt), context, temp, top_k, top_p, rep_pen)
        nxt = sample(dist, rng)
        context.append(nxt)
        gen.append(nxt)
    return gen^
