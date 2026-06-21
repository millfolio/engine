"""safetensors checkpoint loading: a tiny JSON-subset header parser plus the
weight loaders that upload bf16 tensors to device (optionally widening to f32 or
quantizing to group-128 int4). Model-agnostic — dims come from the caller. The
weight representation types are shared from tensor_ops to avoid an import cycle."""

from std.math import ceildiv
from std.os.path import isdir
from std.memory import memcpy
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import cvt_kernel, Q4_GROUP, bf16_widen
from runtime.tensor_ops import BLOCK, DevBuf, WBuf, PBuf, QMat, qmat_bf16


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
    # Read in ≤1 GiB chunks: macOS read() fails with EINVAL for a single count >2 GiB,
    # which the gemma-4 e2b embed_tokens_per_layer (262144×8960 bf16 = 4.7 GiB) hits.
    comptime CHUNK = 1 << 30
    with open(path, "r") as f:
        _ = f.seek(UInt64(begin))
        var host = ctx.enqueue_create_host_buffer[DType.uint16](count)
        ctx.synchronize()
        var dst = host.unsafe_ptr().bitcast[UInt8]()
        var off = 0
        while off < nbytes:
            var want = nbytes - off
            if want > CHUNK:
                want = CHUNK
            var raw = f.read_bytes(want)
            var got = len(raw)
            if got == 0:
                break
            memcpy(dest=dst + off, src=raw.unsafe_ptr(), count=got)
            off += got
        ctx.enqueue_copy(dev_u16, host)
        ctx.synchronize()
    return dev_u16^

def load_named_bf16(ctx: DeviceContext, paths: List[String], entries: List[TensorEntry],
                    name2idx: Dict[String, Int], name: String) raises -> WBuf:
    var idx = name2idx[name]
    return load_one_bf16(ctx, paths[idx], entries[idx].begin, entries[idx].end)


def load_one_q4(ctx: DeviceContext, path: String, begin: Int, end: Int, K: Int) raises -> QMat:
    """Load a bf16 weight [N,K] (row-major; K = reduction dim, a multiple of 128)
    and quantize it to group-128 int4 on the host — symmetric RTN, scale per
    128-wide group along K. Reads the raw bf16 bytes (no full-precision copy ever
    reaches the device), packs 8 nibbles/u32, uploads packed+scales. One-time at
    load (host-side, so it is not fast — a few minutes for the 3B)."""
    var nbytes = end - begin
    var count = nbytes // 2                    # u16 weights = N*K
    var N = count // K
    var NG = K // Q4_GROUP
    var pcount = count // 8                     # u32 words = N*K/8
    var packed_host = ctx.enqueue_create_host_buffer[DType.uint32](pcount)
    var scales_host = ctx.enqueue_create_host_buffer[DType.float32](N * NG)
    ctx.synchronize()
    var pp = packed_host.unsafe_ptr()
    var sp = scales_host.unsafe_ptr()
    for i in range(pcount):
        pp[i] = 0
    with open(path, "r") as f:
        _ = f.seek(UInt64(begin))
        var raw = f.read_bytes(nbytes)
        var u16 = raw.unsafe_ptr().bitcast[UInt16]()    # little-endian bf16 bits
        for n in range(N):
            for g in range(NG):
                var amax = Float32(0.0)
                for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                    var v = bf16_widen(u16[n * K + k])
                    var a = v if v >= 0.0 else -v
                    if a > amax:
                        amax = a
                var s = amax / 7.0 if amax > 0.0 else Float32(1.0)
                sp[n * NG + g] = s
                var inv = 1.0 / s
                for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                    var q = bf16_widen(u16[n * K + k]) * inv
                    var half = Float32(0.5) if q >= 0.0 else Float32(-0.5)
                    var qr = Int(q + half)
                    if qr > 7:
                        qr = 7
                    elif qr < -7:
                        qr = -7
                    var lin = n * K + k
                    pp[lin >> 3] = pp[lin >> 3] | (UInt32(qr + 8) << UInt32((lin & 7) * 4))
    var packed_dev = ctx.enqueue_create_buffer[DType.uint32](pcount)
    var scales_dev = ctx.enqueue_create_buffer[DType.float32](N * NG)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scales_dev, scales_host)
    ctx.synchronize()
    return QMat(ctx.enqueue_create_buffer[DType.uint16](1), packed_dev^, scales_dev^, True)


def load_proj(ctx: DeviceContext, paths: List[String], entries: List[TensorEntry],
              name2idx: Dict[String, Int], name: String, K: Int, q4: Bool) raises -> QMat:
    """A projection weight as QMat: int4 (group-128) if `q4` else bf16."""
    var idx = name2idx[name]
    if q4:
        return load_one_q4(ctx, paths[idx], entries[idx].begin, entries[idx].end, K)
    return qmat_bf16(ctx, load_one_bf16(ctx, paths[idx], entries[idx].begin, entries[idx].end))


def fuse_pair(ctx: DeviceContext, var a: QMat, var b: QMat,
              Na: Int, Nb: Int, K: Int, q4: Bool) raises -> QMat:
    """Concatenate two same-K projection weights along the output dim N (a's rows
    then b's) into one QMat, so a single GEMV computes both (e.g. gate|up, q|k|v).
    Host-side copy at load — a/b are already the right representation."""
    if q4:
        var wa = Na * K // 8
        var wb = Nb * K // 8
        var sa = Na * (K // Q4_GROUP)
        var sb = Nb * (K // Q4_GROUP)
        var pc = ctx.enqueue_create_buffer[DType.uint32](wa + wb)
        var sc = ctx.enqueue_create_buffer[DType.float32](sa + sb)
        with pc.map_to_host() as d:
            with a.packed.map_to_host() as ah:
                memcpy(dest=d.unsafe_ptr().bitcast[UInt8](), src=ah.unsafe_ptr().bitcast[UInt8](), count=wa * 4)
            with b.packed.map_to_host() as bh:
                memcpy(dest=(d.unsafe_ptr() + wa).bitcast[UInt8](), src=bh.unsafe_ptr().bitcast[UInt8](), count=wb * 4)
        with sc.map_to_host() as d:
            with a.scales.map_to_host() as ah:
                memcpy(dest=d.unsafe_ptr().bitcast[UInt8](), src=ah.unsafe_ptr().bitcast[UInt8](), count=sa * 4)
            with b.scales.map_to_host() as bh:
                memcpy(dest=(d.unsafe_ptr() + sa).bitcast[UInt8](), src=bh.unsafe_ptr().bitcast[UInt8](), count=sb * 4)
        return QMat(ctx.enqueue_create_buffer[DType.uint16](1), pc^, sc^, True)
    var na = Na * K
    var nb = Nb * K
    var bc = ctx.enqueue_create_buffer[DType.uint16](na + nb)
    with bc.map_to_host() as d:
        with a.bf16.map_to_host() as ah:
            memcpy(dest=d.unsafe_ptr().bitcast[UInt8](), src=ah.unsafe_ptr().bitcast[UInt8](), count=na * 2)
        with b.bf16.map_to_host() as bh:
            memcpy(dest=(d.unsafe_ptr() + na).bitcast[UInt8](), src=bh.unsafe_ptr().bitcast[UInt8](), count=nb * 2)
    return QMat(bc^, ctx.enqueue_create_buffer[DType.uint32](1), ctx.enqueue_create_buffer[DType.float32](1), False)


def concat_bias(ctx: DeviceContext, var a: DevBuf, var b: DevBuf, na: Int, nb: Int) raises -> DevBuf:
    """Concatenate two f32 bias vectors (for the fused QKV bias)."""
    var c = ctx.enqueue_create_buffer[DType.float32](na + nb)
    with c.map_to_host() as d:
        with a.map_to_host() as ah:
            memcpy(dest=d.unsafe_ptr().bitcast[UInt8](), src=ah.unsafe_ptr().bitcast[UInt8](), count=na * 4)
        with b.map_to_host() as bh:
            memcpy(dest=(d.unsafe_ptr() + na).bitcast[UInt8](), src=bh.unsafe_ptr().bitcast[UInt8](), count=nb * 4)
    return c^


def _str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _parse_shard_names(buf: List[UInt8]) raises -> List[String]:
    """Distinct shard filenames from a safetensors `model.safetensors.index.json`
    weight_map. Reuses the tiny safetensors-header JSON helpers (no jinja2.mojo dep, so
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
        # Single-file checkpoint. `path` may be the file itself or (HF cache /
        # native downloader) the snapshot directory holding model.safetensors.
        var file = path
        if isdir(path):
            file = path + "/model.safetensors"
        var se = read_header(file)
        for e in range(len(se)):
            entries.append(se[e].copy())
            paths.append(file)
    return (entries^, paths^)
