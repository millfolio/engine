"""Safetensors checkpoint loading: a tiny JSON-subset header parser plus the
weight loaders that upload bf16 tensors to device (optionally widening to f32 or
quantizing to group-128 int4). Model-agnostic — dims come from the caller. The
weight representation types are shared from tensor_ops to avoid an import cycle."""

from std.math import ceildiv
from std.os.path import isdir
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import cvt_kernel, Q4_GROUP, bf16_widen
from runtime.tensor_ops import BLOCK, DevBuf, WBuf, PBuf, QMat, qmat_bf16


# ── safetensors header (JSON subset) ──────────────────────────────────────────

comptime QUOTE = 34
"""ASCII byte value for `\"`."""
comptime LBRACE = 123
"""ASCII byte value for `{`."""
comptime RBRACE = 125
"""ASCII byte value for `}`."""
comptime LBRACK = 91
"""ASCII byte value for `[`."""
comptime RBRACK = 93
"""ASCII byte value for `]`."""
comptime COLON = 58
"""ASCII byte value for `:`."""
comptime COMMA = 44
"""ASCII byte value for `,`."""


@fieldwise_init
struct TensorEntry(Copyable, Movable):
    """One header entry: a tensor's name and its byte range in the data blob."""

    var name: String
    """The tensor's name (the safetensors header key)."""
    var begin: Int
    """Start byte offset of the tensor's data."""
    var end: Int
    """End byte offset (exclusive) of the tensor's data."""


def is_ws(c: Int) -> Bool:
    """True if byte `c` is JSON whitespace (space, tab, LF, or CR).

    Args:
        c: A byte value.

    Returns:
        True if `c` is a JSON whitespace byte.
    """
    return c == 32 or c == 9 or c == 10 or c == 13


def skip_ws(buf: List[UInt8], mut pos: Int):
    """Advance `pos` past any run of whitespace in `buf`.

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset, advanced in place past any whitespace.
    """
    while pos < len(buf) and is_ws(Int(buf[pos])):
        pos += 1


def expect(buf: List[UInt8], mut pos: Int, ch: Int) raises:
    """Consume byte `ch` at `pos`, raising a parse error if it isn't there.

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced one byte past `ch` on success.
        ch: The byte value that must appear at `pos`.

    Raises:
        Error: if `pos` is past the end of `buf` or the byte there isn't `ch`.
    """
    if pos >= len(buf) or Int(buf[pos]) != ch:
        raise Error("parse error at byte " + String(pos))
    pos += 1


def parse_string(buf: List[UInt8], mut pos: Int) raises -> String:
    """Parse a quoted JSON string at `pos` (no escape handling) and return it.

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced past the closing quote on success.

    Returns:
        The string contents between the quotes.

    Raises:
        Error: if the opening or closing quote is missing.
    """
    expect(buf, pos, QUOTE)
    var s = String("")
    while pos < len(buf) and Int(buf[pos]) != QUOTE:
        s += chr(Int(buf[pos]))
        pos += 1
    expect(buf, pos, QUOTE)
    return s^


def parse_uint(buf: List[UInt8], mut pos: Int) raises -> Int:
    """Parse a run of decimal digits at `pos` into an int (raises if none).

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced past the digits consumed.

    Returns:
        The parsed unsigned integer value.

    Raises:
        Error: if no decimal digit is present at `pos`.
    """
    var v = 0
    var start = pos
    while pos < len(buf) and Int(buf[pos]) >= 48 and Int(buf[pos]) <= 57:
        v = v * 10 + (Int(buf[pos]) - 48)
        pos += 1
    if pos == start:
        raise Error("expected int at " + String(pos))
    return v


def parse_int_array(buf: List[UInt8], mut pos: Int) raises -> List[Int]:
    """Parse a JSON array of unsigned ints (e.g. data_offsets) at `pos`.

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced past the closing `]`.

    Returns:
        The parsed ints in order (empty for `[]`).

    Raises:
        Error: on a malformed array (missing `[`, `]`, or an element).
    """
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
    """Advance `pos` past one JSON value (string, object, array, or scalar).

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced past the whole value.

    Raises:
        Error: on a malformed string/object/array nested in the value.
    """
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
    """Advance `pos` past a whole JSON object, including nested values.

    Args:
        buf: The header bytes being scanned.
        pos: The scan offset; advanced past the closing `}`.

    Raises:
        Error: on a malformed object (missing `{`, key, `:`, or `}`).
    """
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
    """Parse a safetensors header object into per-tensor entries, skipping the
    `__metadata__` key. begin/end are offsets relative to the data blob.

    Args:
        buf: The safetensors header JSON bytes.

    Returns:
        One `TensorEntry` per tensor, with begin/end relative to the data blob.

    Raises:
        Error: on a malformed header.
    """
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
    """Parse the header; entries' begin/end are ABSOLUTE file offsets.

    Args:
        path: Filesystem path to a `.safetensors` file.

    Returns:
        The tensor entries with begin/end shifted by the 8-byte length prefix
        plus header length, so they are absolute file offsets.

    Raises:
        Error: on I/O failure or a malformed header.
    """
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


def load_one(
    ctx: DeviceContext, path: String, begin: Int, end: Int
) raises -> DevBuf:
    """Load the bf16 tensor in byte range [begin, end) and widen it to a fresh f32
    device buffer (via the cvt kernel).

    Args:
        ctx: The device context to allocate and enqueue on.
        path: Filesystem path holding the tensor bytes.
        begin: Start byte offset of the bf16 tensor.
        end: End byte offset (exclusive) of the bf16 tensor.

    Returns:
        A fresh f32 device buffer holding the widened tensor.

    Raises:
        Error: on I/O failure or a device allocation/enqueue failure.
    """
    var nbytes = end - begin
    var count = nbytes // 2
    var dev_f32 = ctx.enqueue_create_buffer[DType.float32](count)
    with open(path, "r") as f:
        _ = f.seek(begin)
        var raw = f.read_bytes(nbytes)
        var host = ctx.enqueue_create_host_buffer[DType.uint16](count)
        ctx.synchronize()
        unsafe_memcpy(
            dest=host.unsafe_ptr().unsafe_bitcast[UInt8](),
            src=raw.unsafe_ptr(),
            count=nbytes,
        )
        var dev_u16 = ctx.enqueue_create_buffer[DType.uint16](count)
        ctx.enqueue_copy(dev_u16, host)
        var lay = row_major(count)
        comptime k = cvt_kernel[type_of(lay)]
        ctx.enqueue_function[k](
            TileTensor(dev_u16, lay),
            TileTensor(dev_f32, lay),            Int32(count),
            grid_dim=ceildiv(count, BLOCK),
            block_dim=BLOCK)
        ctx.synchronize()
    return dev_f32^


def load_named(
    ctx: DeviceContext,
    paths: List[String],
    entries: List[TensorEntry],
    name2idx: Dict[String, Int],
    name: String,
) raises -> DevBuf:
    """Look up tensor `name` and load it as a widened f32 device buffer.

    Args:
        ctx: The device context to allocate and enqueue on.
        paths: Per-entry file paths (aligned with `entries`).
        entries: The parsed tensor entries.
        name2idx: Map from tensor name to its index in `entries`/`paths`.
        name: The tensor name to load.

    Returns:
        The named tensor as a widened f32 device buffer.

    Raises:
        Error: if `name` is absent, or on I/O / device failure.
    """
    var idx = name2idx[name]
    return load_one(ctx, paths[idx], entries[idx].begin, entries[idx].end)


def load_one_bf16(
    ctx: DeviceContext, path: String, begin: Int, end: Int
) raises -> WBuf:
    """Load a bf16 tensor to device *without* widening to f32 — the matmul/embed
    kernels widen per element (bf16_widen), halving weight read traffic (§11 #12).
    The raw safetensors bytes are already bf16, so this is a plain upload.

    Args:
        ctx: The device context to allocate and enqueue on.
        path: Filesystem path holding the tensor bytes.
        begin: Start byte offset of the bf16 tensor.
        end: End byte offset (exclusive) of the bf16 tensor.

    Returns:
        The raw bf16 tensor as a u16 device buffer (not widened).

    Raises:
        Error: on I/O failure or a device allocation/enqueue failure.
    """
    var nbytes = end - begin
    var count = nbytes // 2
    var dev_u16 = ctx.enqueue_create_buffer[DType.uint16](count)
    # Read in ≤1 GiB chunks: macOS read() fails with EINVAL for a single count >2 GiB,
    # which the gemma-4 e2b embed_tokens_per_layer (262144×8960 bf16 = 4.7 GiB) hits.
    comptime CHUNK = 1 << 30
    with open(path, "r") as f:
        _ = f.seek(begin)
        var host = ctx.enqueue_create_host_buffer[DType.uint16](count)
        ctx.synchronize()
        var dst = host.unsafe_ptr().unsafe_bitcast[UInt8]()
        var off = 0
        while off < nbytes:
            var want = nbytes - off
            if want > CHUNK:
                want = CHUNK
            var raw = f.read_bytes(want)
            var got = len(raw)
            if got == 0:
                break
            unsafe_memcpy(dest=dst.unsafe_offset(off), src=raw.unsafe_ptr(), count=got)
            off += got
        ctx.enqueue_copy(dev_u16, host)
        ctx.synchronize()
    return dev_u16^


def load_named_bf16(
    ctx: DeviceContext,
    paths: List[String],
    entries: List[TensorEntry],
    name2idx: Dict[String, Int],
    name: String,
) raises -> WBuf:
    """Look up tensor `name` and load it as a raw bf16 (u16) device buffer.

    Args:
        ctx: The device context to allocate and enqueue on.
        paths: Per-entry file paths (aligned with `entries`).
        entries: The parsed tensor entries.
        name2idx: Map from tensor name to its index in `entries`/`paths`.
        name: The tensor name to load.

    Returns:
        The named tensor as a raw bf16 (u16) device buffer.

    Raises:
        Error: if `name` is absent, or on I/O / device failure.
    """
    var idx = name2idx[name]
    return load_one_bf16(ctx, paths[idx], entries[idx].begin, entries[idx].end)


def load_one_q4(
    ctx: DeviceContext, path: String, begin: Int, end: Int, K: Int
) raises -> QMat:
    """Load a bf16 weight [N,K] (row-major; K = reduction dim, a multiple of 128)
    and quantize it to group-128 int4 on the host — symmetric RTN, scale per
    128-wide group along K. Reads the raw bf16 bytes (no full-precision copy ever
    reaches the device), packs 8 nibbles/u32, uploads packed+scales. One-time at
    load (host-side, so it is not fast — a few minutes for the 3B).

    Args:
        ctx: The device context to allocate and enqueue on.
        path: Filesystem path holding the tensor bytes.
        begin: Start byte offset of the bf16 weight.
        end: End byte offset (exclusive) of the bf16 weight.
        K: The reduction dim (a multiple of 128); the weight is [N, K] with
            N inferred from the byte count.

    Returns:
        A `QMat` holding the packed group-128 int4 weights and per-group scales.

    Raises:
        Error: on I/O failure or a device allocation/enqueue failure.
    """
    var nbytes = end - begin
    var count = nbytes // 2  # u16 weights = N*K
    var N = count // K
    var NG = K // Q4_GROUP
    var pcount = count // 8  # u32 words = N*K/8
    var packed_host = ctx.enqueue_create_host_buffer[DType.uint32](pcount)
    var scales_host = ctx.enqueue_create_host_buffer[DType.float32](N * NG)
    ctx.synchronize()
    var pp = packed_host.unsafe_ptr()
    var sp = scales_host.unsafe_ptr()
    for i in range(pcount):
        pp[unsafe_offset=i] = 0
    with open(path, "r") as f:
        _ = f.seek(begin)
        var raw = f.read_bytes(nbytes)
        var u16 = raw.unsafe_ptr().unsafe_bitcast[UInt16]()  # little-endian bf16 bits
        for n in range(N):
            for g in range(NG):
                var amax = Float32(0.0)
                for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                    var v = bf16_widen(u16[unsafe_offset=n * K + k])
                    var a = v if v >= 0.0 else -v
                    if a > amax:
                        amax = a
                var s = amax / 7.0 if amax > 0.0 else Float32(1.0)
                sp[unsafe_offset=n * NG + g] = s
                var inv = 1.0 / s
                for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                    var q = bf16_widen(u16[unsafe_offset=n * K + k]) * inv
                    var half = Float32(0.5) if q >= 0.0 else Float32(-0.5)
                    var qr = Int(q + half)
                    if qr > 7:
                        qr = 7
                    elif qr < -7:
                        qr = -7
                    var lin = n * K + k
                    pp[unsafe_offset=lin >> 3] = pp[unsafe_offset=lin >> 3] | (
                        UInt32(qr + 8) << UInt32((lin & 7) * 4)
                    )
    var packed_dev = ctx.enqueue_create_buffer[DType.uint32](pcount)
    var scales_dev = ctx.enqueue_create_buffer[DType.float32](N * NG)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scales_dev, scales_host)
    ctx.synchronize()
    return QMat(
        ctx.enqueue_create_buffer[DType.uint16](1),
        packed_dev^,
        scales_dev^,
        True,
    )


def load_proj(
    ctx: DeviceContext,
    paths: List[String],
    entries: List[TensorEntry],
    name2idx: Dict[String, Int],
    name: String,
    K: Int,
    q4: Bool,
) raises -> QMat:
    """A projection weight as QMat: int4 (group-128) if `q4` else bf16.

    Args:
        ctx: The device context to allocate and enqueue on.
        paths: Per-entry file paths (aligned with `entries`).
        entries: The parsed tensor entries.
        name2idx: Map from tensor name to its index in `entries`/`paths`.
        name: The tensor name to load.
        K: The reduction dim (a multiple of 128); the weight is [N, K].
        q4: If True quantize to group-128 int4, else keep bf16.

    Returns:
        The projection weight as a `QMat` (int4 group-128 if `q4`, else bf16).

    Raises:
        Error: if `name` is absent, or on I/O / device failure.
    """
    var idx = name2idx[name]
    if q4:
        return load_one_q4(
            ctx, paths[idx], entries[idx].begin, entries[idx].end, K
        )
    return qmat_bf16(
        ctx,
        load_one_bf16(ctx, paths[idx], entries[idx].begin, entries[idx].end),
    )


def fuse_pair(
    ctx: DeviceContext,
    var a: QMat,
    var b: QMat,
    Na: Int,
    Nb: Int,
    K: Int,
    q4: Bool,
) raises -> QMat:
    """Concatenate two same-K projection weights along the output dim N (a's rows
    then b's) into one QMat, so a single GEMV computes both (e.g. gate|up, q|k|v).
    Host-side copy at load — a/b are already the right representation.

    Args:
        ctx: The device context to allocate and enqueue on.
        a: The first weight (consumed), contributing the leading rows.
        b: The second weight (consumed), contributing the trailing rows.
        Na: The number of output rows in `a`.
        Nb: The number of output rows in `b`.
        K: The shared reduction dim of both weights.
        q4: True if `a`/`b` are int4 group-128, False if bf16.

    Returns:
        A single `QMat` with `a`'s rows followed by `b`'s along N.

    Raises:
        Error: on a device allocation or buffer-mapping failure.
    """
    if q4:
        var wa = Na * K // 8
        var wb = Nb * K // 8
        var sa = Na * (K // Q4_GROUP)
        var sb = Nb * (K // Q4_GROUP)
        var pc = ctx.enqueue_create_buffer[DType.uint32](wa + wb)
        var sc = ctx.enqueue_create_buffer[DType.float32](sa + sb)
        with pc.map_to_host() as d:
            with a.packed.map_to_host() as ah:
                unsafe_memcpy(
                    dest=d.unsafe_ptr().unsafe_bitcast[UInt8](),
                    src=ah.unsafe_ptr().unsafe_bitcast[UInt8](),
                    count=wa * 4,
                )
            with b.packed.map_to_host() as bh:
                unsafe_memcpy(
                    dest=(d.unsafe_ptr().unsafe_offset(wa)).unsafe_bitcast[UInt8](),
                    src=bh.unsafe_ptr().unsafe_bitcast[UInt8](),
                    count=wb * 4,
                )
        with sc.map_to_host() as d:
            with a.scales.map_to_host() as ah:
                unsafe_memcpy(
                    dest=d.unsafe_ptr().unsafe_bitcast[UInt8](),
                    src=ah.unsafe_ptr().unsafe_bitcast[UInt8](),
                    count=sa * 4,
                )
            with b.scales.map_to_host() as bh:
                unsafe_memcpy(
                    dest=(d.unsafe_ptr().unsafe_offset(sa)).unsafe_bitcast[UInt8](),
                    src=bh.unsafe_ptr().unsafe_bitcast[UInt8](),
                    count=sb * 4,
                )
        return QMat(ctx.enqueue_create_buffer[DType.uint16](1), pc^, sc^, True)
    var na = Na * K
    var nb = Nb * K
    var bc = ctx.enqueue_create_buffer[DType.uint16](na + nb)
    with bc.map_to_host() as d:
        with a.bf16.map_to_host() as ah:
            unsafe_memcpy(
                dest=d.unsafe_ptr().unsafe_bitcast[UInt8](),
                src=ah.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=na * 2,
            )
        with b.bf16.map_to_host() as bh:
            unsafe_memcpy(
                dest=(d.unsafe_ptr().unsafe_offset(na)).unsafe_bitcast[UInt8](),
                src=bh.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=nb * 2,
            )
    return QMat(
        bc^,
        ctx.enqueue_create_buffer[DType.uint32](1),
        ctx.enqueue_create_buffer[DType.float32](1),
        False,
    )


def concat_bias(
    ctx: DeviceContext, var a: DevBuf, var b: DevBuf, na: Int, nb: Int
) raises -> DevBuf:
    """Concatenate two f32 bias vectors (for the fused QKV bias).

    Args:
        ctx: The device context to allocate and enqueue on.
        a: The first f32 bias vector (consumed).
        b: The second f32 bias vector (consumed).
        na: The length of `a`.
        nb: The length of `b`.

    Returns:
        A fresh f32 device buffer holding `a` followed by `b`.

    Raises:
        Error: on a device allocation or buffer-mapping failure.
    """
    var c = ctx.enqueue_create_buffer[DType.float32](na + nb)
    with c.map_to_host() as d:
        with a.map_to_host() as ah:
            unsafe_memcpy(
                dest=d.unsafe_ptr().unsafe_bitcast[UInt8](),
                src=ah.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=na * 4,
            )
        with b.map_to_host() as bh:
            unsafe_memcpy(
                dest=(d.unsafe_ptr().unsafe_offset(na)).unsafe_bitcast[UInt8](),
                src=bh.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=nb * 4,
            )
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
                    _ = parse_string(buf, pos)  # tensor name (ignored)
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


def gather_tensors(
    path: String,
) raises -> Tuple[List[TensorEntry], List[String]]:
    """Resolve a checkpoint into (entries, per-entry file path). `path` is either a
    single .safetensors file (0.5B in the HF cache is one blob) or a directory
    holding sharded shards + model.safetensors.index.json (3B). Detection: try to
    open the index inside `path`-as-dir; absent → treat `path` as a single file.

    Args:
        path: A single `.safetensors` file, or a directory holding shards plus a
            `model.safetensors.index.json`.

    Returns:
        A tuple of (tensor entries with absolute file offsets, the file path each
        entry lives in).

    Raises:
        Error: on I/O failure or a malformed header / index.
    """
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
