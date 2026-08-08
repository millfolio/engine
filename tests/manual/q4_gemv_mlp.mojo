"""Decode GEMV memory-level-parallelism race (the issue-bound fix, take 2).

The M2 Pro bench proved decode is still instruction-issue-bound: our GEMVs
run the same absolute speed on M4 (120 GB/s) and M2 Pro (200 GB/s) while
MLX rides bandwidth (52 -> 76 tok/s). The shipping mrw kernel keeps R
weight streams in flight per lane but its inner loop is ALU-heavy (8
horizontal reduce_adds per 64-element oct per row) and has no cross-
iteration pipelining. Variants raced here, all vs the shipping
matmul_q4_mrw_kernel and a CPU reference:

  mrw[R,W]   : shipping kernel at higher R (more row streams per lane)
  va[R,W]    : vector-accumulate — one 8-wide FMA per word, ONE reduce per
               oct per row (vs 8 mul+reduce+add), direct u32->f32 dequant
  pf[R,W]    : weight prefetch — next oct's R loads issued before this
               oct's unpack (cross-iteration MLP, weights only; x is hot)
  vp[R,W]    : va + pf combined
  w16[R,W]   : 2 octs (= exactly one 128-group) per lane per iteration —
               16-word loads halve load-issue per byte, one scale fetch,
               vector accumulate

Weight-GB/s = packed bytes / kernel time (the decode metric). GPU only, no
model weights (~160 MB peak incl. the lm_head shape). Run on BOTH machines:

    pixi run gemv-mlp-race
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.gpu.primitives.warp import sum as warp_sum
from max.gpu.host import DeviceContext, DeviceBuffer
from std.collections import InlineArray
from std.memory import stack_allocation
from layout import TileTensor, TensorLayout, row_major
from kernels import matmul_q4_mrw_kernel, Q4_GROUP, Q4_SHIFT

comptime _SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)
comptime LayT = type_of(row_major(1))


# ── variants ──────────────────────────────────────────────────────────────────


def gemv_va_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Mrw structure, vector accumulate: per oct per row, 8 FMAs into an
    8-wide register and ONE horizontal reduce (the shipping kernel does 8
    mul+reduce+add chains). Dequant casts u32 nibbles straight to f32
    (exact for 0..15) instead of via int32."""
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
        var k0 = q * 64
        var w8 = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:
                w8[r] = pp.unsafe_offset((n0 + r) * words + q * 8).unsafe_load[
                    width=8
                ]()
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            xv[j] = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                var vacc = SIMD[DType.float32, 8](0)
                comptime for j in range(8):
                    var nibs = (
                        SIMD[DType.uint32, 8](w8[r][j]) >> _SHIFTS
                    ) & 0xF
                    var qf = nibs.cast[DType.float32]() - 8.0
                    vacc = qf.fma(xv[j], vacc)
                acc[r] += vacc.reduce_add() * rebind[Float32](s)
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def gemv_vp_kernel[
    LT: TensorLayout, R: Int, W: Int, VA: Bool
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Mrw + cross-iteration WEIGHT prefetch: the next oct's R weight loads
    are issued before this oct's unpack runs, so R loads are always in
    flight across the dependent ALU chain. VA selects the vector-accumulate
    inner loop (vp) or the shipping mul+reduce chain (pf)."""
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    comptime STRIDE = W * WARP_SIZE
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    var q = w * WARP_SIZE + lane
    var w8c = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
    if q < octs:
        comptime for r in range(R):
            if n0 + r < N:
                w8c[r] = pp.unsafe_offset((n0 + r) * words + q * 8).unsafe_load[
                    width=8
                ]()
    while q < octs:
        var qn = q + STRIDE
        var w8n = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        if qn < octs:
            comptime for r in range(R):
                if n0 + r < N:
                    w8n[r] = pp.unsafe_offset(
                        (n0 + r) * words + qn * 8
                    ).unsafe_load[width=8]()
        var k0 = q * 64
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            xv[j] = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                comptime if VA:
                    var vacc = SIMD[DType.float32, 8](0)
                    comptime for j in range(8):
                        var nibs = (
                            SIMD[DType.uint32, 8](w8c[r][j]) >> _SHIFTS
                        ) & 0xF
                        var qf = nibs.cast[DType.float32]() - 8.0
                        vacc = qf.fma(xv[j], vacc)
                    acc[r] += vacc.reduce_add() * rebind[Float32](s)
                else:
                    var racc = Float32(0.0)
                    comptime for j in range(8):
                        var nibs = (
                            SIMD[DType.uint32, 8](w8c[r][j]) >> _SHIFTS
                        ) & 0xF
                        var qf = (nibs.cast[DType.int32]() - 8).cast[
                            DType.float32
                        ]()
                        racc += (qf * xv[j]).reduce_add()
                    acc[r] += racc * rebind[Float32](s)
        comptime for r in range(R):
            w8c[r] = w8n[r]
        q = qn
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def gemv_w16_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Mrw with 2 octs (= exactly one 128-elem group) per lane-iteration:
    one 16-word (64 B) load per row per iteration halves load-issue per
    byte, the group scale is fetched once, and accumulation is 16 8-wide
    FMAs + one reduce. Requires K % 128 == 0 (the q4 layout guarantees it).
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var groups = K // Q4_GROUP
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    for g in range(w * WARP_SIZE + lane, groups, W * WARP_SIZE):
        var k0 = g * Q4_GROUP
        var w16 = InlineArray[SIMD[DType.uint32, 16], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:
                w16[r] = pp.unsafe_offset(
                    (n0 + r) * words + g * 16
                ).unsafe_load[width=16]()
        var xv = InlineArray[SIMD[DType.float32, 8], 16](fill=0)
        comptime for j in range(16):
            xv[j] = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + g]
                var vacc = SIMD[DType.float32, 8](0)
                comptime for j in range(16):
                    var nibs = (
                        SIMD[DType.uint32, 8](w16[r][j]) >> _SHIFTS
                    ) & 0xF
                    var qf = nibs.cast[DType.float32]() - 8.0
                    vacc = qf.fma(xv[j], vacc)
                acc[r] += vacc.reduce_add() * rebind[Float32](s)
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


# ── host scaffolding (validate vs CPU, then bench weight-GB/s) ───────────────


def prng(mut seed: UInt64) -> Float32:
    seed = seed * 6364136223846793005 + 1442695040888963407
    return Float32(Int((seed >> 33) % 2001) - 1000) / Float32(700.0)


def quantize_g128(
    Wh: List[Float32], N: Int, K: Int
) raises -> Tuple[List[UInt32], List[Float32], List[Float32]]:
    """Symmetric RTN group-128 int4 (the engine's storage contract)."""
    var packed = List[UInt32](length=N * K // 8, fill=0)
    var scales = List[Float32]()
    var deq = List[Float32](length=N * K, fill=0.0)
    var NG = K // Q4_GROUP
    for n in range(N):
        for g in range(NG):
            var amax = Float32(0.0)
            for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                var a = Wh[n * K + k]
                if a < 0:
                    a = -a
                if a > amax:
                    amax = a
            var s = amax / 7.0 if amax > 0 else Float32(1.0)
            scales.append(s)
            for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                var q = Wh[n * K + k] / s
                var half = Float32(0.5) if q >= 0.0 else Float32(-0.5)
                var qr = Int(q + half)
                if qr > 7:
                    qr = 7
                elif qr < -7:
                    qr = -7
                deq[n * K + k] = Float32(qr) * s
                var lin = n * K + k
                packed[lin >> 3] = packed[lin >> 3] | (
                    UInt32(qr + 8) << UInt32((lin & 7) * 4)
                )
    return (packed^, scales^, deq^)


struct Bufs(Movable):
    var xb: DeviceBuffer[DType.float32]
    var pb: DeviceBuffer[DType.uint32]
    var sb: DeviceBuffer[DType.float32]
    var bb: DeviceBuffer[DType.float32]
    var yb: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var xb: DeviceBuffer[DType.float32],
        var pb: DeviceBuffer[DType.uint32],
        var sb: DeviceBuffer[DType.float32],
        var bb: DeviceBuffer[DType.float32],
        var yb: DeviceBuffer[DType.float32],
    ):
        self.xb = xb^
        self.pb = pb^
        self.sb = sb^
        self.bb = bb^
        self.yb = yb^


def upload(
    ctx: DeviceContext,
    Xh: List[Float32],
    packed: List[UInt32],
    scales: List[Float32],
    Bh: List[Float32],
    K: Int,
    N: Int,
) raises -> Bufs:
    var NG = K // Q4_GROUP
    var xb = ctx.enqueue_create_buffer[DType.float32](K)
    var pb = ctx.enqueue_create_buffer[DType.uint32](N * K // 8)
    var sb = ctx.enqueue_create_buffer[DType.float32](N * NG)
    var bb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](N)
    with xb.map_to_host() as h:
        var t = TileTensor(h, row_major(K))
        for i in range(K):
            t[i] = rebind[t.ElementType](Xh[i])
    with pb.map_to_host() as h:
        var t = TileTensor(h, row_major(N * K // 8))
        for i in range(N * K // 8):
            t[i] = rebind[t.ElementType](packed[i])
    with sb.map_to_host() as h:
        var t = TileTensor(h, row_major(N * NG))
        for i in range(N * NG):
            t[i] = rebind[t.ElementType](scales[i])
    with bb.map_to_host() as h:
        var t = TileTensor(h, row_major(N))
        for i in range(N):
            t[i] = rebind[t.ElementType](Bh[i])
    yb.enqueue_fill(0.0)
    return Bufs(xb^, pb^, sb^, bb^, yb^)


comptime VMRW = 0
comptime VVA = 1
comptime VPF = 2
comptime VVP = 3
comptime VW16 = 4


def launch[
    VAR: Int, R: Int, W: Int
](ctx: DeviceContext, mut b: Bufs, K: Int, N: Int) raises:
    var NG = K // Q4_GROUP
    var xt = TileTensor(b.xb, row_major(K))
    var pt = TileTensor(b.pb, row_major(N * K // 8))
    var st = TileTensor(b.sb, row_major(N * NG))
    var bt = TileTensor(b.bb, row_major(N))
    var yt = TileTensor(b.yb, row_major(N))
    var grid = ceildiv(N, R)
    comptime if VAR == VMRW:
        comptime k = matmul_q4_mrw_kernel[LayT, R, W]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=grid,
            block_dim=W * WARP_SIZE,
        )
    elif VAR == VVA:
        comptime k = gemv_va_kernel[LayT, R, W]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=grid,
            block_dim=W * WARP_SIZE,
        )
    elif VAR == VPF:
        comptime k = gemv_vp_kernel[LayT, R, W, False]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=grid,
            block_dim=W * WARP_SIZE,
        )
    elif VAR == VVP:
        comptime k = gemv_vp_kernel[LayT, R, W, True]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=grid,
            block_dim=W * WARP_SIZE,
        )
    else:
        comptime k = gemv_w16_kernel[LayT, R, W]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=grid,
            block_dim=W * WARP_SIZE,
        )


def check[
    VAR: Int, R: Int, W: Int
](ctx: DeviceContext, name: String, K: Int, N: Int) raises:
    var seed = UInt64(0xACE + R * 7 + N * 13 + K)
    var Wh = List[Float32]()
    for _ in range(N * K):
        Wh.append(prng(seed))
    for n in range(N):
        Wh[n * K + (n * 37) % K] = Float32(6.0)  # outliers
    var Xh = List[Float32]()
    for _ in range(K):
        Xh.append(prng(seed))
    var Bh = List[Float32]()
    for _ in range(N):
        Bh.append(prng(seed))
    var qz = quantize_g128(Wh, N, K)
    var cpuref = List[Float32]()
    for n in range(N):
        var acc = Float64(0.0)
        for k in range(K):
            acc += Float64(Xh[k]) * Float64(qz[2][n * K + k])
        cpuref.append(Float32(acc) + Bh[n])
    var b = upload(ctx, Xh, qz[0], qz[1], Bh, K, N)
    launch[VAR, R, W](ctx, b, K, N)
    ctx.synchronize()
    var md = Float64(0.0)
    var mr = Float64(0.0)
    with b.yb.map_to_host() as h:
        var t = TileTensor(h, row_major(N))
        for n in range(N):
            var g = Float64(rebind[Float32](t[n]))
            var c = Float64(cpuref[n])
            var d = g - c
            if d < 0:
                d = -d
            md = md if md > d else d
            var a = c if c >= 0 else -c
            mr = mr if mr > a else a
    var rel = md / mr if mr > 0 else md
    var verdict = String("OK") if rel < 1.0e-3 else String("FAIL")
    print(
        "  check ", name, " K=", K, " N=", N, " rel=", rel, " ", verdict, sep=""
    )
    if rel >= 1.0e-3:
        raise Error("validation failed: " + name)


def bench[
    VAR: Int, R: Int, W: Int
](ctx: DeviceContext, name: String, mut b: Bufs, K: Int, N: Int) raises:
    var iters = 200
    if N * K > 50_000_000:
        iters = 50
    for _ in range(10):
        launch[VAR, R, W](ctx, b, K, N)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        launch[VAR, R, W](ctx, b, K, N)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
    var gbs = (Float64(N) * Float64(K) / 2.0) / (ms * 1.0e6)
    print("    ", name, ": ", ms, " ms   ", gbs, " weight-GB/s", sep="")


def bench_shape(ctx: DeviceContext, label: String, K: Int, N: Int) raises:
    print("  ", label, " N=", N, " K=", K, sep="")
    var NG = K // Q4_GROUP
    var seed = UInt64(0xBEEF)
    var Xh = List[Float32]()
    for _ in range(K):
        Xh.append(prng(seed))
    var packed = List[UInt32](length=N * K // 8, fill=0x93A5C176)
    var scales = List[Float32](length=N * NG, fill=0.008)
    var Bh = List[Float32](length=N, fill=0.0)
    var b = upload(ctx, Xh, packed, scales, Bh, K, N)
    bench[VMRW, 2, 4](ctx, "mrw[2,4] (ship)", b, K, N)
    bench[VMRW, 4, 4](ctx, "mrw[4,4]       ", b, K, N)
    bench[VMRW, 8, 4](ctx, "mrw[8,4]       ", b, K, N)
    bench[VVA, 4, 4](ctx, "va [4,4]       ", b, K, N)
    bench[VPF, 4, 4](ctx, "pf [4,4]       ", b, K, N)
    bench[VVP, 4, 4](ctx, "vp [4,4]       ", b, K, N)
    bench[VVP, 8, 4](ctx, "vp [8,4]       ", b, K, N)
    bench[VW16, 4, 4](ctx, "w16[4,4]       ", b, K, N)
    bench[VW16, 8, 4](ctx, "w16[8,4]       ", b, K, N)


def main() raises:
    var ctx = DeviceContext()
    print("decode GEMV MLP race — validate, then weight-GB/s per variant")
    print("validation (odd shapes, bias on):")
    check[VVA, 4, 4](ctx, "va [4,4]", 384, 37)
    check[VPF, 4, 4](ctx, "pf [4,4]", 384, 37)
    check[VVP, 4, 4](ctx, "vp [4,4]", 1280, 130)
    check[VVP, 8, 4](ctx, "vp [8,4]", 384, 37)
    check[VW16, 4, 4](ctx, "w16[4,4]", 1280, 130)
    check[VW16, 8, 4](ctx, "w16[8,4]", 384, 37)
    print("bench (Qwen2.5-3B decode shapes + lm_head):")
    bench_shape(ctx, "qkv    ", 2048, 2560)
    bench_shape(ctx, "o_proj ", 2048, 2048)
    bench_shape(ctx, "gate_up", 2048, 22016)
    bench_shape(ctx, "down   ", 11008, 2048)
    bench_shape(ctx, "lm_head", 2048, 151936)
