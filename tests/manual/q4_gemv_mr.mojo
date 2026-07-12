"""Bandwidth-optimal M=1 int4 GEMV prototype: R output rows per simdgroup.

The shipping decode GEMV (`matmul_q4_kernel`) assigns ONE simdgroup per output
element: for K=2048 each lane does a single 256-bit weight load, a dependent
unpack chain, then a full warp reduction — one load in flight per lane, x
re-read for every output row, reduction overhead per 2048 weights. Measured
consequence: it is load-ISSUE limited, not bandwidth limited (an M2 Pro with
200 GB/s decodes SLOWER than a base M4 with 120 GB/s; mlx-lm's qmv reaches
~142 GB/s = 71% of ceiling on the same M2 Pro while we sit near ~10%).

This prototype gives each simdgroup R adjacent output rows:
  * R independent 256-bit weight streams in flight per lane (memory-level
    parallelism instead of one dependent chain),
  * each x oct is loaded once and reused for all R rows (x traffic /R),
  * warp-reduction cost amortized R× per K-sweep.

Same storage contract as model.QMat (group-128 int4: u32-packed nibbles,
per-group f32 scales; one scale per 64-element oct is exact since 64 | 128).

Validates every variant vs a CPU reference on odd shapes (incl. N % R != 0),
then benches the shipping kernel vs R ∈ {2,4,8} across the Qwen2.5-3B decode
shapes. Weight-GB/s = packed bytes / kernel time (the metric decode lives on).

    pixi run q4-gemv-mr    (GPU only, no model weights, ~35 MB peak — small
                            enough to run even when big allocations are blocked)
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import global_idx, WARP_SIZE
from std.gpu.primitives.warp import sum as warp_sum
from std.gpu.host import DeviceContext
from std.collections import InlineArray
from layout import TileTensor, TensorLayout, row_major
from kernels import matmul_q4_kernel

comptime Q4_GROUP = 128
comptime Q4_SHIFT = 7  # log2(128)
comptime BLOCK = 128
comptime _Q4_SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)


def matmul_q4_mr_kernel[
    LT: TensorLayout, R: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K: Int,
    N: Int,
    NG: Int,
    use_bias: Int,
):
    """Decode GEMV (M=1), R output rows per simdgroup.

    Per loop step a lane issues R independent 256-bit weight loads (one per
    row) before any unpack math, loads the shared x oct once, then unpacks and
    accumulates R partial dots. Row guards are warp-uniform (n0 and r are the
    same for every lane), so there is no per-lane divergence around the SIMD
    ops (see the divergent-branch miscompile note in the GEMM kernels).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Output rows per simdgroup (2, 4, or 8).

    Args:
        X: Input activations [1, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [1, N] (f32).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    comptime assert X.flat_rank == 1
    var sg = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    var n0 = sg * R
    if n0 >= N:
        return
    var words = K // 8
    var octs = words // 8  # one oct = 8 u32 = 64 weights
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var acc = InlineArray[Float32, R](fill=0.0)
    for q in range(lane, octs, WARP_SIZE):
        var k0 = q * 64
        # R independent weight streams first — all loads in flight before math.
        var w8 = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:  # warp-uniform guard
                w8[r] = (pp + (n0 + r) * words + q * 8).load[width=8]()
        # The shared x oct, loaded once for all R rows.
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            xv[j] = (xp + k0 + j * 8).load[width=8]()
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                var racc = Float32(0.0)
                comptime for j in range(8):
                    var nibs = (SIMD[DType.uint32, 8](w8[r][j]) >> _Q4_SHIFTS) & 0xF
                    var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
                    racc += (qf * xv[j]).reduce_add()
                acc[r] += racc * rebind[Float32](s)
    comptime for r in range(R):
        var total = warp_sum(acc[r])
        if lane == 0 and n0 + r < N:
            if use_bias != 0:
                total += rebind[Scalar[DType.float32]](B[n0 + r])
            Y[n0 + r] = rebind[Y.ElementType](total)


# ── host helpers (same idioms as q4_kernels.mojo) ────────────────────────────


def prng(mut state: UInt64) -> Float32:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float32(Int(state % 2000) - 1000) / 1000.0  # ~[-1,1]


def quantize_g128(
    W: List[Float32], N: Int, K: Int
) raises -> Tuple[List[UInt32], List[Float32], List[Float32]]:
    """Returns (packed u32[N*K/8], scales f32[N*NG], dequant-reference f32[N*K]).
    """
    var NG = K // Q4_GROUP
    var packed = List[UInt32]()
    for _ in range(N * K // 8):
        packed.append(0)
    var scales = List[Float32]()
    for _ in range(N * NG):
        scales.append(0.0)
    var deq = List[Float32]()
    for _ in range(N * K):
        deq.append(0.0)
    for n in range(N):
        for g in range(NG):
            var amax = Float32(0.0)
            for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                var a = W[n * K + k]
                if a < 0.0:
                    a = -a
                if a > amax:
                    amax = a
            var s = amax / 7.0 if amax > 0.0 else Float32(1.0)
            scales[n * NG + g] = s
            var inv = 1.0 / s
            for k in range(g * Q4_GROUP, (g + 1) * Q4_GROUP):
                var q = W[n * K + k] * inv
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


def check[R: Int](ctx: DeviceContext, K: Int, N: Int) raises:
    """Validate the R-row kernel vs a CPU reference (M=1, bias on)."""
    var NG = K // Q4_GROUP
    var seed = UInt64(0x1234567 + R * 7 + N * 13 + K)
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
    var packed = qz[0].copy()
    var scales = qz[1].copy()
    var deq = qz[2].copy()

    var cpuref = List[Float32]()
    for n in range(N):
        var acc = Float64(0.0)
        for k in range(K):
            acc += Float64(Xh[k]) * Float64(deq[n * K + k])
        cpuref.append(Float32(acc) + Bh[n])

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

    var xt = TileTensor(xb, row_major(K))
    var pt = TileTensor(pb, row_major(N * K // 8))
    var st = TileTensor(sb, row_major(N * NG))
    var bt = TileTensor(bb, row_major(N))
    var yt = TileTensor(yb, row_major(N))

    comptime k = matmul_q4_mr_kernel[type_of(row_major(1)), R]
    var warps = ceildiv(N, R)
    ctx.enqueue_function[k](
        xt, pt, st, bt, yt, K, N, NG, 1,
        grid_dim=ceildiv(warps * WARP_SIZE, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()

    var md = Float64(0.0)
    var mr = Float64(0.0)
    with yb.map_to_host() as h:
        var t = TileTensor(h, row_major(N))
        for i in range(N):
            var d = Float64(rebind[Scalar[DType.float32]](t[i]) - cpuref[i])
            if d < 0:
                d = -d
            if d > md:
                md = d
            var rf = Float64(cpuref[i])
            if rf < 0:
                rf = -rf
            if rf > mr:
                mr = rf
    var rel = md / mr if mr > 0 else md
    print(
        "  R=", R, " K=", K, " N=", N, " : max|Δ|=", md, " rel=", rel, " ",
        "OK" if rel < 1.0e-3 else "FAIL", sep="",
    )
    if rel >= 1.0e-3:
        raise Error("mr kernel mismatch")


def bench(ctx: DeviceContext, K: Int, N: Int) raises:
    """Shipping kernel vs R ∈ {2,4,8} on one decode shape; weight-GB/s."""
    var NG = K // Q4_GROUP
    var iters = 200
    var xb = ctx.enqueue_create_buffer[DType.float32](K)
    var pb = ctx.enqueue_create_buffer[DType.uint32](N * K // 8)
    var sb = ctx.enqueue_create_buffer[DType.float32](N * NG)
    var bb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](N)
    xb.enqueue_fill(0.5)
    pb.enqueue_fill(0x99999999)
    sb.enqueue_fill(0.01)
    bb.enqueue_fill(0.0)
    yb.enqueue_fill(0.0)
    var xt = TileTensor(xb, row_major(K))
    var pt = TileTensor(pb, row_major(N * K // 8))
    var st = TileTensor(sb, row_major(N * NG))
    var bt = TileTensor(bb, row_major(N))
    var yt = TileTensor(yb, row_major(N))
    var wbytes = Float64(N * K) / 2.0

    # shipping kernel (one simdgroup per output element)
    comptime kq = matmul_q4_kernel[type_of(row_major(1))]
    var grid0 = ceildiv(N * WARP_SIZE, BLOCK)
    for _ in range(5):
        ctx.enqueue_function[kq](
            xt, pt, st, bt, yt, 1, K, N, NG, 0,
            grid_dim=grid0, block_dim=BLOCK,
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[kq](
            xt, pt, st, bt, yt, 1, K, N, NG, 0,
            grid_dim=grid0, block_dim=BLOCK,
        )
    ctx.synchronize()
    var ship_ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
    print(
        "  K=", K, " N=", N,
        "  ship ", ship_ms, " ms  ", wbytes / (ship_ms * 1.0e6), " GB/s",
        sep="",
    )

    comptime for RR in [2, 4, 8]:
        comptime km = matmul_q4_mr_kernel[type_of(row_major(1)), RR]
        var warps = ceildiv(N, RR)
        var grid = ceildiv(warps * WARP_SIZE, BLOCK)
        for _ in range(5):
            ctx.enqueue_function[km](
                xt, pt, st, bt, yt, K, N, NG, 0,
                grid_dim=grid, block_dim=BLOCK,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(iters):
            ctx.enqueue_function[km](
                xt, pt, st, bt, yt, K, N, NG, 0,
                grid_dim=grid, block_dim=BLOCK,
            )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t1) / Float64(iters) / 1.0e6
        print(
            "            R=", RR, "  ", ms, " ms  ",
            wbytes / (ms * 1.0e6), " GB/s  (", ship_ms / ms, "x ship)",
            sep="",
        )


def main() raises:
    comptime if not has_accelerator():
        print("no GPU — skipping")
        return
    var ctx = DeviceContext()
    print("correctness (vs CPU reference, odd shapes incl. N % R != 0):")
    check[2](ctx, 256, 129)
    check[4](ctx, 256, 130)
    check[8](ctx, 512, 257)
    check[4](ctx, 2048, 512)
    print("bench (Qwen2.5-3B decode shapes; weight-GB/s, 200 iters):")
    bench(ctx, 2048, 2048)    # q/o proj
    bench(ctx, 2048, 256)     # kv proj (tiny N)
    bench(ctx, 2048, 2560)    # small-N stress (memory: caps ~43 GB/s)
    bench(ctx, 2048, 11008)   # gate/up
    bench(ctx, 11008, 2048)   # down
