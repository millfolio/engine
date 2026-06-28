"""Standalone validation for the group-128 int4 path (decode GEMV + simd prefill
GEMM + host quantizer), before wiring any of it into model.mojo.

Storage: per matrix W[N,K] (K a multiple of 128) →
  packed: u32[N*K/8] — 8 signed nibbles/word (q+8 ∈ 0..15); nibble at linear index
          `lin` lives in word lin>>3 at bit-shift 4*(lin&7).
  scales: f32[N*(K/128)] — symmetric RTN per 128-group, qmax=7.
Dequant: deq = (nibble-8) * scale[row, k/128].

The decode GEMV is vectorized: each lane loads one u32 (8 weights) for coalesced
128-byte transactions, and folds the group scale once per word (a 128-group is a
multiple of 8, so an 8-aligned word never straddles a group). The simd prefill
GEMM reuses matmul_simd_kernel verbatim except the W-staging dequantizes int4 —
the matmul math is unchanged, so the 4.5x prefill is preserved. Validates both
kernels vs a CPU reference and benches GEMV vs bf16.

    pixi run q4-kernels
"""

from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx, barrier, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import sum as warp_sum
from std.gpu.host import DeviceContext
from std.collections import InlineArray
from std.memory import stack_allocation
from std.sys.info import external_call
from layout import TileTensor, TensorLayout, row_major
from kernels import (
    matmul_kernel,
    matmul_q4_batch_kernel,
    matmul_q4_small_kernel,
    _SM_BN,
    _SM_TPB,
)

comptime Q4_GROUP = 128
comptime Q4_SHIFT = 7  # log2(128)
comptime BLOCK = 256

comptime _FRAG = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]
comptime _LD = "air.simdgroup_matrix_8x8_load.v64f32.p3f32"
comptime _MAC = (
    "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32"
)
comptime _ST = "air.simdgroup_matrix_8x8_store.v64f32.p3f32"
comptime SG_BM = 32
comptime SG_BN = 32
comptime SG_NCT = 4
comptime SG_TPB = 128


@always_inline
def q4_deq[
    LT: TensorLayout
](
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int,
    k: Int,
    K: Int,
    NG: Int,
) -> Float32:
    """Dequant a single weight (n,k) — used by the prefill GEMM staging."""
    comptime assert P.flat_rank == 1
    var lin = n * K + k
    var w = Int(rebind[Scalar[DType.uint32]](P[lin >> 3]))
    var nib = (w >> ((lin & 7) * 4)) & 0xF
    var s = rebind[Scalar[DType.float32]](S[n * NG + (k >> Q4_SHIFT)])
    return Float32(nib - 8) * s


def q4_gemv_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    NG: Int,
    use_bias: Int,
):
    """Vectorized decode GEMV: one warp per output, each lane consumes one u32
    (8 weights) per step → coalesced 128-byte loads, scale folded per word."""
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var words = K // 8
    var rowword = n * words
    var xbase = m * K
    comptime SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)
    var acc = Float32(0.0)
    for w in range(lane, words, WARP_SIZE):
        var packed = rebind[Scalar[DType.uint32]](P[rowword + w])
        var k0 = w * 8
        var s = rebind[Scalar[DType.float32]](S[n * NG + (k0 >> Q4_SHIFT)])
        # unpack all 8 nibbles (weight order) with vector ops, not a scalar loop
        var nibs = (SIMD[DType.uint32, 8](packed) >> SHIFTS) & 0xF
        var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
        var xv = SIMD[DType.float32, 8](0.0)
        for o in range(8):
            xv[o] = rebind[Scalar[DType.float32]](X[xbase + k0 + o])
        acc += (qf * xv).reduce_add() * s
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def q4_simd_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    NG: Int,
    use_bias: Int,
):
    """Matmul_simd_kernel with int4-dequant W-staging; matmul math identical."""
    comptime assert X.flat_rank == 1
    var tid = thread_idx.x
    var sg = Int(tid) // 32
    var m0 = Int(block_idx.y) * SG_BM
    var n0 = Int(block_idx.x) * SG_BN

    var sA = stack_allocation[
        SG_BM * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var sB = stack_allocation[
        8 * SG_BN, Float32, address_space=AddressSpace.SHARED
    ]()
    var sC = stack_allocation[
        SG_BM * SG_BN, Float32, address_space=AddressSpace.SHARED
    ]()

    var dims = _V2(8, 8)
    var lay8 = _V2(1, 8)
    var layN = _V2(1, SG_BN)
    var origin = _V2(0, 0)
    var acc = InlineArray[_FRAG, SG_NCT](fill=_FRAG(0.0))

    var kt = 0
    while kt < K:
        for j in range(SG_BM * 8 // SG_TPB):
            var c = Int(tid) + SG_TPB * j
            var r = c // 8
            var col = c % 8
            var mm = m0 + r
            var kk = kt + col
            var xv = Float32(0.0)
            if mm < M and kk < K:
                xv = rebind[Scalar[DType.float32]](X[mm * K + kk])
            sA[c] = xv
        for j in range(8 * SG_BN // SG_TPB):
            var c = Int(tid) + SG_TPB * j
            var kr = c // SG_BN
            var nl = c % SG_BN
            var nn = n0 + nl
            var kk = kt + kr
            var wv = Float32(0.0)
            if nn < N and kk < K:
                wv = q4_deq(P, S, nn, kk, K, NG)
            sB[c] = wv
        barrier()
        var fa = external_call[_LD, _FRAG](sA + sg * 8 * 8, dims, lay8, origin)
        for ct in range(SG_NCT):
            var fb = external_call[_LD, _FRAG](sB + ct * 8, dims, layN, origin)
            acc[ct] = external_call[_MAC, _FRAG](fa, fb, acc[ct])
        barrier()
        kt += 8

    for ct in range(SG_NCT):
        external_call[_ST, NoneType](
            acc[ct], sC + sg * 8 * SG_BN + ct * 8, dims, layN, origin
        )
    barrier()
    for j in range(SG_BM * SG_BN // SG_TPB):
        var c = Int(tid) + SG_TPB * j
        var r = c // SG_BN
        var nl = c % SG_BN
        var mm = m0 + r
        var nn = n0 + nl
        if mm < M and nn < N:
            var v = sC[c]
            if use_bias != 0:
                v += rebind[Scalar[DType.float32]](B[nn])
            Y[mm * N + nn] = rebind[Y.ElementType](v)


# ── host helpers ──────────────────────────────────────────────────────────────


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


def maxabs_err(a: List[Float32], b: List[Float32]) -> Tuple[Float64, Float64]:
    var md = Float64(0.0)
    var mr = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i] - b[i])
        if d < 0:
            d = -d
        if d > md:
            md = d
        var r = Float64(b[i])
        if r < 0:
            r = -r
        if r > mr:
            mr = r
    return (md, mr)


def check(
    ctx: DeviceContext, name: String, M: Int, K: Int, N: Int, mode: Int
) raises:
    # mode: 0 = local decode GEMV, 1 = local simd GEMM, 2 = library batched GEMV
    var NG = K // Q4_GROUP
    var seed = UInt64(0x1234567 + M * 7 + N * 13 + K)
    var Wh = List[Float32]()
    for _ in range(N * K):
        Wh.append(prng(seed))
    for n in range(N):
        Wh[n * K + (n * 37) % K] = Float32(6.0)  # outliers
    var Xh = List[Float32]()
    for _ in range(M * K):
        Xh.append(prng(seed))
    var Bh = List[Float32]()
    for _ in range(N):
        Bh.append(prng(seed))

    var qz = quantize_g128(Wh, N, K)
    var packed = qz[0].copy()
    var scales = qz[1].copy()
    var deq = qz[2].copy()

    var cpuref = List[Float32]()
    for m in range(M):
        for n in range(N):
            var acc = Float64(0.0)
            for k in range(K):
                acc += Float64(Xh[m * K + k]) * Float64(deq[n * K + k])
            cpuref.append(Float32(acc) + Bh[n])

    var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
    var pb = ctx.enqueue_create_buffer[DType.uint32](N * K // 8)
    var sb = ctx.enqueue_create_buffer[DType.float32](N * NG)
    var bb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
    with xb.map_to_host() as h:
        var t = TileTensor(h, row_major(M * K))
        for i in range(M * K):
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

    var xt = TileTensor(xb, row_major(M * K))
    var pt = TileTensor(pb, row_major(N * K // 8))
    var st = TileTensor(sb, row_major(N * NG))
    var bt = TileTensor(bb, row_major(N))
    var yt = TileTensor(yb, row_major(M * N))

    if mode == 1:
        comptime k = q4_simd_kernel[type_of(row_major(1))]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            M,
            K,
            N,
            NG,
            1,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    elif mode == 2:
        comptime k = matmul_q4_batch_kernel[type_of(row_major(1))]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            M,
            K,
            N,
            NG,
            1,
            grid_dim=ceildiv(N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif mode == 3:
        comptime k = matmul_q4_small_kernel[type_of(row_major(1))]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            M,
            K,
            N,
            NG,
            1,
            grid_dim=(ceildiv(N, _SM_BN), 1),
            block_dim=_SM_TPB,
        )
    else:
        comptime k = q4_gemv_kernel[type_of(row_major(1))]
        ctx.enqueue_function[k](
            xt,
            pt,
            st,
            bt,
            yt,
            M,
            K,
            N,
            NG,
            1,
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    ctx.synchronize()

    var got = List[Float32]()
    with yb.map_to_host() as h:
        var t = TileTensor(h, row_major(M * N))
        for i in range(M * N):
            got.append(rebind[Scalar[DType.float32]](t[i]))
    var e = maxabs_err(got, cpuref)
    var rel = e[0] / e[1] if e[1] > 0 else e[0]
    print(
        "  ",
        name,
        " M=",
        M,
        " K=",
        K,
        " N=",
        N,
        " : max|Δ|=",
        e[0],
        " rel=",
        rel,
        " ",
        "OK" if rel < 1.0e-3 else "FAIL",
        sep="",
    )


def bench_speed(ctx: DeviceContext, name: String, K: Int, N: Int) raises:
    var M = 1
    var NG = K // Q4_GROUP
    var iters = 200
    var grid = ceildiv(M * N * WARP_SIZE, BLOCK)

    var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
    var wb = ctx.enqueue_create_buffer[DType.uint16](N * K)
    var bb = ctx.enqueue_create_buffer[DType.float32](1)
    var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
    xb.enqueue_fill(0.5)
    wb.enqueue_fill(0x3F80)
    yb.enqueue_fill(0.0)
    var xt = TileTensor(xb, row_major(M * K))
    var wt = TileTensor(wb, row_major(N * K))
    var bt = TileTensor(bb, row_major(1))
    var yt = TileTensor(yb, row_major(M * N))
    comptime kb = matmul_kernel[type_of(row_major(1))]
    for _ in range(5):
        ctx.enqueue_function[kb](
            xt, wt, bt, yt, M, K, N, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[kb](
            xt, wt, bt, yt, M, K, N, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var bf_ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6

    var pb = ctx.enqueue_create_buffer[DType.uint32](N * K // 8)
    var sb = ctx.enqueue_create_buffer[DType.float32](N * NG)
    pb.enqueue_fill(0x99999999)
    sb.enqueue_fill(0.01)
    var pt = TileTensor(pb, row_major(N * K // 8))
    var st = TileTensor(sb, row_major(N * NG))
    comptime kq = q4_gemv_kernel[type_of(row_major(1))]
    for _ in range(5):
        ctx.enqueue_function[kq](
            xt, pt, st, bt, yt, M, K, N, NG, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[kq](
            xt, pt, st, bt, yt, M, K, N, NG, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var q_ms = Float64(perf_counter_ns() - t1) / Float64(iters) / 1.0e6
    print(
        "  ",
        name,
        " K=",
        K,
        " N=",
        N,
        " : bf16 ",
        bf_ms,
        " ms  q4 ",
        q_ms,
        " ms  speedup ",
        bf_ms / q_ms,
        "x",
        sep="",
    )


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU")
    var ctx = DeviceContext()
    print("=== int4 g128 kernel correctness (vs CPU dequant reference) ===")
    print("-- decode GEMV (M=1):")
    check(ctx, "gemv", 1, 256, 64, 0)
    check(ctx, "gemv", 1, 896, 128, 0)
    check(ctx, "gemv", 1, 2048, 320, 0)
    check(ctx, "gemv", 1, 11008, 256, 0)
    print("-- prefill simd GEMM (M>1):")
    check(ctx, "simd", 8, 256, 64, 1)
    check(ctx, "simd", 40, 896, 137, 1)
    check(ctx, "simd", 64, 2048, 320, 1)
    check(ctx, "simd", 33, 11008, 256, 1)
    print("-- batched GEMV (small M, library kernel):")
    check(ctx, "batch", 2, 256, 64, 2)
    check(ctx, "batch", 5, 896, 137, 2)
    check(ctx, "batch", 8, 2048, 320, 2)
    check(ctx, "batch", 4, 11008, 256, 2)
    print("-- small-M int4 GEMM (1 MMA row tile, library kernel):")
    check(ctx, "small", 1, 256, 64, 3)
    check(ctx, "small", 5, 896, 137, 3)
    check(ctx, "small", 8, 2048, 320, 3)
    check(ctx, "small", 4, 11008, 256, 3)
    check(ctx, "small", 8, 2048, 130, 3)  # N not a multiple of _SM_BN

    print("\n=== decode GEMV speed: bf16 vs int4 g128 (M=1, 3B shapes) ===")
    bench_speed(ctx, "q/o  ", 2048, 2048)
    bench_speed(ctx, "kv   ", 2048, 256)
    bench_speed(ctx, "gate ", 2048, 11008)
    bench_speed(ctx, "up   ", 2048, 11008)
    bench_speed(ctx, "down ", 11008, 2048)
    bench_speed(ctx, "lmhd ", 2048, 151936)
