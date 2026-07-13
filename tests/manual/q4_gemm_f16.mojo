"""f16-input prefill GEMM prototype: MLX steel-GEMM economics on our int4 path.

The shipping `matmul_simd_q4_kernel` runs f32 8×8 simdgroup-MMA at ~2.1 TFLOP/s;
MLX's qmm measures ~3.4 TFLOP/s on the same M4 (`.scratch/mlx_op_bench.py`).
Apple's MMA units run half-precision inputs at up to 2× f32 throughput, and the
mixed intrinsic (f16 A/B fragments, f32 accumulate) is confirmed working on M4
(`.scratch/mma_f16_probe.mojo`). This prototype is the shipping kernel with:
  * W dequantized to **f16** in threadgroup shared (half the shared traffic),
  * X fragments loaded f32 → cast f16,
  * `llvm.air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32`,
  * accumulators unchanged (f32) — quantization noise (int4 ±7 levels) dwarfs
    f16 input rounding, so quality risk is contained; the model-level gates
    (q4-validate, simd-parity) decide.

Validates vs a CPU reference on odd shapes, then benches TFLOP/s against the
shipping f32-MMA kernel on the 3B prefill shapes.

    pixi run q4-gemm-f16
"""

from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.sys.intrinsics import llvm_intrinsic
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx, barrier, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.collections import InlineArray
from std.memory import stack_allocation
from layout import TileTensor, TensorLayout, row_major
from kernels import (
    matmul_simd_q4_kernel,
    SG_BM,
    SG_BN,
    SG_TPB,
)

comptime Q4_GROUP = 128
comptime Q4_SHIFT = 7
comptime _Q4_SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)
comptime _Q4_BK = 32
comptime _MMA8 = 8
comptime _FRAG8 = 2
comptime _SG_SGM = SG_BM // 2
comptime _SG_SGN = SG_BN // 2
comptime _SG_NTM = _SG_SGM // _MMA8
comptime _SG_NTN = _SG_SGN // _MMA8


@always_inline
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _mma8x8_h(
    a: SIMD[DType.float16, _FRAG8],
    b: SIMD[DType.float16, _FRAG8],
    c: SIMD[DType.float32, _FRAG8],
) -> SIMD[DType.float32, _FRAG8]:
    """8×8×8 simdgroup MMA with f16 A/B fragments and f32 accumulate."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32",
        SIMD[DType.float32, _FRAG8],
    ](a, b, c)


def matmul_simd_q4_f16_kernel[
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
    """The shipping int4 prefill GEMM with f16 MMA inputs (see module doc)."""
    comptime assert X.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = tid // 32
    var blk_row = Int(block_idx.y) * SG_BM
    var blk_col = Int(block_idx.x) * SG_BN
    var row_base = blk_row + (sg // 2) * _SG_SGM
    var col_base = blk_col + (sg % 2) * _SG_SGN

    var Bs = stack_allocation[
        _Q4_BK * SG_BN, Float16, address_space = AddressSpace.SHARED
    ]()

    var xp = X.ptr
    var pp = P.ptr
    var sp = S.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    var kc = 0
    while kc < K:
        comptime _NW = SG_BN * (_Q4_BK // 8)
        for w in range(tid, _NW, SG_TPB):
            var j_local = w % SG_BN
            var krun = (w // SG_BN) * 8
            var gj = blk_col + j_local
            var gk0 = kc + krun
            if gj < N and gk0 < K:
                var word = pp[(gj * K + gk0) >> 3]
                var scale = sp[gj * NG + (gk0 >> Q4_SHIFT)]
                var nibs = (SIMD[DType.uint32, 8](word) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[
                    DType.float32
                ]() * scale
                comptime for t in range(8):
                    Bs[(krun + t) * SG_BN + j_local] = (
                        qf[t].cast[DType.float16]() if gk0 + t
                        < K else Float16(0.0)
                    )
            else:
                comptime for t in range(8):
                    Bs[(krun + t) * SG_BN + j_local] = Float16(0.0)
        barrier()

        comptime _KS = _Q4_BK // _MMA8
        for kss in range(_KS):
            var kk = kc + kss * _MMA8
            if kk >= K:
                continue
            var ktail = kk + _MMA8 > K
            var afrag = InlineArray[SIMD[DType.float16, _FRAG8], _SG_NTM](
                uninitialized=True
            )
            comptime for mi in range(_SG_NTM):
                var grow = row_base + mi * _MMA8 + frow
                if grow < M and not ktail:
                    afrag[mi] = (
                        (xp + grow * K + kk + fcol)
                        .load[width=_FRAG8]()
                        .cast[DType.float16]()
                    )
                else:
                    var af = SIMD[DType.float16, _FRAG8](0)
                    if grow < M:
                        comptime for s in range(_FRAG8):
                            if kk + fcol + s < K:
                                af[s] = xp[grow * K + kk + fcol + s].cast[
                                    DType.float16
                                ]()
                    afrag[mi] = af
            var bfrag = InlineArray[SIMD[DType.float16, _FRAG8], _SG_NTN](
                uninitialized=True
            )
            comptime for ni in range(_SG_NTN):
                var brow = (
                    (kss * _MMA8 + frow) * SG_BN
                    + (sg % 2) * _SG_SGN
                    + ni * _MMA8
                    + fcol
                )
                bfrag[ni] = (Bs + brow).load[width=_FRAG8]()
            comptime for mi in range(_SG_NTM):
                comptime for ni in range(_SG_NTN):
                    acc[mi * _SG_NTN + ni] = _mma8x8_h(
                        afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                    )
        barrier()
        kc += _Q4_BK

    comptime for mi in range(_SG_NTM):
        comptime for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)



def matmul_simd_q4_bk_kernel[
    LT: TensorLayout, BK: Int
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
    """The shipping f32-MMA int4 GEMM with the K-block size a parameter: BK=32
    (shipping) barriers 64x per K=2048 sweep; BK=64/128 halve/quarter the
    barrier + staging-loop overhead per FLOP at 2x/4x the shared memory
    (BK*64*4B: 8/16/32 KB — all fit)."""
    comptime assert X.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = tid // 32
    var blk_row = Int(block_idx.y) * SG_BM
    var blk_col = Int(block_idx.x) * SG_BN
    var row_base = blk_row + (sg // 2) * _SG_SGM
    var col_base = blk_col + (sg % 2) * _SG_SGN
    var Bs = stack_allocation[
        BK * SG_BN, Float32, address_space = AddressSpace.SHARED
    ]()
    var xp = X.ptr
    var pp = P.ptr
    var sp = S.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )
    var kc = 0
    while kc < K:
        comptime _NW = SG_BN * (BK // 8)
        for w in range(tid, _NW, SG_TPB):
            var j_local = w % SG_BN
            var krun = (w // SG_BN) * 8
            var gj = blk_col + j_local
            var gk0 = kc + krun
            if gj < N and gk0 < K:
                var word = pp[(gj * K + gk0) >> 3]
                var scale = sp[gj * NG + (gk0 >> Q4_SHIFT)]
                var nibs = (SIMD[DType.uint32, 8](word) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[
                    DType.float32
                ]() * scale
                comptime for t in range(8):
                    Bs[(krun + t) * SG_BN + j_local] = (
                        qf[t] if gk0 + t < K else Float32(0.0)
                    )
            else:
                comptime for t in range(8):
                    Bs[(krun + t) * SG_BN + j_local] = Float32(0.0)
        barrier()
        comptime _KS = BK // _MMA8
        for kss in range(_KS):
            var kk = kc + kss * _MMA8
            if kk >= K:
                continue
            var ktail = kk + _MMA8 > K
            var afrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM](
                uninitialized=True
            )
            comptime for mi in range(_SG_NTM):
                var grow = row_base + mi * _MMA8 + frow
                if grow < M and not ktail:
                    afrag[mi] = (xp + grow * K + kk + fcol).load[
                        width=_FRAG8
                    ]()
                else:
                    var af = SIMD[DType.float32, _FRAG8](0)
                    if grow < M:
                        comptime for t in range(_FRAG8):
                            if kk + fcol + t < K:
                                af[t] = xp[grow * K + kk + fcol + t]
                    afrag[mi] = af
            var bfrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTN](
                uninitialized=True
            )
            comptime for ni in range(_SG_NTN):
                var brow = (
                    (kss * _MMA8 + frow) * SG_BN
                    + (sg % 2) * _SG_SGN
                    + ni * _MMA8
                    + fcol
                )
                bfrag[ni] = (Bs + brow).load[width=_FRAG8]()
            comptime for mi in range(_SG_NTM):
                comptime for ni in range(_SG_NTN):
                    acc[mi * _SG_NTN + ni] = _mma8x8(
                        afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                    )
        barrier()
        kc += BK
    comptime for mi in range(_SG_NTM):
        comptime for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            comptime for t in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + t
                if grow < M and gcol < N:
                    var v = frag[t]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


@always_inline
def _mma8x8(
    a: SIMD[DType.float32, _FRAG8],
    b: SIMD[DType.float32, _FRAG8],
    c: SIMD[DType.float32, _FRAG8],
) -> SIMD[DType.float32, _FRAG8]:
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        SIMD[DType.float32, _FRAG8],
    ](a, b, c)


# ── host helpers (same idioms as the other manual gates) ─────────────────────


def prng(mut state: UInt64) -> Float32:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return Float32(Int(state % 2000) - 1000) / 1000.0


def quantize_g128(
    W: List[Float32], N: Int, K: Int
) raises -> Tuple[List[UInt32], List[Float32], List[Float32]]:
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


def check(ctx: DeviceContext, M: Int, K: Int, N: Int) raises:
    """Validate the f16-MMA kernel vs a CPU reference on the dequantized W.
    Tolerance is f16-appropriate: inputs round to 10-bit mantissa before the
    f32-accumulated dot, so rel ~1e-3 is expected and fine (int4's own noise
    is far larger)."""
    var NG = K // Q4_GROUP
    var seed = UInt64(0xBEEF + M * 3 + N * 7 + K)
    var Wh = List[Float32]()
    for _ in range(N * K):
        Wh.append(prng(seed))
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

    comptime k = matmul_simd_q4_f16_kernel[type_of(row_major(1))]
    ctx.enqueue_function[k](
        xt, pt, st, bt, yt, M, K, N, NG, 1,
        grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
        block_dim=SG_TPB,
    )
    ctx.synchronize()

    var md = Float64(0.0)
    var mr = Float64(0.0)
    with yb.map_to_host() as h:
        var t = TileTensor(h, row_major(M * N))
        for m in range(M):
            for n in range(N):
                var acc = Float64(0.0)
                for kk in range(K):
                    acc += Float64(Xh[m * K + kk]) * Float64(deq[n * K + kk])
                var expect = Float32(acc) + Bh[n]
                var d = Float64(
                    rebind[Scalar[DType.float32]](t[m * N + n]) - expect
                )
                if d < 0:
                    d = -d
                if d > md:
                    md = d
                var rf = Float64(expect)
                if rf < 0:
                    rf = -rf
                if rf > mr:
                    mr = rf
    var rel = md / mr if mr > 0 else md
    print(
        "  M=", M, " K=", K, " N=", N, " : max|Δ|=", md, " rel=", rel, " ",
        "OK" if rel < 5.0e-3 else "FAIL", sep="",
    )
    if rel >= 5.0e-3:
        raise Error("f16 GEMM mismatch")


def bench(ctx: DeviceContext, M: Int, K: Int, N: Int) raises:
    var NG = K // Q4_GROUP
    var iters = 30
    var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
    var pb = ctx.enqueue_create_buffer[DType.uint32](N * K // 8)
    var sb = ctx.enqueue_create_buffer[DType.float32](N * NG)
    var bb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
    xb.enqueue_fill(0.25)
    pb.enqueue_fill(0x99999999)
    sb.enqueue_fill(0.01)
    bb.enqueue_fill(0.0)
    var xt = TileTensor(xb, row_major(M * K))
    var pt = TileTensor(pb, row_major(N * K // 8))
    var st = TileTensor(sb, row_major(N * NG))
    var bt = TileTensor(bb, row_major(N))
    var yt = TileTensor(yb, row_major(M * N))
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)

    comptime k32 = matmul_simd_q4_kernel[type_of(row_major(1))]
    for _ in range(3):
        ctx.enqueue_function[k32](
            xt, pt, st, bt, yt, M, K, N, NG, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[k32](
            xt, pt, st, bt, yt, M, K, N, NG, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    ctx.synchronize()
    var f32ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6

    comptime k16 = matmul_simd_q4_f16_kernel[type_of(row_major(1))]
    for _ in range(3):
        ctx.enqueue_function[k16](
            xt, pt, st, bt, yt, M, K, N, NG, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[k16](
            xt, pt, st, bt, yt, M, K, N, NG, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    ctx.synchronize()
    var f16ms = Float64(perf_counter_ns() - t1) / Float64(iters) / 1.0e6
    print(
        "  M=", M, " K=", K, " N=", N,
        "  f32-MMA ", f32ms, " ms (", flops / f32ms / 1.0e9, " TFLOP/s)",
        "  f16-MMA ", f16ms, " ms (", flops / f16ms / 1.0e9, " TFLOP/s)  ",
        f32ms / f16ms, "x", sep="",
    )
    comptime for BK in [64, 128]:
        comptime kbk = matmul_simd_q4_bk_kernel[type_of(row_major(1)), BK]
        for _ in range(3):
            ctx.enqueue_function[kbk](
                xt, pt, st, bt, yt, M, K, N, NG, 0,
                grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(iters):
            ctx.enqueue_function[kbk](
                xt, pt, st, bt, yt, M, K, N, NG, 0,
                grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var bkms = Float64(perf_counter_ns() - t2) / Float64(iters) / 1.0e6
        print(
            "            BK=", BK, "  ", bkms, " ms (",
            flops / bkms / 1.0e9, " TFLOP/s)  ", f32ms / bkms, "x ship",
            sep="",
        )


def main() raises:
    comptime if not has_accelerator():
        print("no GPU — skipping")
        return
    var ctx = DeviceContext()
    print("correctness (f16 inputs, f32 accumulate; rel tol 5e-3):")
    check(ctx, 71, 256, 129)
    check(ctx, 64, 384, 192)
    check(ctx, 130, 2048, 257)
    print("bench (3B prefill shapes; TFLOP/s, 30 iters):")
    bench(ctx, 512, 2048, 2560)     # qkv @ mid prompt
    bench(ctx, 1570, 2048, 2560)    # qkv @ long-code prompt
    bench(ctx, 1570, 2048, 22016)   # merged gate_up (the big one)
    bench(ctx, 1570, 11008, 2048)   # down
