"""Staged-loads race: the MLX steel pattern vs our direct-load GEMM.

The last untested structural element of MLX's prefill GEMM: cooperatively
copy A (BMxBK) and B (BKxBN) tiles into threadgroup memory with wide
vectorized loads, barrier, then feed the MMA through the HARDWARE tile
load from threadgroup space (air.simdgroup_matrix_8x8_load.v64f16.p3f16 —
decoded 2026-08-09, layout bit-exact vs _frag8_layout). Hypothesis: the
hw load's cost is only worth paying from shared memory, where it replaces
per-lane address math with single instructions and the staged copy fixes
DRAM access patterns.

Interior-only (M=1536), f16 operands, f32 accumulate. Baseline = the
direct-load manual kernel (current shipped structure, 3.45 TFLOP/s on M4).
Variants: staged with BK=16 and BK=32. Math order per output is identical
across variants -> parity must be rel_rms == 0.

    pixi run gemm-staged-race
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.sys import llvm_intrinsic
from std.ffi import external_call
from std.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import stack_allocation
from layout import TileTensor, TensorLayout, row_major
from kernels import _frag8_layout, SG_BM, SG_BN, SG_TPB

comptime _MMA8 = 8
comptime _FRAG8 = 2
comptime _SGM = 32
comptime _SGN = 32
comptime _NTM = 4
comptime _NTN = 4


@always_inline
def _mma_v64(
    a: SIMD[DType.float16, 64],
    b: SIMD[DType.float16, 64],
    c: SIMD[DType.float32, 64],
) -> SIMD[DType.float32, 64]:
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, 64],
    ](a, b, c)


def _p3_probe_kernel[
    LT: TensorLayout
](Out: TileTensor[DType.float16, LT, MutAnyOrigin]):
    """Minimal kernel forcing pipeline creation with a p3 hw tile load."""
    comptime assert Out.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var Ts = stack_allocation[64, Float16, address_space=AddressSpace.SHARED]()
    if lane < 32:
        Ts[unsafe_offset=lane] = Float16(lane)
        Ts[unsafe_offset=lane + 32] = Float16(lane + 32)
    barrier()
    var v = external_call[
        "air.simdgroup_matrix_8x8_load.v64f16.p3f16",
        SIMD[DType.float16, 64],
    ](
        Ts,
        SIMD[DType.int64, 2](8, 8),
        SIMD[DType.int64, 2](1, 8),
        SIMD[DType.int64, 2](0, 0),
    )
    Out[lane * 2] = rebind[Out.ElementType](v[0])
    Out[lane * 2 + 1] = rebind[Out.ElementType](v[1])


def p3_supported(ctx: DeviceContext) -> Bool:
    """True if this GPU's backend can compile the p3 hardware tile load.
    The M2-generation Metal backend CRASHES (XPC_ERROR_CONNECTION_
    INTERRUPTED at pipeline creation) on it — surfaces as a catchable
    error, same pattern as probe_simd_gemm."""
    try:
        var ob = ctx.enqueue_create_buffer[DType.float16](64)
        ob.enqueue_fill(0.0)
        var ot = TileTensor(ob, row_major(64))
        comptime k = _p3_probe_kernel[type_of(row_major(1))]
        ctx.enqueue_function[k](ot, grid_dim=1, block_dim=32)
        ctx.synchronize()
        return True
    except:
        return False


def gemm_staged_kernel[
    LT: TensorLayout, BK: Int
](
    X: TileTensor[DType.float16, LT, MutAnyOrigin],
    W: TileTensor[DType.float16, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
):
    """Steel-pattern GEMM: staged A+B tiles, p3 hardware fragment loads."""
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    comptime assert X.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = tid // 32
    var blk_row = Int(block_idx.y) * SG_BM
    var blk_col = Int(block_idx.x) * SG_BN
    var sg_row = (sg // 2) * _SGM  # this simdgroup's row base IN the tile
    var sg_col = (sg % 2) * _SGN

    var As = stack_allocation[
        SG_BM * BK, Float16, address_space=AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * SG_BN, Float16, address_space=AddressSpace.SHARED
    ]()

    var xp = X.ptr
    var wp = W.ptr
    var acc = InlineArray[SIMD[DType.float32, 64], _NTM * _NTN](
        fill=SIMD[DType.float32, 64](0)
    )

    comptime A_CHUNKS = SG_BM * BK // 8  # 8-wide copy chunks
    comptime B_CHUNKS = BK * SG_BN // 8
    comptime KPB = BK // _MMA8  # MMA K-steps per staged block

    var nkb = K // BK
    for kb in range(nkb):
        var k0 = kb * BK
        # cooperative copy: A tile [SG_BM x BK] row-major (stride BK)
        comptime for it in range((A_CHUNKS + SG_TPB - 1) // SG_TPB):
            var c = tid + it * SG_TPB
            if c < A_CHUNKS:
                var row = c // (BK // 8)
                var col = (c % (BK // 8)) * 8
                var v = xp.unsafe_offset(
                    (blk_row + row) * K + k0 + col
                ).unsafe_load[width=8]()
                As.unsafe_offset(row * BK + col).unsafe_store(v)
        # B tile [BK x SG_BN] row-major (stride SG_BN)
        comptime for it in range((B_CHUNKS + SG_TPB - 1) // SG_TPB):
            var c = tid + it * SG_TPB
            if c < B_CHUNKS:
                var krow = c // (SG_BN // 8)
                var col = (c % (SG_BN // 8)) * 8
                var v = wp.unsafe_offset(
                    (k0 + krow) * N + blk_col + col
                ).unsafe_load[width=8]()
                Bs.unsafe_offset(krow * SG_BN + col).unsafe_store(v)
        barrier()

        comptime for ks in range(KPB):
            var afrag = InlineArray[SIMD[DType.float16, 64], _NTM](
                uninitialized=True
            )
            comptime for mi in range(_NTM):
                afrag[mi] = external_call[
                    "air.simdgroup_matrix_8x8_load.v64f16.p3f16",
                    SIMD[DType.float16, 64],
                ](
                    As.unsafe_offset((sg_row + mi * _MMA8) * BK + ks * _MMA8),
                    SIMD[DType.int64, 2](Int64(BK), 8),
                    SIMD[DType.int64, 2](1, Int64(BK)),
                    SIMD[DType.int64, 2](0, 0),
                )
            var bfrag = InlineArray[SIMD[DType.float16, 64], _NTN](
                uninitialized=True
            )
            comptime for ni in range(_NTN):
                bfrag[ni] = external_call[
                    "air.simdgroup_matrix_8x8_load.v64f16.p3f16",
                    SIMD[DType.float16, 64],
                ](
                    Bs.unsafe_offset(ks * _MMA8 * SG_BN + sg_col + ni * _MMA8),
                    SIMD[DType.int64, 2](Int64(SG_BN), 8),
                    SIMD[DType.int64, 2](1, Int64(SG_BN)),
                    SIMD[DType.int64, 2](0, 0),
                )
            comptime for mi in range(_NTM):
                comptime for ni in range(_NTN):
                    acc[mi * _NTN + ni] = _mma_v64(
                        afrag[mi], bfrag[ni], acc[mi * _NTN + ni]
                    )
        barrier()

    comptime for mi in range(_NTM):
        comptime for ni in range(_NTN):
            var frag = acc[mi * _NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = blk_row + sg_row + mi * _MMA8 + frow
                var gcol = blk_col + sg_col + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    Y[grow * N + gcol] = rebind[Y.ElementType](frag[s])


def gemm_manual_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float16, LT, MutAnyOrigin],
    W: TileTensor[DType.float16, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
):
    """Direct-load baseline (shipped structure, interior-only)."""
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    comptime assert X.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(thread_idx.x) // 32
    var row_base = Int(block_idx.y) * SG_BM + (sg // 2) * _SGM
    var col_base = Int(block_idx.x) * SG_BN + (sg % 2) * _SGN

    var xp = X.ptr
    var wp = W.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _NTM * _NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )
    var nkt = K // _MMA8
    for ks in range(nkt):
        var kk = ks * _MMA8
        var afrag = InlineArray[SIMD[DType.float16, _FRAG8], _NTM](
            uninitialized=True
        )
        comptime for mi in range(_NTM):
            var grow = row_base + mi * _MMA8 + frow
            afrag[mi] = xp.unsafe_offset(grow * K + kk + fcol).unsafe_load[
                width=_FRAG8
            ]()
        var bfrag = InlineArray[SIMD[DType.float16, _FRAG8], _NTN](
            uninitialized=True
        )
        comptime for ni in range(_NTN):
            var gj = col_base + ni * _MMA8 + fcol
            bfrag[ni] = wp.unsafe_offset((kk + frow) * N + gj).unsafe_load[
                width=_FRAG8
            ]()
        comptime for mi in range(_NTM):
            comptime for ni in range(_NTN):
                var a_wide = SIMD[DType.float16, 64](0)
                var b_wide = SIMD[DType.float16, 64](0)
                var c_wide = SIMD[DType.float32, 64](0)
                comptime for s in range(_FRAG8):
                    a_wide[s] = afrag[mi][s]
                    b_wide[s] = bfrag[ni][s]
                    c_wide[s] = acc[mi * _NTN + ni][s]
                var d = _mma_v64(a_wide, b_wide, c_wide)
                comptime for s in range(_FRAG8):
                    acc[mi * _NTN + ni][s] = d[s]
    comptime for mi in range(_NTM):
        comptime for ni in range(_NTN):
            var frag = acc[mi * _NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    Y[grow * N + gcol] = rebind[Y.ElementType](frag[s])


def lcg_next(mut state: UInt64) -> UInt64:
    state = state * 6364136223846793005 + 1442695040888963407
    return state >> 33


def rel_rms(
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float64:
    var sd = Float64(0.0)
    var sr = Float64(0.0)
    with a.map_to_host() as ha:
        with b.map_to_host() as hb:
            var ta = TileTensor(ha, row_major(n))
            var tb = TileTensor(hb, row_major(n))
            for i in range(n):
                var x = Float64(rebind[Float32](ta[i]))
                var y = Float64(rebind[Float32](tb[i]))
                sd += (x - y) * (x - y)
                sr += x * x
    return (sd / sr) ** 0.5


def bench[
    STAGED: Bool, BK: Int
](
    ctx: DeviceContext,
    name: String,
    mut xb: DeviceBuffer[DType.float16],
    mut wb: DeviceBuffer[DType.float16],
    mut yb: DeviceBuffer[DType.float32],
    M: Int,
    N: Int,
    K: Int,
) raises:
    var xt = TileTensor(xb, row_major(M * K))
    var wt = TileTensor(wb, row_major(K * N))
    var yt = TileTensor(yb, row_major(M * N))
    var grid = (ceildiv(N, SG_BN), ceildiv(M, SG_BM))
    var iters = Int(6.0e11 / (2.0 * Float64(M) * Float64(N) * Float64(K)))
    if iters < 5:
        iters = 5
    if iters > 60:
        iters = 60
    comptime if STAGED:
        comptime k = gemm_staged_kernel[type_of(row_major(1)), BK]
        for _ in range(3):
            ctx.enqueue_function[k](
                xt,
                wt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid,
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            ctx.enqueue_function[k](
                xt,
                wt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid,
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
        var tf = (
            2.0 * Float64(M) * Float64(N) * Float64(K) / (ms * 1.0e-3) / 1.0e12
        )
        print("    ", name, ": ", ms, " ms  ", tf, " TFLOP/s", sep="")
    else:
        comptime k = gemm_manual_kernel[type_of(row_major(1))]
        for _ in range(3):
            ctx.enqueue_function[k](
                xt,
                wt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid,
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            ctx.enqueue_function[k](
                xt,
                wt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid,
                block_dim=SG_TPB,
            )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
        var tf = (
            2.0 * Float64(M) * Float64(N) * Float64(K) / (ms * 1.0e-3) / 1.0e12
        )
        print("    ", name, ": ", ms, " ms  ", tf, " TFLOP/s", sep="")


def run_shape(
    ctx: DeviceContext,
    label: String,
    M: Int,
    N: Int,
    K: Int,
    staged_ok: Bool,
) raises:
    print("  ", label, " M=", M, " N=", N, " K=", K, sep="")
    var xb = ctx.enqueue_create_buffer[DType.float16](M * K)
    var wb = ctx.enqueue_create_buffer[DType.float16](K * N)
    var y0 = ctx.enqueue_create_buffer[DType.float32](M * N)
    var y1 = ctx.enqueue_create_buffer[DType.float32](M * N)
    var st = UInt64(0x57EE1)
    with xb.map_to_host() as h:
        var t = TileTensor(h, row_major(M * K))
        for i in range(M * K):
            t[i] = rebind[t.ElementType](
                (Float32(Int(lcg_next(st) % 401) - 200) / Float32(100.0)).cast[
                    DType.float16
                ]()
            )
    with wb.map_to_host() as h:
        var t = TileTensor(h, row_major(K * N))
        for i in range(K * N):
            t[i] = rebind[t.ElementType](
                (Float32(Int(lcg_next(st) % 401) - 200) / Float32(1000.0)).cast[
                    DType.float16
                ]()
            )
    bench[False, 0](ctx, "direct (ship) ", xb, wb, y0, M, N, K)
    if not staged_ok:
        print(
            "    staged: SKIPPED — Metal backend crashed compiling the p3"
            " tile load for this GPU (M2-generation backend bug)"
        )
        return
    bench[True, 16](ctx, "staged BK=16  ", xb, wb, y1, M, N, K)
    var r16 = rel_rms(y1, y0, M * N)
    bench[True, 32](ctx, "staged BK=32  ", xb, wb, y1, M, N, K)
    var r32 = rel_rms(y1, y0, M * N)
    print("    parity: BK16=", r16, " BK32=", r32, " (expect 0.0)")


def main() raises:
    var ctx = DeviceContext()
    print("staged (steel-pattern, p3 hw loads) vs direct loads — f16 GEMM")
    var staged_ok = p3_supported(ctx)
    if not staged_ok:
        print(
            "  p3 hardware tile load: BACKEND COMPILE CRASH on this GPU —"
            " staged variants will be skipped (direct baseline still runs)"
        )
    run_shape(ctx, "gate_up", 1536, 22016, 2048, staged_ok)
    run_shape(ctx, "down   ", 1536, 2048, 11008, staged_ok)
    run_shape(ctx, "qkv    ", 1536, 2560, 2048, staged_ok)
