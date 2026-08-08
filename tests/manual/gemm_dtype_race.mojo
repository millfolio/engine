"""Prefill-GEMM operand-dtype race: f32 vs bf16 vs f16 on this machine's GPU.

Why: the dequant-once prefill pipeline runs its GEMM with bf16 MMA operands.
On the M4, f16 and bf16 measure identical (3.18 TFLOP/s) and both beat f32
(~2.75). On pre-M3 GPUs bf16 `simdgroup_matrix` support is the reason the
v64-widening workaround exists (M1/M2 misread the v2bf16 form) — and the M2
Pro bench showed our long prefill got SLOWER there while MLX's got faster,
consistent with bf16 MMA being emulated/half-rate on M2-generation silicon
while f16 is native since M1. This race answers that per machine: if f16 ≫
bf16 here, the dequant scratch should switch to f16 on this GPU family.

Method: one direct-load 64×64/4-simdgroup GEMM (the pipeline kernel's
structure: no staging, K-step 8, f32 accumulate) instantiated per operand
dtype, on the Qwen2.5-3B prefill shapes at M=1570 and M=512. f16/bf16 outputs
are sanity-checked against f32 via relative RMS (expect ~2e-3 = operand
rounding; O(1) means a broken lowering). GPU only, no weights, ~110 MB peak.

    pixi run gemm-dtype-race
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx
from std.sys import llvm_intrinsic
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from kernels import _frag8_layout, _mma8x8, SG_BM, SG_BN, SG_TPB

comptime _MMA8 = 8
comptime _FRAG8 = 2
comptime _SGM = 32
comptime _SGN = 32
comptime _NTM = 4
comptime _NTN = 4


@always_inline
def _mma8x8_w[
    T: DType
](
    a: SIMD[T, _FRAG8],
    b: SIMD[T, _FRAG8],
    c: SIMD[DType.float32, _FRAG8],
) -> SIMD[DType.float32, _FRAG8]:
    """8×8×8 MMA, f32 accumulate. f32 operands use the compact v2 form (the
    shipping `_mma8x8`); f16/bf16 use the v64-widened form every Apple GPU
    handles (M1/M2 misread the compact v2bf16 encoding)."""
    comptime if T == DType.float32:
        return _mma8x8(
            rebind[SIMD[DType.float32, _FRAG8]](a),
            rebind[SIMD[DType.float32, _FRAG8]](b),
            c,
        )
    else:
        var a_wide = SIMD[T, 64](0)
        var b_wide = SIMD[T, 64](0)
        var c_wide = SIMD[DType.float32, 64](0)
        comptime for s in range(_FRAG8):
            a_wide[s] = a[s]
            b_wide[s] = b[s]
            c_wide[s] = c[s]
        var d_wide = llvm_intrinsic[
            "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
            SIMD[DType.float32, 64],
        ](a_wide, b_wide, c_wide)
        var d = SIMD[DType.float32, _FRAG8](0)
        comptime for s in range(_FRAG8):
            d[s] = d_wide[s]
        return d


def gemm_dt[
    LT: TensorLayout, T: DType
](
    X: TileTensor[T, LT, MutAnyOrigin],
    W: TileTensor[T, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
):
    """Y[M,N] = X[M,K]·W[K,N], operand dtype T, f32 accumulate. Direct-load
    64×64 block / 2×2 simdgroups / K-step 8 (the pipeline GEMM's structure).
    Requires K % 8 == 0."""
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
    var interior = (row_base + _SGM <= M) and (col_base + _SGN <= N)

    var xp = X.ptr
    var wp = W.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _NTM * _NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )
    var nkt = K // _MMA8
    for ks in range(nkt):
        var kk = ks * _MMA8
        var afrag = InlineArray[SIMD[T, _FRAG8], _NTM](uninitialized=True)
        comptime for mi in range(_NTM):
            var grow = row_base + mi * _MMA8 + frow
            if interior or grow < M:
                afrag[mi] = xp.unsafe_offset(grow * K + kk + fcol).unsafe_load[
                    width=_FRAG8
                ]()
            else:
                afrag[mi] = SIMD[T, _FRAG8](0)
        var bfrag = InlineArray[SIMD[T, _FRAG8], _NTN](uninitialized=True)
        comptime for ni in range(_NTN):
            var krow = kk + frow
            var gj = col_base + ni * _MMA8 + fcol
            if interior or gj + 1 < N:
                bfrag[ni] = wp.unsafe_offset(krow * N + gj).unsafe_load[
                    width=_FRAG8
                ]()
            else:
                var bf = SIMD[T, _FRAG8](0)
                if gj < N:
                    bf[0] = wp[unsafe_offset=krow * N + gj]
                bfrag[ni] = bf
        comptime for mi in range(_NTM):
            comptime for ni in range(_NTN):
                acc[mi * _NTN + ni] = _mma8x8_w[T](
                    afrag[mi], bfrag[ni], acc[mi * _NTN + ni]
                )
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


def bench_dtype[
    T: DType
](
    ctx: DeviceContext,
    name: String,
    mut yb: DeviceBuffer[DType.float32],
    M: Int,
    N: Int,
    K: Int,
) raises:
    var xb = ctx.enqueue_create_buffer[T](M * K)
    var wb = ctx.enqueue_create_buffer[T](K * N)
    var st = UInt64(0xD7A3E)
    with xb.map_to_host() as h:
        var t = TileTensor(h, row_major(M * K))
        for i in range(M * K):
            t[i] = rebind[t.ElementType](
                (Float32(Int(lcg_next(st) % 401) - 200) / Float32(100.0)).cast[
                    T
                ]()
            )
    with wb.map_to_host() as h:
        var t = TileTensor(h, row_major(K * N))
        for i in range(K * N):
            t[i] = rebind[t.ElementType](
                (Float32(Int(lcg_next(st) % 401) - 200) / Float32(1000.0)).cast[
                    T
                ]()
            )
    var xt = TileTensor(xb, row_major(M * K))
    var wt = TileTensor(wb, row_major(K * N))
    var yt = TileTensor(yb, row_major(M * N))
    comptime kv = gemm_dt[type_of(row_major(1)), T]
    var grid = (ceildiv(N, SG_BN), ceildiv(M, SG_BM))
    var iters = Int(6.0e11 / (2.0 * Float64(M) * Float64(N) * Float64(K)))
    if iters < 5:
        iters = 5
    if iters > 60:
        iters = 60
    for _ in range(3):
        ctx.enqueue_function[kv](
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
        ctx.enqueue_function[kv](
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
    var tf = 2.0 * Float64(M) * Float64(N) * Float64(K) / (ms * 1.0e-3) / 1.0e12
    print("    ", name, ": ", ms, " ms  ", tf, " TFLOP/s", sep="")


def run_shape(ctx: DeviceContext, label: String, M: Int, N: Int, K: Int) raises:
    print("  ", label, " M=", M, " N=", N, " K=", K, sep="")
    var y32 = ctx.enqueue_create_buffer[DType.float32](M * N)
    var yh = ctx.enqueue_create_buffer[DType.float32](M * N)
    bench_dtype[DType.float32](ctx, "f32 ", y32, M, N, K)
    bench_dtype[DType.bfloat16](ctx, "bf16", yh, M, N, K)
    var rb = rel_rms(yh, y32, M * N)
    bench_dtype[DType.float16](ctx, "f16 ", yh, M, N, K)
    var rh = rel_rms(yh, y32, M * N)
    print(
        "    sanity vs f32: bf16 rel_rms=",
        rb,
        "  f16 rel_rms=",
        rh,
        "  (expect ~2e-3; O(1) = broken lowering)",
    )


def main() raises:
    var ctx = DeviceContext()
    print("prefill-GEMM operand dtype race (f32 / bf16 / f16), f32 accumulate")
    run_shape(ctx, "gate_up", 1570, 22016, 2048)
    run_shape(ctx, "down   ", 1570, 2048, 11008)
    run_shape(ctx, "qkv    ", 1570, 2560, 2048)
    run_shape(ctx, "gate_up", 512, 22016, 2048)
