"""Speed half of the int8 eval: a per-channel int8 dequant GEMV vs the shipped
bf16 matmul_kernel, at the real Qwen-0.5B decode (M=1) shapes. int8 halves the
weight bytes (1 vs 2 per element), so a bandwidth-bound GEMV should approach 2x.
Compares three: bf16 (shipped), naive int8 (1-byte loads), vectorized int8x4
(4 packed in an int32 → 128-byte coalesced transactions). Needs a Metal GPU; no
weights required (synthetic buffers). Self-contained — does not touch src.

    pixi run int8-gemv-bench
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import global_idx, WARP_SIZE
from std.gpu.primitives.warp import sum as warp_sum
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major
from kernels import matmul_kernel

comptime BLOCK = 256


def int8_gemv_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.int8, LT, MutAnyOrigin],  # per-channel int8 weights
    S: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # per-row (output) scale [N]
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
):
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var acc = Float32(0.0)
    for k in range(lane, K, WARP_SIZE):
        var xv = rebind[Scalar[DType.float32]](X[m * K + k])
        var wv = rebind[Scalar[DType.int8]](W[n * K + k]).cast[DType.float32]()
        acc += xv * wv
    var total = warp_sum(acc)
    if lane == 0:
        Y[m * N + n] = rebind[Y.ElementType](
            total * rebind[Scalar[DType.float32]](S[n])
        )


def int8x4_gemv_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[
        DType.int32, LT, MutAnyOrigin
    ],  # 4 packed int8 per int32 word
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
):
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var Kp = K // 4  # int32 words per row
    var acc = Float32(0.0)
    for kp in range(lane, Kp, WARP_SIZE):
        var packed = rebind[Scalar[DType.int32]](W[n * Kp + kp])
        var quad = UnsafePointer(to=packed).bitcast[SIMD[DType.int8, 4]]()[0]
        var k0 = kp * 4
        acc += (
            rebind[Scalar[DType.float32]](X[m * K + k0 + 0])
            * quad[0].cast[DType.float32]()
        )
        acc += (
            rebind[Scalar[DType.float32]](X[m * K + k0 + 1])
            * quad[1].cast[DType.float32]()
        )
        acc += (
            rebind[Scalar[DType.float32]](X[m * K + k0 + 2])
            * quad[2].cast[DType.float32]()
        )
        acc += (
            rebind[Scalar[DType.float32]](X[m * K + k0 + 3])
            * quad[3].cast[DType.float32]()
        )
    var total = warp_sum(acc)
    if lane == 0:
        Y[m * N + n] = rebind[Y.ElementType](
            total * rebind[Scalar[DType.float32]](S[n])
        )


def bench_int8x4(ctx: DeviceContext, name: String, K: Int, N: Int) raises:
    var M = 1
    var Kp = K // 4
    var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
    var wb = ctx.enqueue_create_buffer[DType.int32](N * Kp)
    var sb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
    xb.enqueue_fill(0.5)
    wb.enqueue_fill(0x01010101)
    sb.enqueue_fill(0.01)
    yb.enqueue_fill(0.0)
    var xt = TileTensor(xb, row_major(M * K))
    var wt = TileTensor(wb, row_major(N * Kp))
    var st = TileTensor(sb, row_major(N))
    var yt = TileTensor(yb, row_major(M * N))
    comptime k = int8x4_gemv_kernel[type_of(row_major(1))]
    var grid = ceildiv(M * N * WARP_SIZE, BLOCK)
    var iters = 200
    for _ in range(5):
        ctx.enqueue_function[k](
            xt, wt, st, yt, M, K, N, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[k](
            xt, wt, st, yt, M, K, N, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
    var gbps = Float64(N * K * 1 + M * K * 4 + N * 4) / (ms * 1.0e6)
    print(
        "  i8x4 ", name, " K=", K, " N=", N, " : ", ms, " ms  ", gbps, " GB/s"
    )


def bench_bf16(ctx: DeviceContext, name: String, K: Int, N: Int) raises:
    var M = 1
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
    comptime k = matmul_kernel[type_of(row_major(1))]
    var grid = ceildiv(M * N * WARP_SIZE, BLOCK)
    var iters = 200
    for _ in range(5):
        ctx.enqueue_function[k](
            xt, wt, bt, yt, M, K, N, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[k](
            xt, wt, bt, yt, M, K, N, 0, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
    var gbps = Float64(N * K * 2 + M * K * 4) / (ms * 1.0e6)
    print(
        "  bf16 ", name, " K=", K, " N=", N, " : ", ms, " ms  ", gbps, " GB/s"
    )


def bench_int8(
    ctx: DeviceContext, name: String, K: Int, N: Int
) raises -> Float64:
    var M = 1
    var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
    var wb = ctx.enqueue_create_buffer[DType.int8](N * K)
    var sb = ctx.enqueue_create_buffer[DType.float32](N)
    var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
    xb.enqueue_fill(0.5)
    wb.enqueue_fill(1)
    sb.enqueue_fill(0.01)
    yb.enqueue_fill(0.0)
    var xt = TileTensor(xb, row_major(M * K))
    var wt = TileTensor(wb, row_major(N * K))
    var st = TileTensor(sb, row_major(N))
    var yt = TileTensor(yb, row_major(M * N))
    comptime k = int8_gemv_kernel[type_of(row_major(1))]
    var grid = ceildiv(M * N * WARP_SIZE, BLOCK)
    var iters = 200
    for _ in range(5):
        ctx.enqueue_function[k](
            xt, wt, st, yt, M, K, N, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function[k](
            xt, wt, st, yt, M, K, N, grid_dim=grid, block_dim=BLOCK
        )
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / Float64(iters) / 1.0e6
    var gbps = Float64(N * K * 1 + M * K * 4 + N * 4) / (ms * 1.0e6)
    print(
        "  int8 ", name, " K=", K, " N=", N, " : ", ms, " ms  ", gbps, " GB/s"
    )
    return ms


def both(ctx: DeviceContext, name: String, K: Int, N: Int) raises:
    bench_bf16(ctx, name, K, N)
    _ = bench_int8(ctx, name, K, N)
    bench_int8x4(ctx, name, K, N)


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU")
    var ctx = DeviceContext()
    print("=== decode GEMV: bf16 vs per-channel int8 (M=1) ===")
    both(ctx, "qkv ", 896, 1152)
    both(ctx, "o   ", 896, 896)
    both(ctx, "gate", 896, 4864)
    both(ctx, "up  ", 896, 4864)
    both(ctx, "down", 4864, 896)
    both(ctx, "lmhd", 896, 151936)
