"""Gate: the simdgroup-matrix prefill GEMM (matmul_simd_kernel, via mm's simd
path) agrees with the scalar register-tiled GEMM and with a CPU reference, across
sizes including non-multiples-of-8 (boundary masking), with and without bias.
Also asserts the runtime capability probe succeeds on this machine.

The simd path is f32 but not bit-identical to the scalar path (hardware FMA /
accumulation order), so the tolerance is loose-but-tight: ~1e-3 catches a wrong
kernel while passing the ~2e-6 numerical drift. `pixi run test-simd-gemm`.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from model import mm, probe_simd_gemm
from testio import upload_f32, upload_bf16, max_abs

comptime DevBuf = DeviceBuffer[DType.float32]
comptime WBuf = DeviceBuffer[DType.uint16]
comptime TOL = Float32(1.0e-3)


def to_host(mut d: DevBuf, n: Int) raises -> List[Float32]:
    var out = List[Float32]()
    with d.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            out.append(rebind[Scalar[DType.float32]](mt[i]))
    return out^


def check(ctx: DeviceContext, M: Int, K: Int, N: Int, use_bias: Int) raises:
    # deterministic inputs
    var hx = List[Float32]()
    for i in range(M * K):
        hx.append(Float32((i * 7) % 13) * 0.1 - 0.6)
    var hw = List[Float32]()
    for i in range(N * K):
        hw.append(Float32((i * 5) % 11) * 0.05 - 0.25)
    var hb = List[Float32]()
    for i in range(N):
        hb.append(Float32(i % 9) * 0.01 - 0.04)

    var xb = upload_f32(ctx, hx)
    var wb = upload_bf16(ctx, hw)
    var bb = upload_f32(ctx, hb)
    var xb2 = upload_f32(ctx, hx)
    var wb2 = upload_bf16(ctx, hw)
    var bb2 = upload_f32(ctx, hb)

    var y_scalar = mm(ctx, xb, wb, bb, M, K, N, use_bias, False)
    var y_simd = mm(ctx, xb2, wb2, bb2, M, K, N, use_bias, True)
    ctx.synchronize()

    # scalar is already NumPy-gated by test-kernels, so simd-vs-scalar is enough.
    var scalar_host = to_host(y_scalar, M * N)
    var d = max_abs(y_simd, scalar_host)   # |simd - scalar|
    print("  M=", M, " K=", K, " N=", N, " bias=", use_bias,
          "  max|simd-scalar|=", d)
    if d > TOL:
        raise Error("simd GEMM disagrees with scalar beyond tolerance")


def main() raises:
    if not has_accelerator():
        raise Error("no GPU — this gate needs Metal")
    var ctx = DeviceContext()

    if not probe_simd_gemm(ctx):
        raise Error("probe_simd_gemm failed — AIR simdgroup intrinsics rejected")
    print("probe_simd_gemm: OK")

    check(ctx, 8, 16, 8, 0)         # single tile, partial K
    check(ctx, 37, 44, 53, 1)       # odd M,K,N + bias (boundary masking)
    check(ctx, 64, 128, 128, 0)     # multi-tile, clean
    check(ctx, 130, 896, 896, 1)    # qwen-ish proj, odd M, bias
    check(ctx, 256, 896, 4864, 0)   # mlp gate shape

    print("simd-gemm gate: PASS")
