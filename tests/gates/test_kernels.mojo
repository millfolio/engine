"""Gate: building-block kernels vs NumPy (ARCHITECTURE.md §6 Phase 2, §11 #4).

Exercises the library op launchers (matmul+bias, RMSNorm, SwiGLU = matmul →
silu·mul → matmul) on synthetic + real fixtures (`pixi run kernels-capture`).
`pixi run test-kernels`.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.os.path import exists

from model import mm, rmsnorm, silu_mul
from testio import read_text, read_f32, upload_f32, upload_bf16, max_abs

comptime TOL = Float32(3.0e-3)
comptime DevBuf = DeviceBuffer[DType.float32]


def read_ints(path: String) raises -> List[Int]:
    var out = List[Int]()
    with open(path, "r") as f:
        for part in f.read().split(" "):
            var q = String(part).strip()
            if q.byte_length() > 0:
                out.append(Int(atol(q)))
    return out^


def report(name: String, m: Float32) -> Bool:
    var ok = m < TOL
    print("  ", name, " max_abs=", m, " [", "OK" if ok else "FAIL", "]", sep="")
    return ok


def run_rmsnorm(ctx: DeviceContext, dir: String) raises -> Bool:
    var meta = read_ints(dir + "/meta.txt")
    var T = meta[0]
    var dim = meta[1]
    var x = upload_f32(ctx, read_f32(dir + "/x.bin"))
    var w = upload_f32(ctx, read_f32(dir + "/w.bin"))
    var y = rmsnorm(ctx, x, w, T, dim)
    ctx.synchronize()
    return report(dir, max_abs(y, read_f32(dir + "/expected.bin")))


def run_matmul(ctx: DeviceContext, dir: String) raises -> Bool:
    var meta = read_ints(dir + "/meta.txt")
    var M = meta[0]
    var K = meta[1]
    var N = meta[2]
    var use_bias = meta[3]
    var x = upload_f32(ctx, read_f32(dir + "/x.bin"))
    var w = upload_bf16(ctx, read_f32(dir + "/W.bin"))   # weights are bf16 on device
    var b = upload_f32(ctx, read_f32(dir + "/b.bin"))
    var y = mm(ctx, x, w, b, M, K, N, use_bias)
    ctx.synchronize()
    return report(dir, max_abs(y, read_f32(dir + "/expected.bin")))


def run_swiglu(ctx: DeviceContext, dir: String) raises -> Bool:
    var meta = read_ints(dir + "/meta.txt")
    var T = meta[0]
    var dim = meta[1]
    var I = meta[2]
    var x = upload_f32(ctx, read_f32(dir + "/x.bin"))
    var wg = upload_bf16(ctx, read_f32(dir + "/w_gate.bin"))
    var wu = upload_bf16(ctx, read_f32(dir + "/w_up.bin"))
    var wd = upload_bf16(ctx, read_f32(dir + "/w_down.bin"))
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)
    var g = mm(ctx, x, wg, dummy, T, dim, I, 0)
    var u = mm(ctx, x, wu, dummy, T, dim, I, 0)
    var gu = silu_mul(ctx, g, u, T * I)
    var out = mm(ctx, gu, wd, dummy, T, I, dim, 0)
    ctx.synchronize()
    return report(dir, max_abs(out, read_f32(dir + "/expected.bin")))


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")
    var ctx = DeviceContext()
    var root = "tests/fixtures/kernels/"
    print("kernels gate — GPU vs NumPy (tol", TOL, "):")

    var all_ok = True
    var names = [
        String("syn_rmsnorm"), String("syn_matmul"), String("syn_swiglu"),
        String("real_rmsnorm"), String("real_matmul"), String("real_swiglu"),
    ]
    for name in names:
        var dir = root + name
        if not exists(dir + "/meta.txt"):
            print("  ", dir, " [skipped — run kernels-capture]", sep="")
            continue
        var ok = (
            run_rmsnorm(ctx, dir) if name.endswith("rmsnorm")
            else run_matmul(ctx, dir) if name.endswith("matmul")
            else run_swiglu(ctx, dir)
        )
        all_ok = all_ok and ok

    if not all_ok:
        raise Error("a kernel does NOT match the reference — gate FAILED")
    print("OK — matmul / RMSNorm / SwiGLU match the reference")
