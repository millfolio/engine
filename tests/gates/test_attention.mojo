"""Gate: RoPE + causal GQA attention vs NumPy (ARCHITECTURE.md §6 Phase 1, §11 #1).

Feeds projected Q/K/V fixtures (`pixi run attn-capture`) through the library's
cached-attention launcher (prefill = q_offset 0, the K/V are the cache) and checks
the output matches the NumPy reference. `pixi run test-attention`.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from model import attn_cached, rope_k
from testio import read_f32, upload_f32, max_abs

comptime HQ = 14
comptime HKV = 2
comptime HEAD_DIM = 64
comptime TOL = Float32(2.0e-3)


def run(ctx: DeviceContext, dir: String) raises -> Bool:
    var q = read_f32(dir + "/q.bin")
    var T = len(q) // (HQ * HEAD_DIM)
    var qd = upload_f32(ctx, q)
    var kraw = upload_f32(ctx, read_f32(dir + "/k.bin"))
    var vd = upload_f32(ctx, read_f32(dir + "/v.bin"))
    var cache_len = T * HKV * HEAD_DIM
    # K is now RoPE-rotated on write into the cache (§11 #12); rotate then attend.
    var kc = ctx.enqueue_create_buffer[DType.float32](cache_len)
    # arch 0 (Qwen2.5-0.5B) has no QK-norm; pass a size-1 dummy weight (ignored).
    var dummy = ctx.enqueue_create_buffer[DType.float32](1)
    rope_k(ctx, kraw, kc, dummy, T, 0, cache_len)
    var o = attn_cached(ctx, qd, kc, vd, dummy, T, 0, cache_len)
    ctx.synchronize()
    var m = max_abs(o, read_f32(dir + "/expected.bin"))
    var ok = m < TOL
    print(
        "  ",
        dir,
        " T=",
        T,
        " max_abs=",
        m,
        " [",
        "OK" if ok else "FAIL",
        "]",
        sep="",
    )
    return ok


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")
    var ctx = DeviceContext()
    var root = "tests/fixtures/attention/"
    print("attention gate — GPU vs NumPy reference (tol", TOL, "):")
    var all_ok = True
    for name in [String("synthetic"), String("real_L0"), String("real_L23")]:
        all_ok = run(ctx, root + name) and all_ok
    if not all_ok:
        raise Error("GPU attention does NOT match the reference — gate FAILED")
    print(
        "OK — Mojo Metal RoPE+GQA attention matches the reference on all"
        " fixtures"
    )
