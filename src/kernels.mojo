"""Reusable Mojo Metal kernels for the Qwen2 forward pass (ARCHITECTURE.md §3).

Each kernel was verified in isolation against a NumPy/torch reference (Phase 1–2,
see §11): attention+RoPE, matmul(+bias), RMSNorm, SwiGLU's silu·mul, plus the
embedding gather, residual add, and bf16→f32 weight conversion the full model
needs. All operate on flat 1D buffers; callers bind the layout type and launch.

Hardcoded to Qwen2.5-0.5B (ARCHITECTURE.md §2): 14 query heads, 2 kv heads,
head_dim 64, RoPE θ=1e6, RMSNorm ε=1e-6.
"""

from std.math import sqrt, exp, log, cos, sin
from std.gpu import global_idx
from std.collections import InlineArray
from layout import TileTensor, TensorLayout

comptime HQ = 14
comptime HKV = 2
comptime HEAD_DIM = 64
comptime HALF = HEAD_DIM // 2
comptime GROUP = HQ // HKV
comptime THETA = Float32(1000000.0)
comptime EPS = Float32(1.0e-6)


def cvt_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.uint16, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int,
):
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var u = rebind[Scalar[DType.uint16]](src[i])
    var bits: UInt32 = UInt32(u) << 16
    dst[i] = rebind[dst.ElementType](UnsafePointer(to=bits).bitcast[Float32]()[0])


def embed_kernel[
    LT: TensorLayout
](
    ids: TileTensor[DType.int32, LT, MutAnyOrigin],
    emb: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    T: Int,
    H: Int,
):
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= T * H:
        return
    var t = i // H
    var d = i % H
    var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
    dst[i] = rebind[dst.ElementType](emb[tok * H + d])


def add_kernel[
    LT: TensorLayout
](
    a: TileTensor[DType.float32, LT, MutAnyOrigin],
    b: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int,
):
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var av = rebind[Scalar[DType.float32]](a[i])
    var bv = rebind[Scalar[DType.float32]](b[i])
    dst[i] = rebind[dst.ElementType](av + bv)


def rmsnorm_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    T: Int,
    H: Int,
):
    comptime assert X.flat_rank == 1
    var t = global_idx.x
    if t >= T:
        return
    var ss = Float32(0.0)
    for d in range(H):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        ss += v * v
    var rms = sqrt(ss / Float32(H) + EPS)
    for d in range(H):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        var wv = rebind[Scalar[DType.float32]](W[d])
        Y[t * H + d] = rebind[Y.ElementType](v / rms * wv)


def matmul_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
):
    comptime assert X.flat_rank == 1
    var idx = global_idx.x
    if idx >= M * N:
        return
    var m = idx // N
    var n = idx % N
    var acc = Float32(0.0)
    for k in range(K):
        var xv = rebind[Scalar[DType.float32]](X[m * K + k])
        var wv = rebind[Scalar[DType.float32]](W[n * K + k])
        acc += xv * wv
    if use_bias != 0:
        acc += rebind[Scalar[DType.float32]](B[n])
    Y[m * N + n] = rebind[Y.ElementType](acc)


def silu_mul_kernel[
    LT: TensorLayout
](
    A: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int,
):
    comptime assert A.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var a = rebind[Scalar[DType.float32]](A[i])
    var b = rebind[Scalar[DType.float32]](B[i])
    Y[i] = rebind[Y.ElementType]((a / (1.0 + exp(-a))) * b)


def copy_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst_offset: Int,
    n: Int,
):
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    dst[dst_offset + i] = rebind[dst.ElementType](src[i])


def attn_cached_kernel[
    LT: TensorLayout
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM] raw (pre-RoPE)
    Kc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM] raw, row = abs position
    Vc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM]
    Tq: Int,
    q_offset: Int,
):
    """Attention of Tq queries (at absolute positions q_offset..q_offset+Tq-1)
    over a KV cache. RoPE applied here using the cache row as the position, so
    the cache stores raw K/V — no separate rope pass or rotated cache needed."""
    comptime assert Q.flat_rank == 1
    var h = global_idx.x % HQ
    var t = global_idx.x // HQ
    if t >= Tq:
        return
    var kvh = h // GROUP
    var qpos = q_offset + t

    var qbase = (t * HQ + h) * HEAD_DIM
    var qr = InlineArray[Float32, HEAD_DIM](fill=0.0)
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(qpos) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Q[qbase + d])
        var x1 = rebind[Scalar[DType.float32]](Q[qbase + d + HALF])
        qr[d] = x0 * c - x1 * s
        qr[d + HALF] = x1 * c + x0 * s

    var scale = 1.0 / sqrt(Float32(HEAD_DIM))
    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var acc = InlineArray[Float32, HEAD_DIM](fill=0.0)

    for j in range(qpos + 1):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        var score = Float32(0.0)
        for d in range(HALF):
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
            var ang = Float32(j) * freq
            var c = cos(ang)
            var s = sin(ang)
            var x0 = rebind[Scalar[DType.float32]](Kc[kbase + d])
            var x1 = rebind[Scalar[DType.float32]](Kc[kbase + d + HALF])
            score += qr[d] * (x0 * c - x1 * s) + qr[d + HALF] * (x1 * c + x0 * s)
        score *= scale

        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        var vbase = (j * HKV + kvh) * HEAD_DIM
        for d in range(HEAD_DIM):
            acc[d] = acc[d] * corr + p * rebind[Scalar[DType.float32]](Vc[vbase + d])
        m = m_new

    var obase = (t * HQ + h) * HEAD_DIM
    for d in range(HEAD_DIM):
        O[obase + d] = rebind[O.ElementType](acc[d] / l)


def attn_kernel[
    LT: TensorLayout
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],
    K: TileTensor[DType.float32, LT, MutAnyOrigin],
    V: TileTensor[DType.float32, LT, MutAnyOrigin],
    O: TileTensor[DType.float32, LT, MutAnyOrigin],
    T: Int,
):
    comptime assert Q.flat_rank == 1
    var h = global_idx.x % HQ
    var t = global_idx.x // HQ
    if t >= T:
        return
    var kvh = h // GROUP

    var qbase = (t * HQ + h) * HEAD_DIM
    var qr = InlineArray[Float32, HEAD_DIM](fill=0.0)
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(t) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Q[qbase + d])
        var x1 = rebind[Scalar[DType.float32]](Q[qbase + d + HALF])
        qr[d] = x0 * c - x1 * s
        qr[d + HALF] = x1 * c + x0 * s

    var scale = 1.0 / sqrt(Float32(HEAD_DIM))
    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var acc = InlineArray[Float32, HEAD_DIM](fill=0.0)

    for j in range(t + 1):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        var score = Float32(0.0)
        for d in range(HALF):
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
            var ang = Float32(j) * freq
            var c = cos(ang)
            var s = sin(ang)
            var x0 = rebind[Scalar[DType.float32]](K[kbase + d])
            var x1 = rebind[Scalar[DType.float32]](K[kbase + d + HALF])
            score += qr[d] * (x0 * c - x1 * s) + qr[d + HALF] * (x1 * c + x0 * s)
        score *= scale

        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        var vbase = (j * HKV + kvh) * HEAD_DIM
        for d in range(HEAD_DIM):
            acc[d] = acc[d] * corr + p * rebind[Scalar[DType.float32]](V[vbase + d])
        m = m_new

    var obase = (t * HQ + h) * HEAD_DIM
    for d in range(HEAD_DIM):
        O[obase + d] = rebind[O.ElementType](acc[d] / l)
