"""Reusable Mojo Metal kernels for the Qwen2 forward pass (ARCHITECTURE.md §3).

Each kernel was verified in isolation against a NumPy/torch reference (Phase 1–2,
see §11): attention+RoPE, matmul(+bias), RMSNorm, SwiGLU's silu·mul, plus the
embedding gather, residual add, and bf16→f32 weight conversion the full model
needs. All operate on flat 1D buffers; callers bind the layout type and launch.

Hardcoded to Qwen2.5-0.5B (ARCHITECTURE.md §2): 14 query heads, 2 kv heads,
head_dim 64, RoPE θ=1e6, RMSNorm ε=1e-6.
"""

from std.math import sqrt, exp, log, cos, sin, ceildiv
from std.gpu import global_idx, WARP_SIZE
from std.gpu.primitives.warp import sum as warp_sum, max as warp_max
from std.collections import InlineArray
from layout import TileTensor, TensorLayout

comptime HQ = 14
comptime HKV = 2
comptime HEAD_DIM = 64
comptime HALF = HEAD_DIM // 2
comptime GROUP = HQ // HKV
comptime THETA = Float32(1000000.0)
comptime EPS = Float32(1.0e-6)


@always_inline
def bf16_widen(u: Scalar[DType.uint16]) -> Float32:
    """Widen a bf16 (stored as its raw u16 bits) to f32 — exact, since bf16 is
    the top 16 bits of f32. Weights live on-device as bf16 to halve matmul read
    traffic; the accumulate stays f32 (§11 #12)."""
    var bits: UInt32 = UInt32(u) << 16
    return UnsafePointer(to=bits).bitcast[Float32]()[0]


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
    emb: TileTensor[DType.uint16, LT, MutAnyOrigin],   # bf16 embedding table
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
    dst[i] = rebind[dst.ElementType](bf16_widen(rebind[Scalar[DType.uint16]](emb[tok * H + d])))


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
    # One warp per row: the old kernel ran the whole H-element reduction on a
    # single thread (one thread per row → 1 thread for decode's T=1), which made
    # RMSNorm as costly as a 4864-wide matmul (§11 #12). Lanes split the row,
    # warp_sum reduces, then each lane writes its slice (coalesced).
    var t = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if t >= T:
        return
    var ss = Float32(0.0)
    for d in range(lane, H, WARP_SIZE):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        ss += v * v
    var rms = sqrt(warp_sum(ss) / Float32(H) + EPS)   # warp_sum broadcasts to all lanes
    for d in range(lane, H, WARP_SIZE):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        var wv = rebind[Scalar[DType.float32]](W[d])
        Y[t * H + d] = rebind[Y.ElementType](v / rms * wv)


def matmul_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.uint16, LT, MutAnyOrigin],   # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
):
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias). One warp per output element.

    The decode path is memory-bound GEMV (M=1): the cost is streaming the
    weight matrix W from device memory. The earlier one-thread-per-output
    kernel had each thread walk a full row W[n*K + k] — so adjacent threads
    read addresses K apart, uncoalesced, wasting most of the bandwidth. Here a
    whole warp cooperates on one output: lane L reads W[n*K + L], W[n*K + L+32],
    … so the 32 lanes touch 32 consecutive words each step (coalesced), then
    `warp_sum` reduces the per-lane partials. Pure f32 accumulate, same as
    before, so greedy parity is preserved (§11 #8)."""
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE     # one warp per output element
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var acc = Float32(0.0)
    for k in range(lane, K, WARP_SIZE):
        var xv = rebind[Scalar[DType.float32]](X[m * K + k])
        var wv = bf16_widen(rebind[Scalar[DType.uint16]](W[n * K + k]))
        acc += xv * wv
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_tiled_kernel[
    LT: TensorLayout, TM: Int, CN: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.uint16, LT, MutAnyOrigin],   # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
):
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias) for the *prefill* path (M > 1).

    The decode GEMV (matmul_kernel) gives each (m,n) output its own warp, so it
    re-streams the whole weight matrix once per token. At prefill (M ≈ thousands)
    that is M× the weight traffic. A first cut tiled only the token axis (TM
    tokens/warp, each weight read once and reused TM×), but profiling a 2048-token
    prefill showed it stalled at ~110 GFLOP/s: each of the N output columns got its
    own warp that re-streamed the *whole* X matrix, so X traffic was N·M·K·4 ≈
    36 GB for the MLP gate — 16× the weight traffic and the real bottleneck.

    So tile *both* axes: a warp owns a TM-token × CN-column block (m0…, n0…). Its
    lanes split K; per k each lane reads TM X-values and CN weights once and does
    TM·CN MACs, so X is reused CN× and W is reused TM× — cutting X traffic CN-fold.
    The TM·CN partials are reduced with `warp_sum` at the end. Pure f32 accumulate
    + bf16 widen and the same lane-strided-K → warp_sum reduction as the GEMV, so
    output is bit-identical and greedy parity is preserved (§11 #8, #12). TM=CN=8
    measured ~2× the token-only kernel (~210 GFLOP/s) on the M4."""
    comptime assert X.flat_rank == 1
    var ncols = ceildiv(N, CN)
    var tile = Int(global_idx.x) // WARP_SIZE     # one warp per (column-tile, token-tile)
    var lane = Int(global_idx.x) % WARP_SIZE
    if tile >= ncols * ceildiv(M, TM):
        return
    var n0 = (tile % ncols) * CN
    var m0 = (tile // ncols) * TM
    var acc = InlineArray[Float32, TM * CN](fill=0.0)
    for k in range(lane, K, WARP_SIZE):
        var wv = InlineArray[Float32, CN](fill=0.0)
        for c in range(CN):
            if n0 + c < N:
                wv[c] = bf16_widen(rebind[Scalar[DType.uint16]](W[(n0 + c) * K + k]))
        for mm in range(TM):
            var m = m0 + mm
            if m < M:
                var xv = rebind[Scalar[DType.float32]](X[m * K + k])
                for c in range(CN):
                    acc[mm * CN + c] += xv * wv[c]
    for mm in range(TM):
        var m = m0 + mm
        for c in range(CN):
            var total = warp_sum(acc[mm * CN + c])   # warp collective — every lane
            var n = n0 + c
            if lane == 0 and m < M and n < N:
                var bias = Float32(0.0)
                if use_bias != 0:
                    bias = rebind[Scalar[DType.float32]](B[n])
                Y[m * N + n] = rebind[Y.ElementType](total + bias)


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


def slice_row_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    src_offset: Int,
    n: Int,
):
    """Copy n contiguous elements from src starting at src_offset into dst[0:n].
    Used to lift the last token's hidden row out before the LM head, so prefill
    runs the (VOCAB-wide) head on one row instead of all T (§11 #12)."""
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    dst[i] = rebind[dst.ElementType](src[src_offset + i])


def rope_k_kernel[
    LT: TensorLayout
](
    Kin: TileTensor[DType.float32, LT, MutAnyOrigin],   # [Tq, HKV, HEAD_DIM] raw K from projection
    Kc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM] cache (rotated)
    Tq: Int,
    q_offset: Int,
):
    """Apply RoPE to freshly-projected K and write it into the cache at its
    absolute-position rows. Doing this once on write (one thread per token×kv-
    head) replaces recomputing K's RoPE for every past key on every decode step
    inside attention (§11 #12). Same split-half rotation/θ as the Q path."""
    comptime assert Kin.flat_rank == 1
    var nkv = HKV * HEAD_DIM
    var idx = Int(global_idx.x)
    if idx >= Tq * HKV:
        return
    var t = idx // HKV
    var kvh = idx % HKV
    var pos = q_offset + t
    var inbase = t * nkv + kvh * HEAD_DIM
    var outbase = (q_offset + t) * nkv + kvh * HEAD_DIM   # cache row = absolute position
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(pos) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Kin[inbase + d])
        var x1 = rebind[Scalar[DType.float32]](Kin[inbase + d + HALF])
        Kc[outbase + d] = rebind[Kc.ElementType](x0 * c - x1 * s)
        Kc[outbase + d + HALF] = rebind[Kc.ElementType](x1 * c + x0 * s)


def rope_q_kernel[
    LT: TensorLayout
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM] raw
    Qr: TileTensor[DType.float32, LT, MutAnyOrigin],    # [Tq, HQ, HEAD_DIM] rotated out
    Tq: Int,
    q_offset: Int,
):
    """Apply RoPE to Q (one thread per query×head) into a rotated buffer, so the
    attention kernel itself does no transcendentals — same as K is rotated on
    write (§11 #12). Position = absolute query position q_offset+t."""
    comptime assert Q.flat_rank == 1
    var idx = Int(global_idx.x)
    if idx >= Tq * HQ:
        return
    var t = idx // HQ
    var pos = q_offset + t
    var base = idx * HEAD_DIM
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(pos) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Q[base + d])
        var x1 = rebind[Scalar[DType.float32]](Q[base + d + HALF])
        Qr[base + d] = rebind[Qr.ElementType](x0 * c - x1 * s)
        Qr[base + d + HALF] = rebind[Qr.ElementType](x1 * c + x0 * s)


def attn_cached_kernel[
    LT: TensorLayout
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM] *RoPE-rotated*
    Kc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM] RoPE-rotated, row = abs position
    Vc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM]
    Tq: Int,
    q_offset: Int,
):
    """Causal GQA attention over a KV cache — one *warp* per (query, head).

    The old kernel ran one *thread* per (query, head): for a decode step that is
    14 threads total, each looping every past key serially, so attention was the
    dominant decode cost and grew badly with context (§11 #12). Here the warp's
    32 lanes split the keys; each lane runs a flash/online softmax over its
    subset, then a single cross-lane merge (max → rescale → sum) combines them.
    Q and K are already RoPE-rotated (rope_q/rope_k), so this kernel has no
    transcendentals — just dot products + the online softmax.

    Two refinements over the first warp version (measured ~2.6× at M=2048): Q is
    loaded into registers once instead of re-read from memory for every key, and
    the per-key Q·K dot and V accumulate use SIMD[VEC] vector loads. The vector
    dot sums VEC partials before the horizontal reduce, so the 64-term sum order
    differs from a scalar loop — output drifts by ≤4e-9 (pure f32 rounding), far
    under the forward tolerance, and greedy decode stays token-for-token (§11 #12)."""
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    var qh = Int(global_idx.x) // WARP_SIZE     # one warp per (query, head)
    var lane = Int(global_idx.x) % WARP_SIZE
    var h = qh % HQ
    var t = qh // HQ
    if t >= Tq:
        return
    var kvh = h // GROUP
    var qpos = q_offset + t
    var qbase = (t * HQ + h) * HEAD_DIM
    var scale = 1.0 / sqrt(Float32(HEAD_DIM))

    # Q lives in registers for the whole key loop (NVEC vector chunks).
    var qreg = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for c in range(NVEC):
        qreg[c] = Q.raw_load[VEC](qbase + c * VEC)

    # Each lane runs flash softmax over its slice of keys (j = lane, lane+32, …).
    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var accv = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for j in range(lane, qpos + 1, WARP_SIZE):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        var s = SIMD[DType.float32, VEC](0.0)
        for c in range(NVEC):
            s += qreg[c] * Kc.raw_load[VEC](kbase + c * VEC)
        var score = s.reduce_add() * scale
        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        for c in range(NVEC):
            accv[c] = accv[c] * corr + p * Vc.raw_load[VEC](kbase + c * VEC)
        m = m_new

    # Cross-lane merge: global max, rescale each lane's partials, then sum.
    var m_g = warp_max(m)
    var f = exp(m - m_g)
    var l_g = warp_sum(l * f)
    var obase = (t * HQ + h) * HEAD_DIM
    for c in range(NVEC):
        for e in range(VEC):
            var a = warp_sum(accv[c][e] * f)
            if lane == 0:
                O[obase + c * VEC + e] = rebind[O.ElementType](a / l_g)
