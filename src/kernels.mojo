"""Reusable Mojo Metal kernels for the Qwen2 forward pass (ARCHITECTURE.md §3).

Each kernel was verified in isolation against a NumPy/torch reference (Phase 1–2,
see §11): attention+RoPE, matmul(+bias), RMSNorm, SwiGLU's silu·mul, plus the
embedding gather, residual add, and bf16→f32 weight conversion the full model
needs. All operate on flat 1D buffers; callers bind the layout type and launch.

Hardcoded to Qwen2.5-0.5B (ARCHITECTURE.md §2): 14 query heads, 2 kv heads,
head_dim 64, RoPE θ=1e6, RMSNorm ε=1e-6.
"""

from std.math import sqrt, exp, log, cos, sin, ceildiv
from std.gpu import global_idx, thread_idx, block_idx, barrier, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import sum as warp_sum, max as warp_max
from std.memory import stack_allocation
from std.collections import InlineArray
from std.sys.info import external_call   # emit AIR simdgroup-matrix ops (see matmul_simd_kernel)
from layout import TileTensor, TensorLayout

# Head/hidden dims are NOT fixed here: the head-sensitive kernels (rope_q/k,
# attn_cached, flash) take HQ/HKV/HEAD_DIM as comptime params so one build serves
# multiple Qwen2.5 sizes (0.5B: 14/2/64, 3B: 16/2/128). THETA/EPS are shared
# across all Qwen2.5 sizes, so they stay module constants.
comptime THETA = Float32(1000000.0)   # RoPE base
comptime EPS = Float32(1.0e-6)        # RMSNorm epsilon
comptime FLASH_PW = 3                 # flash query-tile: warps/block = FLASH_PW * GROUP


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


# ── simdgroup-matrix GEMM (prefill, opt-in via runtime capability gate) ────────
# Apple's AIR simdgroup_matrix_8x8 ops run X·Wᵀ on the GPU's matrix units. They
# are *external AIR functions* (not LLVM intrinsics), so `external_call` names
# them — `llvm_intrinsic` can't (it only sees upstream LLVM intrinsics) and Metal
# has no inline-asm dialect. The exact mangled symbols + signatures below were
# verified by disassembling a compiled .metal (metal-objdump): the fragment is
# <64 x float> (the whole 8x8 matrix), and load/store take three <2xi64> vectors
# (dims, row-stride, origin). A runtime probe (model.probe_simd_gemm) checks the
# toolchain accepts them and falls back to matmul_tiled_kernel otherwise — the
# symbol names are not an API contract and a Metal toolchain bump could break
# them (a mismatch is a catchable pipeline-state error, not a crash).
#
# Each threadgroup computes a 32×32 output tile with 4 simdgroups; per k-step it
# stages an X block (32×8) and a transposed+widened W block (8×32, so the fragment
# orientation yields X·Wᵀ) into threadgroup memory, then each simdgroup loads one
# 8×8 A-fragment and reuses it across 4 B-column fragments — 16 MACs/threadgroup,
# f32 accumulate. The 32×32 tiling cuts redundant device traffic ~4× vs a naive
# 8×8-per-simdgroup kernel (which is memory-bandwidth-bound), giving ~4.5× the
# scalar matmul_tiled_kernel (~1.1 TFLOP/s vs ~250 GFLOP/s on the M4). Output is
# f32 like the scalar path but NOT bit-identical (hardware FMA/order differ;
# measured |Δ| ≲ 2.4e-6), so greedy-parity is re-checked after integration.
comptime _SG_FRAG = SIMD[DType.float32, 64]   # an 8×8 fragment = the whole matrix
comptime _SG_V2 = SIMD[DType.int64, 2]
comptime _SG_LD = "air.simdgroup_matrix_8x8_load.v64f32.p3f32"
comptime _SG_MAC = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32"
comptime _SG_ST = "air.simdgroup_matrix_8x8_store.v64f32.p3f32"
comptime SG_BM = 32              # threadgroup output rows
comptime SG_BN = 32              # threadgroup output cols
comptime SG_NCT = 4              # column-tiles per simdgroup (SG_BN // 8)
comptime SG_TPB = 128            # threads/block = (SG_BM // 8) simdgroups × 32


def matmul_simd_kernel[
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
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias) on the simdgroup-matrix units. Same
    signature/semantics as matmul_tiled_kernel; launch with grid_dim=
    (ceildiv(N,SG_BN), ceildiv(M,SG_BM)), block_dim=SG_TPB."""
    comptime assert X.flat_rank == 1
    var tid = thread_idx.x
    var sg = Int(tid) // 32                    # simdgroup id 0..3
    var m0 = Int(block_idx.y) * SG_BM
    var n0 = Int(block_idx.x) * SG_BN

    var sA = stack_allocation[SG_BM * 8, Float32, address_space = AddressSpace.SHARED]()
    var sB = stack_allocation[8 * SG_BN, Float32, address_space = AddressSpace.SHARED]()
    var sC = stack_allocation[SG_BM * SG_BN, Float32, address_space = AddressSpace.SHARED]()

    var dims = _SG_V2(8, 8)
    var lay8 = _SG_V2(1, 8)                    # row stride 8  (sA is SG_BM×8)
    var layN = _SG_V2(1, SG_BN)                # row stride 32 (sB, sC are *×SG_BN)
    var origin = _SG_V2(0, 0)

    var acc = InlineArray[_SG_FRAG, SG_NCT](fill=_SG_FRAG(0.0))

    var kt = 0
    while kt < K:
        # stage X block (SG_BM×8) → sA[r*8+c]
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
        # stage W block transposed+widened → sB[k*SG_BN+n] = bf16(W[n0+n, kt+k])
        for j in range(8 * SG_BN // SG_TPB):
            var c = Int(tid) + SG_TPB * j
            var kr = c // SG_BN
            var nl = c % SG_BN
            var nn = n0 + nl
            var kk = kt + kr
            var wv = Float32(0.0)
            if nn < N and kk < K:
                wv = bf16_widen(rebind[Scalar[DType.uint16]](W[nn * K + kk]))
            sB[c] = wv
        barrier()

        var fa = external_call[_SG_LD, _SG_FRAG](sA + sg * 8 * 8, dims, lay8, origin)
        for ct in range(SG_NCT):
            var fb = external_call[_SG_LD, _SG_FRAG](sB + ct * 8, dims, layN, origin)
            acc[ct] = external_call[_SG_MAC, _SG_FRAG](fa, fb, acc[ct])
        barrier()
        kt += 8

    for ct in range(SG_NCT):
        external_call[_SG_ST, NoneType](acc[ct], sC + sg * 8 * SG_BN + ct * 8, dims, layN, origin)
    barrier()

    # copy sC (SG_BM×SG_BN) → Y with bias + boundary mask
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
    LT: TensorLayout, HKV: Int, HEAD_DIM: Int
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
    comptime HALF = HEAD_DIM // 2
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
    LT: TensorLayout, HQ: Int, HEAD_DIM: Int
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
    comptime HALF = HEAD_DIM // 2
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
    LT: TensorLayout, HQ: Int, HKV: Int, HEAD_DIM: Int
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
    comptime GROUP = HQ // HKV
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


comptime FLASH_BK = WARP_SIZE   # flash keys per tile = one per lane


def flash_attn_kernel[
    LT: TensorLayout, HQ: Int, HKV: Int, HEAD_DIM: Int, PW: Int
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM] *RoPE-rotated*
    Kc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM] rotated, row = abs pos
    Vc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],     # [Tq, HQ, HEAD_DIM]
    Tq: Int,
    q_offset: Int,
):
    """Flash variant of attn_cached_kernel for *long context*: identical math,
    K/V streamed through threadgroup shared memory instead of re-read from global.

    attn_cached_kernel gives each (query, head) its own warp that reads every past
    K/V straight from the cache — fine until the f32 KV working set (≈ pos·128·8 B)
    outgrows the M4 system cache, at which point attention goes DRAM-bound and the
    cost super-cliffs (measured ~M^3.9 past ~16K tokens). Here a block owns FLASH_PW
    consecutive query positions × all GROUP query heads of one kv-head — PW*GROUP
    warps that all share the *same* K/V. They cooperatively stage each FLASH_BK-key
    tile of K/V into shared memory once and every warp reads it from there, so K/V
    global traffic drops by the full GROUP (head reuse) × FLASH_PW (query reuse) and
    the kernel scales as clean O(M²) — ~2.5× over attn_cached at 32K (but ~3× slower
    below the cliff from the staging overhead, so the caller dispatches by context
    length). Packing all GROUP heads of a kv-head (vs one head per block) is a
    further ~1.3× over the single-head layout; FLASH_PW=3 (21 warps / 672 threads)
    is the measured occupancy sweet spot — bigger blocks regress on register pressure.

    Lane l still owns keys l, l+FLASH_BK, l+2·FLASH_BK, … in increasing order — the
    exact per-lane sequence and online-softmax update order of attn_cached_kernel —
    and the staged values are bit-identical f32 copies, so the output is bit-for-bit
    the same (verified max|diff|=0). Only the read path differs."""
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    comptime GROUP = HQ // HKV
    comptime NWARP = PW * GROUP
    comptime NTHREAD = NWARP * WARP_SIZE
    var Ks = stack_allocation[FLASH_BK * HEAD_DIM, Float32, address_space = AddressSpace.SHARED]()
    var Vs = stack_allocation[FLASH_BK * HEAD_DIM, Float32, address_space = AddressSpace.SHARED]()

    var tib = Int(thread_idx.x)
    var warp = tib // WARP_SIZE
    var lane = tib % WARP_SIZE
    var blk = Int(block_idx.x)
    var kvh = blk % HKV
    var q0 = (blk // HKV) * PW
    var qi = warp // GROUP          # query position within the tile (0 … PW-1)
    var gi = warp % GROUP           # head within the kv-group (0 … GROUP-1)
    var t = q0 + qi
    var h = kvh * GROUP + gi
    var qpos = q_offset + t
    var scale = 1.0 / sqrt(Float32(HEAD_DIM))
    var active = t < Tq

    var qreg = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    if active:
        var qbase = (t * HQ + h) * HEAD_DIM
        for c in range(NVEC):
            qreg[c] = Q.raw_load[VEC](qbase + c * VEC)

    var m = Float32(-1.0e30)
    var lsum = Float32(0.0)
    var accv = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)

    # Block-uniform key range: every warp runs the same tile count so barriers line up.
    var t_max = q0 + PW - 1
    if t_max > Tq - 1:
        t_max = Tq - 1
    var kpos_max = q_offset + t_max

    var kt0 = 0
    while kt0 <= kpos_max:
        for idx in range(tib, FLASH_BK * HEAD_DIM, NTHREAD):
            var r = idx // HEAD_DIM
            var c = idx % HEAD_DIM
            var gk = kt0 + r
            if gk <= kpos_max:
                var src = (gk * HKV + kvh) * HEAD_DIM + c
                Ks[idx] = rebind[Scalar[DType.float32]](Kc[src])
                Vs[idx] = rebind[Scalar[DType.float32]](Vc[src])
            else:
                Ks[idx] = Float32(0.0)
                Vs[idx] = Float32(0.0)
        barrier()

        if active:
            var j = kt0 + lane
            if j <= qpos:
                var kb = lane * HEAD_DIM
                var s = SIMD[DType.float32, VEC](0.0)
                for c in range(NVEC):
                    s += qreg[c] * Ks.load[width=VEC](kb + c * VEC)
                var score = s.reduce_add() * scale
                var m_new = max(m, score)
                var corr = exp(m - m_new)
                var p = exp(score - m_new)
                lsum = lsum * corr + p
                for c in range(NVEC):
                    accv[c] = accv[c] * corr + p * Vs.load[width=VEC](kb + c * VEC)
                m = m_new
        barrier()
        kt0 += FLASH_BK

    if active:
        var m_g = warp_max(m)
        var f = exp(m - m_g)
        var l_g = warp_sum(lsum * f)
        var obase = (t * HQ + h) * HEAD_DIM
        for c in range(NVEC):
            for e in range(VEC):
                var a = warp_sum(accv[c][e] * f)
                if lane == 0:
                    O[obase + c * VEC + e] = rebind[O.ElementType](a / l_g)
