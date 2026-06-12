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
from std.sys import llvm_intrinsic        # compact AIR simdgroup-matrix MMA (see matmul_simd_kernel)
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
# Apple's AIR simdgroup_matrix_8x8 multiply-accumulate runs X·Wᵀ on the GPU's
# matrix units. Modular shipped the COMPACT 8×8 op as the LLVM intrinsic
# `llvm.air.simdgroup_matrix_8x8_multiply_accumulate` (their
# max/kernels/.../gpu/apple/matmul_8x8.mojo, commit cc40bcd — see
# .scratch/ref_matmul_8x8.mojo), so we reach it via `llvm_intrinsic`, no
# external_call / no disassembled-symbol ABI. The fragment is COMPACT —
# SIMD[f32,2] (2 floats per lane), so the whole MxN accumulator grid stays
# register-resident across the K-loop. That is the MLX register-blocking lever
# the old full-`SIMD[f32,64]` external_call ABI blocked: it spilled
# (.scratch/simd2_gemm.mojo, 0.14 TFLOP/s). The compact path does NOT spill and
# runs ~2× the external_call kernel on the M4 (~2.1 vs ~1.1 TFLOP/s).
#
# Layout (Modular's `_frag8_layout`, ground-truthed via Metal `thread_elements()`):
# lane owns (frow, fcol) and (frow, fcol+1). Each threadgroup computes a 64×64
# output tile with 4 simdgroups; each simdgroup owns a 32×32 subtile = a 4×4 grid
# of 8×8 fragments (16 f32 accumulators / lane). Fragments load DIRECTLY from
# global memory at the lane's computed slot — no threadgroup staging, no barriers.
# K-loop steps by 8; the final partial-K block (K%8≠0) is masked. f32 accumulate;
# output is f32 but NOT bit-identical to the scalar path (hardware FMA/order
# differ; measured |Δ| ≲ 1.5e-4 at prefill sizes), so greedy-parity is re-checked.
#
# A runtime probe (model.probe_simd_gemm) still gates the path and falls back to
# matmul_tiled_kernel if the toolchain/GPU rejects the intrinsic. (The compact
# 16×16 op is hardware-gated to M5/GPUFamily10 — see .scratch/mma16_test.mojo —
# but this 8×8 op runs on M1–M4.)
comptime _MMA8 = 8
comptime _FRAG8 = 2              # 8×8 = 64 elems / 32 lanes = 2 floats per lane
comptime SG_BM = 64              # threadgroup output rows
comptime SG_BN = 64              # threadgroup output cols
comptime _SG_SGM = SG_BM // 2    # 32 — simdgroup subtile rows
comptime _SG_SGN = SG_BN // 2    # 32 — simdgroup subtile cols
comptime _SG_NTM = _SG_SGM // _MMA8   # 4 row-fragments per simdgroup
comptime _SG_NTN = _SG_SGN // _MMA8   # 4 col-fragments per simdgroup
comptime SG_TPB = 4 * 32         # 128 threads/block = 4 simdgroups × 32


@always_inline
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8×8 simdgroup-matrix per-lane layout (Modular's `_frag8_layout`,
    ground-truthed via Metal `thread_elements()`). Lane owns (row, col_base) and
    (row, col_base+1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _mma8x8(
    a: SIMD[DType.float32, _FRAG8],
    b: SIMD[DType.float32, _FRAG8],
    c: SIMD[DType.float32, _FRAG8],
) -> SIMD[DType.float32, _FRAG8]:
    """One 8×8×8 simdgroup-matrix multiply-accumulate: D = A·B + C (compact frag)."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, _FRAG8],
    ](a, b, c)


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
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias) on the compact 8×8 simdgroup-matrix units.
    Same signature/semantics as matmul_tiled_kernel; launch with grid_dim=
    (ceildiv(N,SG_BN), ceildiv(M,SG_BM)), block_dim=SG_TPB."""
    comptime assert X.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(thread_idx.x) // 32           # simdgroup id 0..3
    var row_base = Int(block_idx.y) * SG_BM + (sg // 2) * _SG_SGM
    var col_base = Int(block_idx.x) * SG_BN + (sg % 2) * _SG_SGN
    # Fully-interior subtile → unguarded loads (the common case).
    var interior = (row_base + _SG_SGM <= M) and (col_base + _SG_SGN <= N)

    var xp = X.ptr
    var wp = W.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    var nkt = ceildiv(K, _MMA8)
    for ks in range(nkt):
        var kk = ks * _MMA8
        var ktail = (kk + _MMA8 > K)           # final K-block is partial
        # A (X) is f32 [M,K] row-major: lane's 2 frag elems are consecutive K
        # cols kk+fcol, kk+fcol+1. K bound only on the partial tail block.
        var afrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM](uninitialized=True)
        for mi in range(_SG_NTM):
            var grow = row_base + mi * _MMA8 + frow
            if (interior or grow < M) and not ktail:
                afrag[mi] = (xp + grow * K + kk + fcol).load[width=_FRAG8]()
            else:
                var af = SIMD[DType.float32, _FRAG8](0)
                if interior or grow < M:
                    for s in range(_FRAG8):
                        if kk + fcol + s < K:
                            af[s] = xp[grow * K + kk + fcol + s]
                afrag[mi] = af
        # B (W) is bf16 [N,K] (transpose_b): B[k_idx, j] = bf16(W[j, k_idx]).
        # frag slots differ in j (col); row is kk+frow (K bound only on the tail).
        var bfrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTN](uninitialized=True)
        for ni in range(_SG_NTN):
            var bf = SIMD[DType.float32, _FRAG8](0)
            if not ktail or kk + frow < K:
                for s in range(_FRAG8):
                    var gj = col_base + ni * _MMA8 + fcol + s
                    if interior or gj < N:
                        bf[s] = bf16_widen(wp[gj * K + kk + frow])
            bfrag[ni] = bf
        for mi in range(_SG_NTM):
            for ni in range(_SG_NTN):
                acc[mi * _SG_NTN + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                )

    for mi in range(_SG_NTM):
        for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


# ── group-128 int4 weights (opt-in, e.g. for the 3B) ──────────────────────────
# Weight W[N,K] (K a multiple of 128) is stored as symmetric RTN int4 in
# 128-wide groups along K: packed u32[N*K/8] (8 signed nibbles/word, q+8 ∈ 0..15;
# the nibble for linear index `lin = n*K+k` sits in word lin>>3 at bit-shift
# 4*(lin&7)) + scales f32[N*(K/128)]. Dequant = (nibble-8)*scale[n, k/128]. This
# keeps coherent quality on the 3B (per-channel int4 collapses on weight
# outliers; 128-groups bound each scale's span — validated ~85% top-1, KL 0.16).
# Only the W-read changes vs the bf16 kernels; the matmul math (and the
# simdgroup-matrix path) is identical, so the 4.5× prefill carries over.
comptime Q4_GROUP = 128
comptime Q4_SHIFT = 7                  # log2(Q4_GROUP)
comptime _Q4_SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)


@always_inline
def q4_deq[LT: TensorLayout](
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int, k: Int, K: Int, NG: Int,
) -> Float32:
    """Dequant a single weight (n,k). Used by the prefill GEMM W-staging, where
    the matmul (not the dequant) dominates."""
    comptime assert P.flat_rank == 1
    var lin = n * K + k
    var w = Int(rebind[Scalar[DType.uint32]](P[lin >> 3]))
    var nib = (w >> ((lin & 7) * 4)) & 0xF
    var s = rebind[Scalar[DType.float32]](S[n * NG + (k >> Q4_SHIFT)])
    return Float32(nib - 8) * s


def matmul_q4_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int, K: Int, N: Int, NG: Int, use_bias: Int,
):
    """Decode GEMV for int4 weights (M=1). Like matmul_kernel, but each lane
    consumes one u32 (8 weights) per step → coalesced 128-byte loads, and the 8
    nibbles are unpacked with vector ops (a scalar unpack loop is ~2.5× slower —
    it makes the kernel ALU-bound, wasting the 4× lower weight traffic). The
    group scale is folded once per word (a 128-group is a multiple of 8, so an
    8-aligned word never straddles two groups). ~2× the bf16 GEMV on the M4."""
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var words = K // 8
    var rowword = n * words
    var xbase = m * K
    var acc = Float32(0.0)
    for w in range(lane, words, WARP_SIZE):
        var packed = rebind[Scalar[DType.uint32]](P[rowword + w])
        var k0 = w * 8
        var s = rebind[Scalar[DType.float32]](S[n * NG + (k0 >> Q4_SHIFT)])
        var nibs = (SIMD[DType.uint32, 8](packed) >> _Q4_SHIFTS) & 0xF
        var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
        var xv = SIMD[DType.float32, 8](0.0)
        for o in range(8):
            xv[o] = rebind[Scalar[DType.float32]](X[xbase + k0 + o])
        acc += (qf * xv).reduce_add() * s
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_q4_resid_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[DType.float32, LT, MutAnyOrigin],   # residual added in the epilogue
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int, K: Int, N: Int, NG: Int, use_bias: Int,
):
    """matmul_q4_kernel with a fused residual add (Y = X·Wᵀ(+bias) + R). Folds the
    decode residual into the proj GEMV, saving one launch per layer (×2: o & down)."""
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var words = K // 8
    var rowword = n * words
    var xbase = m * K
    var acc = Float32(0.0)
    for w in range(lane, words, WARP_SIZE):
        var packed = rebind[Scalar[DType.uint32]](P[rowword + w])
        var k0 = w * 8
        var s = rebind[Scalar[DType.float32]](S[n * NG + (k0 >> Q4_SHIFT)])
        var nibs = (SIMD[DType.uint32, 8](packed) >> _Q4_SHIFTS) & 0xF
        var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
        var xv = SIMD[DType.float32, 8](0.0)
        for o in range(8):
            xv[o] = rebind[Scalar[DType.float32]](X[xbase + k0 + o])
        acc += (qf * xv).reduce_add() * s
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        total += rebind[Scalar[DType.float32]](R[m * N + n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_resid_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.uint16, LT, MutAnyOrigin],   # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[DType.float32, LT, MutAnyOrigin],   # residual added in the epilogue
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int, K: Int, N: Int, use_bias: Int,
):
    """matmul_kernel (bf16 decode GEMV) with a fused residual add."""
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
        var wv = bf16_widen(rebind[Scalar[DType.uint16]](W[n * K + k]))
        acc += xv * wv
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        total += rebind[Scalar[DType.float32]](R[m * N + n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_tiled_q4_kernel[
    LT: TensorLayout, TM: Int, CN: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int, K: Int, N: Int, NG: Int, use_bias: Int,
):
    """int4 scalar prefill fallback — matmul_tiled_kernel with q4_deq W-reads.
    Used only if the simdgroup-matrix probe fails."""
    comptime assert X.flat_rank == 1
    var ncols = ceildiv(N, CN)
    var tile = Int(global_idx.x) // WARP_SIZE
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
                wv[c] = q4_deq(P, S, n0 + c, k, K, NG)
        for mm in range(TM):
            var m = m0 + mm
            if m < M:
                var xv = rebind[Scalar[DType.float32]](X[m * K + k])
                for c in range(CN):
                    acc[mm * CN + c] += xv * wv[c]
    for mm in range(TM):
        var m = m0 + mm
        for c in range(CN):
            var total = warp_sum(acc[mm * CN + c])
            var n = n0 + c
            if lane == 0 and m < M and n < N:
                var bias = Float32(0.0)
                if use_bias != 0:
                    bias = rebind[Scalar[DType.float32]](B[n])
                Y[m * N + n] = rebind[Y.ElementType](total + bias)


def matmul_simd_q4_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M: Int, K: Int, N: Int, NG: Int, use_bias: Int,
):
    """int4 prefill GEMM: matmul_simd_kernel with int4-dequant W reads. The
    compact 8×8 simdgroup-matrix math is byte-for-byte the bf16 kernel's — only
    the B fragment is filled from q4_deq instead of bf16_widen — so the ~2×
    register-blocked speedup carries."""
    comptime assert X.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(thread_idx.x) // 32
    var row_base = Int(block_idx.y) * SG_BM + (sg // 2) * _SG_SGM
    var col_base = Int(block_idx.x) * SG_BN + (sg % 2) * _SG_SGN
    var interior = (row_base + _SG_SGM <= M) and (col_base + _SG_SGN <= N)

    var xp = X.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    var nkt = ceildiv(K, _MMA8)
    for ks in range(nkt):
        var kk = ks * _MMA8
        var ktail = (kk + _MMA8 > K)
        var afrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM](uninitialized=True)
        for mi in range(_SG_NTM):
            var grow = row_base + mi * _MMA8 + frow
            if (interior or grow < M) and not ktail:
                afrag[mi] = (xp + grow * K + kk + fcol).load[width=_FRAG8]()
            else:
                var af = SIMD[DType.float32, _FRAG8](0)
                if interior or grow < M:
                    for s in range(_FRAG8):
                        if kk + fcol + s < K:
                            af[s] = xp[grow * K + kk + fcol + s]
                afrag[mi] = af
        # B (W) is int4 [N,K] (transpose_b): B[k_idx, j] = deq(W[j, k_idx]).
        var bfrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTN](uninitialized=True)
        for ni in range(_SG_NTN):
            var bf = SIMD[DType.float32, _FRAG8](0)
            var krow = kk + frow
            if not ktail or krow < K:
                for s in range(_FRAG8):
                    var gj = col_base + ni * _MMA8 + fcol + s
                    if interior or gj < N:
                        bf[s] = q4_deq(P, S, gj, krow, K, NG)
            bfrag[ni] = bf
        for mi in range(_SG_NTM):
            for ni in range(_SG_NTN):
                acc[mi * _SG_NTN + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                )

    for mi in range(_SG_NTM):
        for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


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


def silu_mul_cat_kernel[
    LT: TensorLayout
](
    GU: TileTensor[DType.float32, LT, MutAnyOrigin],   # [T, 2*inter]: row = gate(inter) ++ up(inter)
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],    # [T, inter]
    T: Int,
    inter: Int,
):
    """SwiGLU activation reading the *fused* gate+up GEMV output (one buffer, gate
    then up per row): Y = silu(gate)·up. Lets gate+up be one GEMV instead of two —
    the split happens here for free instead of via separate buffers."""
    comptime assert GU.flat_rank == 1
    var idx = global_idx.x
    if idx >= T * inter:
        return
    var t = idx // inter
    var i = idx % inter
    var g = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i])
    var u = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i + inter])
    Y[idx] = rebind[Y.ElementType]((g / (1.0 + exp(-g))) * u)


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


def copy_strided_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],   # row = in_stride, slice at in_off
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    T: Int,
    in_stride: Int,
    in_off: Int,
    dst_off: Int,
    n: Int,                                             # slice width per row
):
    """Copy a [T, n] column-slice out of a strided source (e.g. the V part of a
    fused [q|k|v] buffer) into dst[dst_off:] contiguously — V into its cache rows."""
    comptime assert dst.flat_rank == 1
    var idx = global_idx.x
    if idx >= T * n:
        return
    var t = idx // n
    var j = idx % n
    dst[dst_off + t * n + j] = rebind[dst.ElementType](src[t * in_stride + in_off + j])


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


@always_inline
def _head_rrms[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],
    base: Int,
    HEAD_DIM: Int,
) -> Float32:
    """Reciprocal RMS over one head's HEAD_DIM contiguous elements at `base`:
    1 / sqrt(mean(x²) + EPS). Qwen3 QK-RMSNorm normalizes per head before RoPE."""
    var ss = Float32(0.0)
    for d in range(HEAD_DIM):
        var v = rebind[Scalar[DType.float32]](src[base + d])
        ss += v * v
    return 1.0 / sqrt(ss / Float32(HEAD_DIM) + EPS)


def rope_k_kernel[
    LT: TensorLayout, HKV: Int, HEAD_DIM: Int, QK_NORM: Bool = False
](
    Kin: TileTensor[DType.float32, LT, MutAnyOrigin],   # K source (row = in_stride, K at in_off)
    Kc: TileTensor[DType.float32, LT, MutAnyOrigin],    # [max, HKV, HEAD_DIM] cache (rotated)
    Kn: TileTensor[DType.float32, LT, MutAnyOrigin],    # [HEAD_DIM] k_norm weight (dummy if !QK_NORM)
    Tq: Int,
    q_offset: Int,
    in_stride: Int,    # source row stride (= nkv unfused, = hd+2nkv when reading fused qkv)
    in_off: Int,       # source column offset of K within the row (0 unfused, hd when fused)
):
    """Apply RoPE to freshly-projected K and write it into the cache at its
    absolute-position rows. Doing this once on write (one thread per token×kv-
    head) replaces recomputing K's RoPE for every past key on every decode step
    inside attention (§11 #12). Same split-half rotation/θ as the Q path. Kin may
    be a strided slice of a fused [q|k|v] buffer (in_stride/in_off).

    When QK_NORM (Qwen3), each head is first RMS-normalized over HEAD_DIM and
    scaled by Kn[d] before the rotation (HF: q_norm/k_norm applied pre-RoPE)."""
    comptime assert Kin.flat_rank == 1
    comptime HALF = HEAD_DIM // 2
    var nkv = HKV * HEAD_DIM
    var idx = Int(global_idx.x)
    if idx >= Tq * HKV:
        return
    var t = idx // HKV
    var kvh = idx % HKV
    var pos = q_offset + t
    var inbase = t * in_stride + in_off + kvh * HEAD_DIM
    var outbase = (q_offset + t) * nkv + kvh * HEAD_DIM   # cache row = absolute position
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(Kin, inbase, HEAD_DIM)
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(pos) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Kin[inbase + d])
        var x1 = rebind[Scalar[DType.float32]](Kin[inbase + d + HALF])
        comptime if QK_NORM:
            var g0 = rebind[Scalar[DType.float32]](Kn[d])
            var g1 = rebind[Scalar[DType.float32]](Kn[d + HALF])
            x0 = x0 * rrms * g0
            x1 = x1 * rrms * g1
        Kc[outbase + d] = rebind[Kc.ElementType](x0 * c - x1 * s)
        Kc[outbase + d + HALF] = rebind[Kc.ElementType](x1 * c + x0 * s)


def rope_q_kernel[
    LT: TensorLayout, HQ: Int, HEAD_DIM: Int, QK_NORM: Bool = False
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],     # Q source (row = in_stride, Q at in_off)
    Qr: TileTensor[DType.float32, LT, MutAnyOrigin],    # [Tq, HQ, HEAD_DIM] rotated out (contiguous)
    Qn: TileTensor[DType.float32, LT, MutAnyOrigin],    # [HEAD_DIM] q_norm weight (dummy if !QK_NORM)
    Tq: Int,
    q_offset: Int,
    in_stride: Int,    # source row stride (= hd unfused, = hd+2nkv when reading fused qkv)
    in_off: Int,       # source column offset of Q within the row (0; Q is first in [q|k|v])
):
    """Apply RoPE to Q (one thread per query×head) into a rotated buffer, so the
    attention kernel itself does no transcendentals — same as K is rotated on
    write (§11 #12). Position = absolute query position q_offset+t. Q may be a
    strided slice of a fused [q|k|v] buffer (in_stride/in_off); Qr is contiguous.

    When QK_NORM (Qwen3), each head is first RMS-normalized over HEAD_DIM and
    scaled by Qn[d] before the rotation (HF: q_norm applied pre-RoPE)."""
    comptime assert Q.flat_rank == 1
    comptime HALF = HEAD_DIM // 2
    var idx = Int(global_idx.x)
    if idx >= Tq * HQ:
        return
    var t = idx // HQ
    var h = idx % HQ
    var pos = q_offset + t
    var inb = t * in_stride + in_off + h * HEAD_DIM
    var base = idx * HEAD_DIM
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(Q, inb, HEAD_DIM)
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(pos) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Q[inb + d])
        var x1 = rebind[Scalar[DType.float32]](Q[inb + d + HALF])
        comptime if QK_NORM:
            var g0 = rebind[Scalar[DType.float32]](Qn[d])
            var g1 = rebind[Scalar[DType.float32]](Qn[d + HALF])
            x0 = x0 * rrms * g0
            x1 = x1 * rrms * g1
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
