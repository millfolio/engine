"""Reusable Mojo Metal kernels for the Qwen2 forward pass (ARCHITECTURE.md §3).

Each kernel was verified in isolation against a NumPy/torch reference (Phase 1–2,
see §11): attention+RoPE, matmul(+bias), RMSNorm, SwiGLU's silu·mul, plus the
embedding gather, residual add, and bf16→f32 weight conversion the full model
needs. All operate on flat 1D buffers; callers bind the layout type and launch.

Hardcoded to Qwen2.5-0.5B (ARCHITECTURE.md §2): 14 query heads, 2 kv heads,
head_dim 64, RoPE θ=1e6, RMSNorm ε=1e-6.
"""

from std.math import sqrt, exp, log, cos, sin, tanh, ceildiv
from std.gpu import (
    global_idx,
    thread_idx,
    block_idx,
    block_dim,
    grid_dim,
    WARP_SIZE,
)
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.gpu.primitives.warp import (
    sum as warp_sum,
    max as warp_max,
    shuffle_down as warp_shuffle_down,
    shuffle_xor,
)
from std.memory import stack_allocation
from std.collections import InlineArray
from std.sys import (
    llvm_intrinsic,
)  # compact AIR simdgroup-matrix MMA (see matmul_simd_kernel)
from layout import TileTensor, TensorLayout

# Head/hidden dims are NOT fixed here: the head-sensitive kernels (rope_q/k,
# attn_cached, flash) take HQ/HKV/HEAD_DIM as comptime params so one build serves
# multiple Qwen2.5 sizes (0.5B: 14/2/64, 3B: 16/2/128). THETA/EPS are shared
# across all Qwen2.5 sizes, so they stay module constants.
comptime THETA = Float32(1000000.0)  # RoPE base
"""RoPE base frequency θ (shared across all Qwen2.5 sizes)."""
comptime EPS = Float32(1.0e-6)  # RMSNorm epsilon
"""RMSNorm epsilon ε added under the square root."""
comptime FLASH_PW = 3  # flash query-tile: warps/block = FLASH_PW * GROUP
"""Flash query-tile width: warps per block = FLASH_PW * GROUP."""


@always_inline
def bf16_widen(u: Scalar[DType.uint16]) -> Float32:
    """Widen a bf16 (stored as its raw u16 bits) to f32 — exact, since bf16 is
    the top 16 bits of f32. Weights live on-device as bf16 to halve matmul read
    traffic; the accumulate stays f32 (§11 #12).

    Args:
        u: A bf16 value as its raw uint16 bit pattern.

    Returns:
        The exact f32 value (u's 16 bits as the high half of the f32).
    """
    var bits: UInt32 = UInt32(u) << 16
    return Pointer(to=bits).unsafe_bitcast[Float32]()[unsafe_offset=0]


def cvt_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.uint16, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    n_arg: Int32,
):
    """Widen `n` bf16 values (raw u16 bits in `src`) to f32 in `dst`, elementwise.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        src: Source buffer of `n` bf16 values (raw u16 bits).
        dst: Output f32 buffer (length `n`), receives the widened values.
        n: Element count.
    """
    var n = Int(n_arg)
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var u = rebind[Scalar[DType.uint16]](src[i])
    var bits: UInt32 = UInt32(u) << 16
    dst[i] = rebind[dst.ElementType](
        Pointer(to=bits).unsafe_bitcast[Float32]()[unsafe_offset=0]
    )


def embed_kernel[
    LT: TensorLayout
](
    ids: TileTensor[DType.int32, LT, MutAnyOrigin],
    emb: TileTensor[DType.uint16, LT, MutAnyOrigin],  # bf16 embedding table
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    T_arg: Int32,
    H_arg: Int32,
):
    """Embedding gather: dst[t,:] = bf16→f32 of emb row ids[t], for T tokens × H dims.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        ids: [T] int32 token ids, one per token.
        emb: Bf16 embedding table (raw u16 bits), row-major [vocab, H].
        dst: Output f32 buffer [T, H], the gathered + widened rows.
        T: Number of tokens.
        H: Hidden size (embedding dimension).
    """
    var T = Int(T_arg)
    var H = Int(H_arg)
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= T * H:
        return
    var t = i // H
    var d = i % H
    var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
    dst[i] = rebind[dst.ElementType](
        bf16_widen(rebind[Scalar[DType.uint16]](emb[tok * H + d]))
    )


def add_kernel[
    LT: TensorLayout
](
    a: TileTensor[DType.float32, LT, MutAnyOrigin],
    b: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    n_arg: Int32,
):
    """Elementwise residual add: dst[i] = a[i] + b[i] over `n` elements.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        a: Input A buffer (f32, length `n`).
        b: Input B buffer (f32, length `n`).
        dst: Output buffer (f32, length `n`), receives a + b.
        n: Element count.
    """
    var n = Int(n_arg)
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
    T_arg: Int32,
    H_arg: Int32,
):
    """RMSNorm: Y[t,d] = X[t,d] / √(mean_d(X[t,:]²)+EPS) · W[d], one warp per row.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input rows [T, H] (f32).
        W: Per-channel RMSNorm weight [H] (f32).
        Y: Output [T, H] (f32), the normalized · weighted rows.
        T: Number of rows (tokens).
        H: Row width (hidden size).
    """
    var T = Int(T_arg)
    var H = Int(H_arg)
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
    var rms = sqrt(
        warp_sum(ss) / Float32(H) + EPS
    )  # warp_sum broadcasts to all lanes
    for d in range(lane, H, WARP_SIZE):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        var wv = rebind[Scalar[DType.float32]](W[d])
        Y[t * H + d] = rebind[Y.ElementType](v / rms * wv)


def nll_gather_kernel[
    LT: TensorLayout
](
    L: TileTensor[DType.float32, LT, MutAnyOrigin],  # [n, vocab] logits
    TGT: TileTensor[
        DType.int32, LT, MutAnyOrigin
    ],  # [n] target token id per row
    OUT: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [n] log P(target | row) = log_softmax(L[i])[tgt]
    n_arg: Int32,
    vocab_arg: Int32,
):
    """Per-row log-probability of a target token, in ONE pass over the logits on
    the GPU — for perplexity / echo logprobs. One warp per row: lanes split the
    vocab to find the row max + Σexp(x−max) (numerically stable), then lane 0 emits
    L[i,tgt] − max − log(Σexp). Avoids ever copying the [n×vocab] logits to host.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        L: [n, vocab] logits (f32), row-major.
        TGT: [n] target token id per row (int32).
        OUT: [n] output (f32), log P(target | row) = log_softmax(L[i])[tgt].
        n: Number of rows.
        vocab: Vocabulary size (row width of L).
    """
    var n = Int(n_arg)
    var vocab = Int(vocab_arg)
    comptime assert L.flat_rank == 1
    var i = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if i >= n:
        return
    var base = i * vocab
    var m = Float32(-3.0e38)
    for v in range(lane, vocab, WARP_SIZE):
        var x = rebind[Scalar[DType.float32]](L[base + v])
        if x > m:
            m = x
    m = warp_max(m)
    var s = Float32(0.0)
    for v in range(lane, vocab, WARP_SIZE):
        s += exp(rebind[Scalar[DType.float32]](L[base + v]) - m)
    s = warp_sum(s)
    if lane == 0:
        var tgt = Int(rebind[Scalar[DType.int32]](TGT[i]))
        var lt = rebind[Scalar[DType.float32]](L[base + tgt])
        OUT[i] = rebind[OUT.ElementType](lt - m - log(s))


def rmsnorm_add_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # residual, added after norm
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    T_arg: Int32,
    H_arg: Int32,
    scale: Float32,  # final ×scale (1.0 = none)
):
    """Y = (RMSNorm(X)·W + R) · scale — fuses the rmsnorm + residual add (+ an
    optional per-layer scalar) that Gemma applies after every proj into ONE launch
    (3 of these per e2b layer). Same warp-per-row reduction as rmsnorm_kernel.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input rows [T, H] (f32).
        W: Per-channel RMSNorm weight [H] (f32).
        R: Residual [T, H] (f32), added after the norm.
        Y: Output [T, H] (f32) = (RMSNorm(X)·W + R) · scale.
        T: Number of rows (tokens).
        H: Row width (hidden size).
        scale: Final scalar multiplier (1.0 = none).
    """
    var T = Int(T_arg)
    var H = Int(H_arg)
    comptime assert X.flat_rank == 1
    var t = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if t >= T:
        return
    var ss = Float32(0.0)
    for d in range(lane, H, WARP_SIZE):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        ss += v * v
    var rms = sqrt(warp_sum(ss) / Float32(H) + EPS)
    for d in range(lane, H, WARP_SIZE):
        var v = rebind[Scalar[DType.float32]](X[t * H + d])
        var wv = rebind[Scalar[DType.float32]](W[d])
        var rv = rebind[Scalar[DType.float32]](R[t * H + d])
        Y[t * H + d] = rebind[Y.ElementType]((v / rms * wv + rv) * scale)


def matmul_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[
        DType.uint16, LT, MutAnyOrigin
    ],  # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
):
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias). One warp per output element.

    The decode path is memory-bound GEMV (M=1): the cost is streaming the
    weight matrix W from device memory. The earlier one-thread-per-output
    kernel had each thread walk a full row W[n*K + k] — so adjacent threads
    read addresses K apart, uncoalesced, wasting most of the bandwidth. Here a
    whole warp cooperates on one output: lane L reads W[n*K + L], W[n*K + L+32],
    … so the 32 lanes touch 32 consecutive words each step (coalesced), then
    `warp_sum` reduces the per-lane partials. Pure f32 accumulate, same as
    before, so greedy parity is preserved (§11 #8).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        W: Bf16 weights (raw u16 bits), row-major [N, K] (the dot is W[n,:]·X[m,:]).
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE  # one warp per output element
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
    W: TileTensor[
        DType.uint16, LT, MutAnyOrigin
    ],  # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
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
    measured ~2× the token-only kernel (~210 GFLOP/s) on the M4.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        TM: Token rows per warp output tile (token-axis reuse of W).
        CN: Output columns per warp output tile (column-axis reuse of X).

    Args:
        X: Input activations [M, K] (f32).
        W: Bf16 weights (raw u16 bits), row-major [N, K].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var ncols = ceildiv(N, CN)
    var tile = (
        Int(global_idx.x) // WARP_SIZE
    )  # one warp per (column-tile, token-tile)
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
                wv[c] = bf16_widen(
                    rebind[Scalar[DType.uint16]](W[(n0 + c) * K + k])
                )
        for mm in range(TM):
            var m = m0 + mm
            if m < M:
                var xv = rebind[Scalar[DType.float32]](X[m * K + k])
                for c in range(CN):
                    acc[mm * CN + c] += xv * wv[c]
    for mm in range(TM):
        var m = m0 + mm
        for c in range(CN):
            var total = warp_sum(
                acc[mm * CN + c]
            )  # warp collective — every lane
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
comptime _FRAG8 = 2  # 8×8 = 64 elems / 32 lanes = 2 floats per lane
comptime SG_BM = 64  # threadgroup output rows
"""Threadgroup output-tile rows (M) for the simdgroup-matrix GEMM."""
comptime SG_BN = 64  # threadgroup output cols
"""Threadgroup output-tile columns (N) for the simdgroup-matrix GEMM."""
comptime _SG_SGM = SG_BM // 2  # 32 — simdgroup subtile rows
comptime _SG_SGN = SG_BN // 2  # 32 — simdgroup subtile cols
comptime _SG_NTM = _SG_SGM // _MMA8  # 4 row-fragments per simdgroup
comptime _SG_NTN = _SG_SGN // _MMA8  # 4 col-fragments per simdgroup
comptime SG_TPB = 4 * 32  # 128 threads/block = 4 simdgroups × 32
"""Threads per block for the simdgroup-matrix GEMM (4 simdgroups × 32 lanes)."""


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
    """One 8×8×8 simdgroup-matrix multiply-accumulate: D = A·B + C (compact frag).
    """
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, _FRAG8],
    ](a, b, c)


@always_inline
def _frag_row_max(v: Float32) -> Float32:
    """Reduce a value ACROSS one fragment row (used by tensor-core attention's
    online softmax). In `_frag8_layout` the 4 lanes sharing a row differ only in
    lane bits 0 and 3, so two butterfly shuffles broadcast the row's max to all.
    """
    var r = v
    var a = shuffle_xor(r, UInt32(1))
    r = a if a > r else r
    var b = shuffle_xor(r, UInt32(8))
    r = b if b > r else r
    return r


@always_inline
def _frag_row_sum(v: Float32) -> Float32:
    """Reduce a value across one fragment row (companion to `_frag_row_max`)."""
    var r = v
    r += shuffle_xor(r, UInt32(1))
    r += shuffle_xor(r, UInt32(8))
    return r


def matmul_simd_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[
        DType.uint16, LT, MutAnyOrigin
    ],  # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
):
    """Y[M,N] = X[M,K] · W[N,K]ᵀ (+bias) on the compact 8×8 simdgroup-matrix units.
    Same signature/semantics as matmul_tiled_kernel; launch with grid_dim=
    (ceildiv(N,SG_BN), ceildiv(M,SG_BM)), block_dim=SG_TPB.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        W: Bf16 weights (raw u16 bits), row-major [N, K] (transposed operand).
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(thread_idx.x) // 32  # simdgroup id 0..3
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
        var ktail = kk + _MMA8 > K  # final K-block is partial
        # A (X) is f32 [M,K] row-major: lane's 2 frag elems are consecutive K
        # cols kk+fcol, kk+fcol+1. K bound only on the partial tail block.
        var afrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM](
            uninitialized=True
        )
        comptime for mi in range(_SG_NTM):
            var grow = row_base + mi * _MMA8 + frow
            if (interior or grow < M) and not ktail:
                afrag[mi] = xp.unsafe_offset(grow * K + kk + fcol).unsafe_load[
                    width=_FRAG8
                ]()
            else:
                var af = SIMD[DType.float32, _FRAG8](0)
                if interior or grow < M:
                    comptime for s in range(_FRAG8):
                        if kk + fcol + s < K:
                            af[s] = xp[unsafe_offset=grow * K + kk + fcol + s]
                afrag[mi] = af
        # B (W) is bf16 [N,K] (transpose_b): B[k_idx, j] = bf16(W[j, k_idx]).
        # frag slots differ in j (col); row is kk+frow (K bound only on the tail).
        var bfrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTN](
            uninitialized=True
        )
        comptime for ni in range(_SG_NTN):
            var bf = SIMD[DType.float32, _FRAG8](0)
            if not ktail or kk + frow < K:
                comptime for s in range(_FRAG8):
                    var gj = col_base + ni * _MMA8 + fcol + s
                    if interior or gj < N:
                        bf[s] = bf16_widen(wp[unsafe_offset=gj * K + kk + frow])
            bfrag[ni] = bf
        comptime for mi in range(_SG_NTM):
            comptime for ni in range(_SG_NTN):
                acc[mi * _SG_NTN + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                )

    comptime for mi in range(_SG_NTM):
        comptime for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


# ── dequant-once bf16 prefill GEMM ────────────────────────────────────────────
# Large-M prefill routes int4 weights through a one-pass dequant into a bf16
# [K,N] scratch, then a bf16-operand 8×8 simdgroup-matrix GEMM (f32 accumulate).
# Rationale (MAX linalg bake-off 2026-08-07, M4, M=1570): bf16 MMA operands +
# contiguous-N vector loads on the B side reach 3.07 TFLOP/s vs 2.15 for the
# fused-dequant f32-MMA kernel above; with the dequant pass included the
# pipeline still nets ~2.96 (+37%). The GEMM body is structurally
# `matmul_simd_kernel` (which already mirrors Modular's Apache-2.0
# `linalg/matmul/gpu/apple/matmul_8x8.mojo`); the bf16 MMA widening trick
# below is theirs too (stdlib `_mma_apple_8x8`).


@always_inline
def _mma8x8_h[
    T: DType = DType.float16
](
    a: SIMD[T, _FRAG8],
    b: SIMD[T, _FRAG8],
    c: SIMD[DType.float32, _FRAG8],
) -> SIMD[DType.float32, _FRAG8]:
    """One 8×8×8 MMA with half-precision operands (f16/bf16), f32 accumulate:
    D = A·B + C. Compact vec2 half fragments emit AIR forms that M1/M2
    mishandle; widening to 64 lanes forces the `.v64` form MSL emits, and
    air-lld collapses the sparse vector back to the per-lane fragment
    (perf-neutral — Modular's workaround, see stdlib `_mma_apple_8x8`).
    Default f16: bf16 MMA measured ~20% SLOWER on M2-generation GPUs while
    f16 == bf16 on M4 (`pixi run gemm-dtype-race`, 2026-08-08)."""
    comptime assert T == DType.float16 or T == DType.bfloat16
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


def matmul_hkn_kernel[
    LT: TensorLayout, WIDE: Bool = False, T: DType = DType.float16
](
    X: TileTensor[T, LT, MutAnyOrigin],
    W: TileTensor[T, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
):
    """Y[M,N] = X[M,K]·W[K,N] (+bias), half-precision operands (T), f32 accumulate.

    W is NON-transposed [K,N] (the dequant-scratch layout) so B-side fragment
    loads are contiguous 2-wide vector loads — the [N,K] transposed gather
    costs ~13% (bake-off). Default geometry matches `matmul_simd_kernel`
    (64×64 block, 2×2 simdgroups): grid=(ceildiv(N,SG_BN), ceildiv(M,SG_BM)),
    block_dim=SG_TPB. WIDE re-arranges the same 4 simdgroups 1×4 (32×128
    block; grid=(ceildiv(N,128), ceildiv(M,32))) — bit-identical outputs,
    and 5-16% faster when ceildiv(M,32) is ODD (half an M-tile of edge waste
    saved; neutral when even — route on that parity, small-M race 2026-08-08).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        WIDE: 32×128 block geometry instead of 64×64 (for odd ceildiv(M,32)).
        T: Operand dtype (f16 default; bf16 slow on M2-gen GPUs).

    Args:
        X: Input activations [M, K] (half, pre-cast by `cast_f32_h_kernel`).
        W: Dequantized weights [K, N] (half, from `dequant_q4_ht_kernel`).
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction dimension — MUST be a multiple of 8 (guaranteed by the
            group-128 dequant contract, K % 128 == 0).
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(thread_idx.x) // 32  # simdgroup id 0..3
    var row_base: Int
    var col_base: Int
    comptime if WIDE:
        row_base = Int(block_idx.y) * 32
        col_base = Int(block_idx.x) * 128 + sg * _SG_SGN
    else:
        row_base = Int(block_idx.y) * SG_BM + (sg // 2) * _SG_SGM
        col_base = Int(block_idx.x) * SG_BN + (sg % 2) * _SG_SGN
    # Fully-interior subtile → unguarded loads (the common case).
    var interior = (row_base + _SG_SGM <= M) and (col_base + _SG_SGN <= N)

    var xp = X.ptr
    var wp = W.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    # K is a multiple of _MMA8 by contract: this kernel only runs behind the
    # q4 dequant path, whose group-128 layout guarantees K % 128 == 0. The
    # earlier per-K-step `ktail` guard was dead weight — removing it is worth
    # ~4% (race: .scratch/gemm_tune.mojo, 3.07 -> 3.18 TFLOP/s).
    var nkt = K // _MMA8
    # The guard branch is hoisted OUT of the K-loop: interior subtiles (the
    # vast majority at prefill shapes — M=1570 has 24 full block-rows and one
    # edge strip) run a completely unguarded loop. The per-fragment
    # `interior or in-bounds` checks measured ~8% (air_gemm_race 2026-08-09:
    # interior-only 3.45 TFLOP/s vs 3.18 shipped).
    if interior:
        for ks in range(nkt):
            var kk = ks * _MMA8
            var afrag = InlineArray[SIMD[T, _FRAG8], _SG_NTM](
                uninitialized=True
            )
            comptime for mi in range(_SG_NTM):
                var grow = row_base + mi * _MMA8 + frow
                afrag[mi] = xp.unsafe_offset(grow * K + kk + fcol).unsafe_load[
                    width=_FRAG8
                ]()
            var bfrag = InlineArray[SIMD[T, _FRAG8], _SG_NTN](
                uninitialized=True
            )
            comptime for ni in range(_SG_NTN):
                var gj = col_base + ni * _MMA8 + fcol
                bfrag[ni] = wp.unsafe_offset((kk + frow) * N + gj).unsafe_load[
                    width=_FRAG8
                ]()
            comptime for mi in range(_SG_NTM):
                comptime for ni in range(_SG_NTN):
                    acc[mi * _SG_NTN + ni] = _mma8x8_h[T](
                        afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                    )
    else:
        for ks in range(nkt):
            var kk = ks * _MMA8
            # A (X) is [M,K] row-major: lane's 2 frag elems are consecutive K
            # cols kk+fcol, kk+fcol+1 (always in-bounds in K).
            var afrag = InlineArray[SIMD[T, _FRAG8], _SG_NTM](
                uninitialized=True
            )
            comptime for mi in range(_SG_NTM):
                var grow = row_base + mi * _MMA8 + frow
                if grow < M:
                    afrag[mi] = xp.unsafe_offset(
                        grow * K + kk + fcol
                    ).unsafe_load[width=_FRAG8]()
                else:
                    afrag[mi] = SIMD[T, _FRAG8](0)
            # B (W) is [K,N]: B[k_idx, j] = W[k_idx*N + j]; frag slots are
            # consecutive j → one 2-wide vector load. Col needs an N bound on
            # edge subtiles; the K row is always in-bounds.
            var bfrag = InlineArray[SIMD[T, _FRAG8], _SG_NTN](
                uninitialized=True
            )
            comptime for ni in range(_SG_NTN):
                var krow = kk + frow
                var gj = col_base + ni * _MMA8 + fcol
                if gj + 1 < N:
                    bfrag[ni] = wp.unsafe_offset(krow * N + gj).unsafe_load[
                        width=_FRAG8
                    ]()
                else:
                    var bf = SIMD[T, _FRAG8](0)
                    if gj < N:
                        bf[0] = wp[unsafe_offset=krow * N + gj]
                    bfrag[ni] = bf
            comptime for mi in range(_SG_NTM):
                comptime for ni in range(_SG_NTN):
                    acc[mi * _SG_NTN + ni] = _mma8x8_h[T](
                        afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                    )

    comptime for mi in range(_SG_NTM):
        comptime for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


def dequant_q4_ht_kernel[
    LT: TensorLayout, T: DType = DType.float16
](
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    W2: TileTensor[T, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
):
    """Dequantize group-128 int4 [N,K] into a TRANSPOSED half-precision (T)
    [K,N] scratch. Default f16: bf16 measured ~20% slower on M2-gen GPUs.

    Thread (x=n, y=kw) unpacks one packed u32 = 8 K-consecutive nibbles of
    weight row n ((nib−8)·scale, same contract as `q4_deq`); the 8 stores are
    N-strided per element but coalesce across adjacent-n threads. Launch:
    grid=(ceildiv(N,32), ceildiv(K//8,4)), block_dim=(32,4).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        W2: Output half-precision scratch (T), [K, N] row-major.
        K: Contraction dimension (multiple of 128).
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    comptime assert P.flat_rank == 1
    var n = Int(global_idx.x)
    var kw = Int(global_idx.y)
    if n >= N or kw >= K // 8:
        return
    var pp = P.ptr
    var sp = S.ptr
    var wp = W2.ptr
    var word = pp[unsafe_offset=n * (K // 8) + kw]
    var g = (kw * 8) >> Q4_SHIFT
    var s = sp[unsafe_offset=n * NG + g]
    var nibs = (SIMD[DType.uint32, 8](word) >> _Q4_SHIFTS) & 0xF
    var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]() * s
    var wf = qf.cast[T]()
    var k0 = kw * 8
    comptime for j in range(8):
        wp[unsafe_offset=(k0 + j) * N + n] = wf[j]


def cast_f32_h_kernel[
    LT: TensorLayout, T: DType = DType.float16
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[T, LT, MutAnyOrigin],
    n_arg: Int32,
):
    """Elementwise f32 → T (f16 default; round-to-nearest-even), 4 per thread.
    Launch: grid=ceildiv(ceildiv(n,4), BLOCK), block_dim=BLOCK.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Source f32 buffer [n].
        Y: Destination half-precision buffer (T) [n].
        n: Element count.
    """
    var n = Int(n_arg)
    comptime assert X.flat_rank == 1
    var i = Int(global_idx.x) * 4
    if i >= n:
        return
    var xp = X.ptr
    var yp = Y.ptr
    if i + 4 <= n:
        var v = xp.unsafe_offset(i).unsafe_load[width=4]()
        var w = v.cast[T]()
        comptime for s in range(4):
            yp[unsafe_offset=i + s] = w[s]
    else:
        for j in range(i, n):
            yp[unsafe_offset=j] = xp[unsafe_offset=j].cast[T]()


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
"""Int4 quantization group size along K (one scale per 128 weights)."""
comptime Q4_SHIFT = 7  # log2(Q4_GROUP)
"""`log2(Q4_GROUP)` — right-shift a K index to its group index."""
comptime _Q4_SHIFTS = SIMD[DType.uint32, 8](0, 4, 8, 12, 16, 20, 24, 28)
comptime _Q4_BK = 32  # K-chunk dequantized into shared per barrier
# in matmul_simd_q4_kernel (mult of 8; 64×32 fp32 = 8 KB)


@always_inline
def q4_deq[
    LT: TensorLayout
](
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    n: Int,
    k: Int,
    K: Int,
    NG: Int,
) -> Float32:
    """Dequant a single weight (n,k). Used by the prefill GEMM W-staging, where
    the matmul (not the dequant) dominates.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        P: Packed int4 weights (u32, 8 nibbles/word) for the [N, K] matrix.
        S: Per-group f32 scales [N, NG] (NG = K/Q4_GROUP).
        n: Output-channel (row) index.
        k: Contraction index within the row.
        K: Row width (contraction dim).
        NG: Number of groups per row (K / Q4_GROUP).

    Returns:
        The dequantized weight W[n,k] = (nibble − 8) · scale[n, k/Q4_GROUP].
    """
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
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Decode GEMV for int4 weights (M=1). One warp per output; the warp's lanes
    split K. Each lane consumes **eight** packed u32 per step via a single 256-bit
    load (`P.load[width=8]` = 64 nibbles), so adjacent lanes touch 256 contiguous
    bytes — the load count is 8× lower than a per-word load and the kernel goes
    from load-issue bound toward bandwidth bound (microbench: ~1.3–1.5× the
    per-word kernel across decode shapes, e.g. 11008×2048 ~54→~80 GB/s on the M4).
    The 8 nibbles of each word unpack with vector ops (a scalar unpack is ~2.5×
    slower — ALU-bound). One 256-bit oct = 64 elements = half a 128-group, so a
    single group scale per oct is exact (K, hence words=K/8, is a multiple of 16
    → words divisible by 8, no tail). ~2× the bf16 GEMV on the M4.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
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
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var acc = Float32(0.0)
    var octs = words // 8  # 8 packed u32 = 64 weights per lane/step
    for q in range(lane, octs, WARP_SIZE):
        var word8 = pp.unsafe_offset(rowword + q * 8).unsafe_load[width=8]()
        var k0 = q * 64
        var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
        comptime for j in range(8):
            var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
            var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
            var xv = xp.unsafe_offset(xbase + k0 + j * 8).unsafe_load[width=8]()
            acc += (qf * xv).reduce_add() * s
    var total = warp_sum(acc)
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_q4_rw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Decode GEMV (M=1): R rows × W K-slice warps per threadgroup. Same inner
    loop as `matmul_q4_kernel`, but each row's K sweep is split across W warps
    (partials folded in threadgroup memory), multiplying warps in flight per
    row. The shipping one-warp-per-output kernel is load-ISSUE limited on
    short-K/large-N shapes — a warp covers K=2048 in ONE 256-bit load per lane
    (2048×2048 measures 29 GB/s while 2048×11008 reaches 80 with the same
    loop, purely from having more warps). Measured (M4, `pixi run
    q4-gemv-mr`): R=1 W=4 → 92.9 GB/s on 2048×11008 (1.79×, 77% of ceiling);
    R=2 W=4 → 1.37× on 2048×2048. Long-K shapes (e.g. 11008×2048) stay on the
    shipping kernel — it already saturates there (mm_w routes by shape). Row
    guards are warp-uniform (no per-lane divergence around SIMD ops — see the
    divergent-branch miscompile note on the MMA kernels).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Rows per threadgroup.
        W: K-slice warps per row.

    Args:
        X: Input activations [1, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [1, N] (f32).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var wid = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var r = wid % R
    var w = wid // R
    var n = n0 + r
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    if n < N:  # warp-uniform (n0, r are per-warp constants)
        for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
            var k0 = q * 64
            var word8 = pp.unsafe_offset(n * words + q * 8).unsafe_load[
                width=8
            ]()
            var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
            var racc = Float32(0.0)
            comptime for j in range(8):
                var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
                var xv = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
                racc += (qf * xv).reduce_add()
            acc += racc * rebind[Float32](s)
    var p = warp_sum(acc)
    if lane == 0:
        part[unsafe_offset=r * W + w] = p
    barrier()
    if wid == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


comptime SPEC_MAX_M = 8  # small-M int4 path cap. Above this the flat 64-row
# simdgroup GEMM wins; mm_w routes M>SPEC_MAX_M there.
"""Max M routed to the small-M int4 paths; above it the 64-row simd GEMM wins."""
comptime SPEC_SMALL_MIN = 5  # M in [SPEC_SMALL_MIN, SPEC_MAX_M] uses the 1-tile
# MMA GEMM (flat); M in [2, SPEC_SMALL_MIN) uses the
# batched GEMV (cheaper at the very smallest batches).
"""M threshold: ≥ this uses the 1-tile MMA GEMM, below it the batched GEMV."""


def matmul_q4_batch_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Batched int4 GEMV for small M (2..SPEC_MAX_M, e.g. speculative verify):
    one warp per output COLUMN n. The warp loads each 256-bit weight oct ONCE —
    the bandwidth cost — and accumulates all M activation rows in registers, so
    weight traffic ≈ the M=1 GEMV regardless of M (x rows are small and re-read
    per m). This keeps small-M verify forwards far below the simdgroup-GEMM's
    flat ~700 ms. Weight unpack (the ALU-heavy nibble→f32) is done once per oct
    and reused across all M rows; same exact-scale invariant (one group scale per
    64-element oct, since 8|128).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (2..SPEC_MAX_M).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if n >= N:
        return
    var words = K // 8
    var rowword = n * words
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var acc = InlineArray[Float32, SPEC_MAX_M](fill=0.0)
    var octs = words // 8  # 8 packed u32 = 64 weights per lane/step
    for q in range(lane, octs, WARP_SIZE):
        var word8 = pp.unsafe_offset(rowword + q * 8).unsafe_load[width=8]()
        var k0 = q * 64
        var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
        comptime for j in range(8):
            var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
            var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
            for m in range(M):
                var xv = xp.unsafe_offset(m * K + k0 + j * 8).unsafe_load[
                    width=8
                ]()
                acc[m] += (qf * xv).reduce_add() * s
    for m in range(M):
        var total = warp_sum(acc[m])
        if lane == 0:
            if use_bias != 0:
                total += rebind[Scalar[DType.float32]](B[n])
            Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_norm_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    LNW: TileTensor[DType.float32, LT, MutAnyOrigin],  # RMSNorm weight [K]
    W: TileTensor[
        DType.uint16, LT, MutAnyOrigin
    ],  # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
):
    """Bf16 decode GEMV with RMSNorm fused into the input row: each warp already
    streams the full input row x[k] for its dot, so it accumulates Σx[k]² in the
    same K-loop — `out = (Σ x[k]·lnw[k]·W[n,k]) / rms`, rms = √(mean(x²)+EPS) — and
    the separate RMSNorm launch (and its x round-trip) disappears (decode only).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32), un-normalized.
        LNW: RMSNorm weight [K] (f32), applied to X inside the dot.
        W: Bf16 weights (raw u16 bits), row-major [N, K].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var acc = Float32(0.0)
    var ss = Float32(0.0)
    for k in range(lane, K, WARP_SIZE):
        var xv = rebind[Scalar[DType.float32]](X[m * K + k])
        ss += xv * xv
        var lw = rebind[Scalar[DType.float32]](LNW[k])
        var wv = bf16_widen(rebind[Scalar[DType.uint16]](W[n * K + k]))
        acc += (xv * lw) * wv
    var rms = sqrt(warp_sum(ss) / Float32(K) + EPS)
    var total = warp_sum(acc) / rms
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_q4_norm_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    LNW: TileTensor[DType.float32, LT, MutAnyOrigin],  # RMSNorm weight [K]
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Int4 decode GEMV with RMSNorm fused in (see matmul_norm_kernel). Folds the
    pre-projection RMSNorm into the qkv / gate_up GEMVs — −2 launches per layer.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32), un-normalized.
        LNW: RMSNorm weight [K] (f32), applied to X inside the dot.
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
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
    var pp = P.ptr
    var xp = X.ptr
    var lwp = LNW.ptr
    var sp = S.ptr
    var acc = Float32(0.0)
    var ss = Float32(0.0)
    var octs = words // 8  # 256-bit weight loads, see matmul_q4_kernel
    for q in range(lane, octs, WARP_SIZE):
        var word8 = pp.unsafe_offset(rowword + q * 8).unsafe_load[width=8]()
        var k0 = q * 64
        var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
        comptime for j in range(8):
            var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
            var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
            var xv = xp.unsafe_offset(xbase + k0 + j * 8).unsafe_load[width=8]()
            var lw = lwp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
            ss += (xv * xv).reduce_add()
            acc += (qf * (xv * lw)).reduce_add() * s
    var rms = sqrt(warp_sum(ss) / Float32(K) + EPS)
    var total = warp_sum(acc) / rms
    if lane == 0:
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_q4_norm_rw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    LNW: TileTensor[DType.float32, LT, MutAnyOrigin],  # RMSNorm weight [K]
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """`matmul_q4_norm_kernel` in the K-split shape (M=1): R rows × W K-slice
    warps per threadgroup (see matmul_q4_rw_kernel for why — decode's qkv and
    the merged gate_up run through the norm-fused GEMV, and gate_up's
    2048→2·11008 is exactly the shape where K-splitting measured 1.79×).
    The RMS sum-of-squares is row-independent, so each K-slice's partial ss is
    folded across warps alongside the R per-row dot partials; warps with r>0
    compute the same slice ss and simply don't store it.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Rows per threadgroup.
        W: K-slice warps per row.

    Args:
        X: Input activations [1, K] (f32), un-normalized.
        LNW: RMSNorm weight [K] (f32), applied to X inside the dot.
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [1, N] (f32).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var wid = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var r = wid % R
    var w = wid // R
    var n = n0 + r
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var lwp = LNW.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var ssp = stack_allocation[W, Float32, address_space=AddressSpace.SHARED]()
    var acc = Float32(0.0)
    var ss = Float32(0.0)
    if n < N:  # warp-uniform (n0, r are per-warp constants)
        for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
            var word8 = pp.unsafe_offset(n * words + q * 8).unsafe_load[
                width=8
            ]()
            var k0 = q * 64
            var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
            var racc = Float32(0.0)
            comptime for j in range(8):
                var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
                var xv = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
                var lw = lwp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
                ss += (xv * xv).reduce_add()
                racc += (qf * (xv * lw)).reduce_add()
            acc += racc * rebind[Float32](s)
    var pa = warp_sum(acc)
    var ps = warp_sum(ss)
    if lane == 0:
        part[unsafe_offset=r * W + w] = pa
        if r == 0:
            ssp[unsafe_offset=w] = ps
    barrier()
    if wid == 0 and lane < R and n0 + lane < N:
        var sstot = Float32(0.0)
        comptime for i in range(W):
            sstot += ssp[unsafe_offset=i]
        var rms = sqrt(sstot / Float32(K) + EPS)
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        total /= rms
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def matmul_q4_resid_rw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    RS: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # residual added in the epilogue
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """`matmul_q4_resid_kernel` in the K-split shape (M=1) — the o-proj is a
    square 2048×2048, the shape class where one-warp-per-output is pure
    latency (measured 1.37× from K-splitting). Same body as
    matmul_q4_rw_kernel plus the residual add in the store.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Rows per threadgroup.
        W: K-slice warps per row.

    Args:
        X: Input activations [1, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        RS: Residual [1, N] (f32), added in the epilogue.
        Y: Output [1, N] (f32) = X·Wᵀ (+bias) + RS.
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var wid = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var r = wid % R
    var w = wid // R
    var n = n0 + r
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    if n < N:  # warp-uniform
        for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
            var word8 = pp.unsafe_offset(n * words + q * 8).unsafe_load[
                width=8
            ]()
            var k0 = q * 64
            var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
            var racc = Float32(0.0)
            comptime for j in range(8):
                var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
                var xv = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
                racc += (qf * xv).reduce_add()
            acc += racc * rebind[Float32](s)
    var p = warp_sum(acc)
    if lane == 0:
        part[unsafe_offset=r * W + w] = p
    barrier()
    if wid == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        total += rebind[Scalar[DType.float32]](RS[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def matmul_q4_resid_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # residual added in the epilogue
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Matmul_q4_kernel with a fused residual add (Y = X·Wᵀ(+bias) + R). Folds the
    decode residual into the proj GEMV, saving one launch per layer (×2: o & down).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        R: Residual [M, N] (f32), added in the epilogue.
        Y: Output [M, N] (f32) = X·Wᵀ (+bias) + R.
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
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
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var acc = Float32(0.0)
    var octs = words // 8  # 256-bit weight loads, see matmul_q4_kernel
    for q in range(lane, octs, WARP_SIZE):
        var word8 = pp.unsafe_offset(rowword + q * 8).unsafe_load[width=8]()
        var k0 = q * 64
        var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
        comptime for j in range(8):
            var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
            var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
            var xv = xp.unsafe_offset(xbase + k0 + j * 8).unsafe_load[width=8]()
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
    W: TileTensor[
        DType.uint16, LT, MutAnyOrigin
    ],  # bf16 weights (raw u16 bits)
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # residual added in the epilogue
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    use_bias_arg: Int32,
):
    """Matmul_kernel (bf16 decode GEMV) with a fused residual add.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        W: Bf16 weights (raw u16 bits), row-major [N, K].
        B: Bias [N] (f32), added when use_bias != 0.
        R: Residual [M, N] (f32), added in the epilogue.
        Y: Output [M, N] (f32) = X·Wᵀ (+bias) + R.
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var use_bias = Int(use_bias_arg)
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


def matmul_q4_silu_resid_kernel[
    LT: TensorLayout
](
    GU: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [M, 2*K]: row = gate(K) ++ up(K)
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    R: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
):
    """Int4 down-proj decode GEMV with SwiGLU fused on the input: reads the fused
    gate|up GEMV output and forms act[k]=silu(gate[k])·up[k] on load, so the
    separate silu_mul_cat launch (and its `act` buffer) disappears. K = inter, so
    up[k] is GU[k+K]. Residual fused in the epilogue (down's resid).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        GU: Fused gate|up GEMV output [M, 2*K] (f32); row = gate(K) ++ up(K).
            act[k] = silu(gate[k]) · up[k] is formed on load.
        P: Packed int4 down-proj weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        R: Residual [M, N] (f32), added in the epilogue.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Intermediate size (gate/up width = contraction dim).
        N: Number of output (hidden) channels.
        NG: Groups per row (K / Q4_GROUP).
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    comptime assert GU.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var words = K // 8
    var rowword = n * words
    var gbase = m * 2 * K
    var pp = P.ptr
    var gp = GU.ptr
    var sp = S.ptr
    var acc = Float32(0.0)
    var octs = words // 8  # 256-bit weight loads, see matmul_q4_kernel
    for q in range(lane, octs, WARP_SIZE):
        var word8 = pp.unsafe_offset(rowword + q * 8).unsafe_load[width=8]()
        var k0 = q * 64
        var s = sp[unsafe_offset=n * NG + (k0 >> Q4_SHIFT)]
        comptime for j in range(8):
            var nibs = (SIMD[DType.uint32, 8](word8[j]) >> _Q4_SHIFTS) & 0xF
            var qf = (nibs.cast[DType.int32]() - 8).cast[DType.float32]()
            var g = gp.unsafe_offset(gbase + k0 + j * 8).unsafe_load[width=8]()
            var u = gp.unsafe_offset(gbase + k0 + j * 8 + K).unsafe_load[
                width=8
            ]()
            var xv = (g / (1.0 + exp(-g))) * u
            acc += (qf * xv).reduce_add() * s
    var total = warp_sum(acc)
    if lane == 0:
        total += rebind[Scalar[DType.float32]](R[m * N + n])
        Y[m * N + n] = rebind[Y.ElementType](total)


def matmul_silu_resid_kernel[
    LT: TensorLayout
](
    GU: TileTensor[DType.float32, LT, MutAnyOrigin],  # [M, 2*K]: gate ++ up
    W: TileTensor[DType.uint16, LT, MutAnyOrigin],  # bf16 weights
    R: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
):
    """Bf16 down-proj decode GEMV with SwiGLU fused on the input (see q4 variant).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        GU: Fused gate|up GEMV output [M, 2*K] (f32); row = gate(K) ++ up(K).
            act[k] = silu(gate[k]) · up[k] is formed on load.
        W: Bf16 down-proj weights (raw u16 bits), row-major [N, K].
        R: Residual [M, N] (f32), added in the epilogue.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Intermediate size (gate/up width = contraction dim).
        N: Number of output (hidden) channels.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    comptime assert GU.flat_rank == 1
    var out = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    if out >= M * N:
        return
    var m = out // N
    var n = out % N
    var gbase = m * 2 * K
    var acc = Float32(0.0)
    for k in range(lane, K, WARP_SIZE):
        var g = rebind[Scalar[DType.float32]](GU[gbase + k])
        var u = rebind[Scalar[DType.float32]](GU[gbase + k + K])
        var act = (g / (1.0 + exp(-g))) * u
        var wv = bf16_widen(rebind[Scalar[DType.uint16]](W[n * K + k]))
        acc += act * wv
    var total = warp_sum(acc)
    if lane == 0:
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
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Int4 scalar prefill fallback — matmul_tiled_kernel with q4_deq W-reads.
    Used only if the simdgroup-matrix probe fails.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        TM: Token rows per warp output tile (token-axis reuse of W).
        CN: Output columns per warp output tile (column-axis reuse of X).

    Args:
        X: Input activations [M, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
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
    LT: TensorLayout, WIDE: Bool = False
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Int4 prefill GEMM on the compact 8×8 simdgroup-matrix units, with the
    weight tile **dequantized into threadgroup shared memory once per block**
    (MLX's `QuantizedBlockLoader` pattern). The earlier version filled each B
    fragment from `q4_deq` *every* K-step from global — unpacking+scaling in the
    hot loop and re-dequantizing shared columns once per simdgroup — and ran at
    ~1.0 TFLOP/s (~2.1× slower than the bf16 GEMM). Here all 128 threads
    cooperatively unpack a 64×`_Q4_BK` tile of W (one packed u32 = 8 nibbles per
    thread, group scale folded once), `barrier()`, then every simdgroup runs its
    MMAs reading fp32 from shared. ~2.2× the global-load kernel — **on par with
    bf16 (~2.1 TFLOP/s)**, since the packed int4 moved into shared is 4× smaller
    and the dequant is amortized. Only W is staged; X stays on the cache-served
    global path (bf16 X-staging measured negative on M4). Matmul math/output are
    byte-for-byte the bf16 kernel's.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (tokens).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    # WIDE: the same 4 simdgroups arranged 1×4 (32×128 block, grid=
    # (ceildiv(N,128), ceildiv(M,32))) instead of 2×2 (64×64). Bit-identical;
    # 5-15% faster when ceildiv(M,32) is ODD (small-M race 2026-08-08) —
    # callers route on that parity.
    comptime _BN = 128 if WIDE else SG_BN
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = tid // 32
    var blk_row: Int
    var blk_col: Int
    var row_base: Int
    var col_base: Int
    comptime if WIDE:
        blk_row = Int(block_idx.y) * 32
        blk_col = Int(block_idx.x) * _BN
        row_base = blk_row
        col_base = blk_col + sg * _SG_SGN
    else:
        blk_row = Int(block_idx.y) * SG_BM
        blk_col = Int(block_idx.x) * _BN
        row_base = blk_row + (sg // 2) * _SG_SGM
        col_base = blk_col + (sg % 2) * _SG_SGN
    var col_off = col_base - blk_col  # this simdgroup's column base in the tile

    var Bs = stack_allocation[
        _Q4_BK * _BN, Float32, address_space=AddressSpace.SHARED
    ]()

    var xp = X.ptr
    var pp = P.ptr
    var sp = S.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM * _SG_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    var kc = 0
    while kc < K:
        # Cooperative WORD-vectorized dequant of W[blk_col..+_BN, kc..+_Q4_BK] → Bs
        # (row-major [k_local][j_local]). Each thread takes one packed u32 = 8
        # nibbles of an 8-aligned k-run of one N column, unpacks all 8 + folds the
        # group scale once (8|128 so an 8-run never straddles a group).
        comptime _NW = _BN * (_Q4_BK // 8)
        for w in range(tid, _NW, SG_TPB):
            var j_local = w % _BN
            var krun = (w // _BN) * 8
            var gj = blk_col + j_local
            var gk0 = kc + krun
            if gj < N and gk0 < K:
                var word = pp[unsafe_offset=(gj * K + gk0) >> 3]
                var scale = sp[unsafe_offset=gj * NG + (gk0 >> Q4_SHIFT)]
                var nibs = (SIMD[DType.uint32, 8](word) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[
                    DType.float32
                ]() * scale
                comptime for t in range(8):
                    Bs[unsafe_offset=(krun + t) * _BN + j_local] = (
                        qf[t] if gk0 + t < K else 0.0
                    )
            else:
                comptime for t in range(8):
                    Bs[unsafe_offset=(krun + t) * _BN + j_local] = 0.0
        barrier()

        comptime _KS = _Q4_BK // _MMA8
        for kss in range(_KS):
            var kk = kc + kss * _MMA8
            if kk >= K:
                continue
            var ktail = kk + _MMA8 > K
            var afrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTM](
                uninitialized=True
            )
            comptime for mi in range(_SG_NTM):
                var grow = row_base + mi * _MMA8 + frow
                if grow < M and not ktail:
                    afrag[mi] = xp.unsafe_offset(
                        grow * K + kk + fcol
                    ).unsafe_load[width=_FRAG8]()
                else:
                    var af = SIMD[DType.float32, _FRAG8](0)
                    if grow < M:
                        comptime for s in range(_FRAG8):
                            if kk + fcol + s < K:
                                af[s] = xp[
                                    unsafe_offset=grow * K + kk + fcol + s
                                ]
                    afrag[mi] = af
            # B (W) read from shared: Bs[(kk_local+frow)][col_off + ni*8 + fcol]
            var bfrag = InlineArray[SIMD[DType.float32, _FRAG8], _SG_NTN](
                uninitialized=True
            )
            comptime for ni in range(_SG_NTN):
                var brow = (
                    (kss * _MMA8 + frow) * _BN + col_off + ni * _MMA8 + fcol
                )
                bfrag[ni] = Bs.unsafe_offset(brow).unsafe_load[width=_FRAG8]()
            comptime for mi in range(_SG_NTM):
                comptime for ni in range(_SG_NTN):
                    acc[mi * _SG_NTN + ni] = _mma8x8(
                        afrag[mi], bfrag[ni], acc[mi * _SG_NTN + ni]
                    )
        barrier()
        kc += _Q4_BK

    comptime for mi in range(_SG_NTM):
        comptime for ni in range(_SG_NTN):
            var frag = acc[mi * _SG_NTN + ni]
            comptime for s in range(_FRAG8):
                var grow = row_base + mi * _MMA8 + frow
                var gcol = col_base + ni * _MMA8 + fcol + s
                if grow < M and gcol < N:
                    var v = frag[s]
                    if use_bias != 0:
                        v += rebind[Scalar[DType.float32]](B[gcol])
                    Y[grow * N + gcol] = rebind[Y.ElementType](v)


# Small-M (≤8) int4 GEMM: ONE 8-row MMA tile, so a Q≤8 speculative-verify forward
# wastes ≤3 padding rows instead of the 64-row simd GEMM's 56–59 (it computes a
# full 64-row tile regardless of M → compute-bound on wasted rows at small M). Same
# coalesced shared-W dequant + 8×8 MMA math as matmul_simd_q4_kernel, just with the
# M dimension collapsed to a single fragment shared across all the col simdgroups.
comptime _SM_BN = 64  # output cols per block
comptime _SM_NSG = 4  # simdgroups/block (col-tiled, 1 row tile)
comptime _SM_SGN = _SM_BN // _SM_NSG  # 16 cols per simdgroup
comptime _SM_NTN = _SM_SGN // _MMA8  # 2 col-fragments per simdgroup
comptime _SM_TPB = _SM_NSG * 32  # 128 threads/block
comptime _SM_BK = 32  # K-chunk staged to shared (32×64 fp32
# = 8 KB; 4 words/col ⇒ 4-wide coalesce)


def matmul_q4_small_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    M_arg: Int32,
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Int4 GEMM for M ≤ 8 (speculative verify). grid=(ceildiv(N,_SM_BN),1),
    block_dim=_SM_TPB. One 8-row MMA tile (row_base=0) is shared across the block's
    `_SM_NSG` col-simdgroups; W[blk_col..+64, kc..+_SM_BK] is cooperatively
    dequantized into shared once per K-chunk and each simdgroup MMAs its 2 col-
    fragments from it. X stays on the global cache path. Output/scale invariants
    match matmul_simd_q4_kernel.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input activations [M, K] (f32), M ≤ 8.
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [M, N] (f32).
        M: Number of input rows (≤ 8).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var M = Int(M_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = tid // 32
    var blk_col = Int(block_idx.x) * _SM_BN
    var col_base = blk_col + sg * _SM_SGN

    var Bs = stack_allocation[
        _SM_BK * _SM_BN, Float32, address_space=AddressSpace.SHARED
    ]()
    var xp = X.ptr
    var pp = P.ptr
    var sp = S.ptr
    var acc = InlineArray[SIMD[DType.float32, _FRAG8], _SM_NTN](
        fill=SIMD[DType.float32, _FRAG8](0)
    )

    var kc = 0
    while kc < K:
        # Column-major thread→word map (consecutive threads = consecutive columns
        # at the same k-run): writes to Bs are bank-conflict-free (consecutive
        # shared slots). A K-major map would coalesce the global W reads but strides
        # the Bs writes by SG_BN → shared bank conflicts that measured net-slower.
        comptime _NW = _SM_BN * (_SM_BK // 8)
        for w in range(tid, _NW, _SM_TPB):
            var j_local = w % _SM_BN
            var krun = (w // _SM_BN) * 8
            var gj = blk_col + j_local
            var gk0 = kc + krun
            if gj < N and gk0 < K:
                var word = pp[unsafe_offset=(gj * K + gk0) >> 3]
                var scale = sp[unsafe_offset=gj * NG + (gk0 >> Q4_SHIFT)]
                var nibs = (SIMD[DType.uint32, 8](word) >> _Q4_SHIFTS) & 0xF
                var qf = (nibs.cast[DType.int32]() - 8).cast[
                    DType.float32
                ]() * scale
                comptime for t in range(8):
                    Bs[unsafe_offset=(krun + t) * _SM_BN + j_local] = (
                        qf[t] if gk0 + t < K else 0.0
                    )
            else:
                comptime for t in range(8):
                    Bs[unsafe_offset=(krun + t) * _SM_BN + j_local] = 0.0
        barrier()

        comptime _KS = _SM_BK // _MMA8
        for kss in range(_KS):
            var kk = kc + kss * _MMA8
            if kk >= K:
                continue
            var ktail = kk + _MMA8 > K
            # A (X) fragment — single 8-row tile (rows 0..7 = frow), cols kk+fcol,+1.
            var grow = frow
            var afrag = SIMD[DType.float32, _FRAG8](0)
            if grow < M and not ktail:
                afrag = xp.unsafe_offset(grow * K + kk + fcol).unsafe_load[
                    width=_FRAG8
                ]()
            elif grow < M:
                comptime for s in range(_FRAG8):
                    if kk + fcol + s < K:
                        afrag[s] = xp[unsafe_offset=grow * K + kk + fcol + s]
            comptime for ni in range(_SM_NTN):
                var brow = (
                    (kss * _MMA8 + frow) * _SM_BN
                    + sg * _SM_SGN
                    + ni * _MMA8
                    + fcol
                )
                var bfrag = Bs.unsafe_offset(brow).unsafe_load[width=_FRAG8]()
                acc[ni] = _mma8x8(afrag, bfrag, acc[ni])
        barrier()
        kc += _SM_BK

    comptime for ni in range(_SM_NTN):
        var frag = acc[ni]
        comptime for s in range(_FRAG8):
            var grow = frow
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
    n_arg: Int32,
):
    """SwiGLU on two separate buffers: Y[i] = silu(A[i]) · B[i] over `n` elements.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        A: Gate buffer (f32, length `n`).
        B: Multiplier buffer (f32, length `n`).
        Y: Output buffer (f32, length `n`) = silu(A) · B.
        n: Element count.
    """
    var n = Int(n_arg)
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
    GU: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [T, 2*inter]: row = gate(inter) ++ up(inter)
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],  # [T, inter]
    T_arg: Int32,
    inter_arg: Int32,
):
    """SwiGLU activation reading the *fused* gate+up GEMV output (one buffer, gate
    then up per row): Y = silu(gate)·up. Lets gate+up be one GEMV instead of two —
    the split happens here for free instead of via separate buffers.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        GU: Fused gate+up GEMV output [T, 2*inter] (f32); row = gate(inter) ++ up(inter).
        Y: Output [T, inter] (f32) = silu(gate) · up.
        T: Number of rows (tokens).
        inter: Intermediate size (gate/up width).
    """
    var T = Int(T_arg)
    var inter = Int(inter_arg)
    comptime assert GU.flat_rank == 1
    var idx = global_idx.x
    if idx >= T * inter:
        return
    var t = idx // inter
    var i = idx % inter
    var g = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i])
    var u = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i + inter])
    Y[idx] = rebind[Y.ElementType]((g / (1.0 + exp(-g))) * u)


comptime _GELU_C = Float32(
    0.7978845608028654
)  # √(2/π), for the tanh GELU approx


def gelu_mul_cat_kernel[
    LT: TensorLayout
](
    GU: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [T, 2*inter]: gate(inter) ++ up(inter)
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],  # [T, inter]
    T_arg: Int32,
    inter_arg: Int32,
):
    """GeGLU activation (Gemma): Y = gelu_tanh(gate)·up, reading the fused gate+up
    GEMV output — the GELU sibling of silu_mul_cat_kernel. `gelu_pytorch_tanh`:
    0.5·g·(1 + tanh(√(2/π)·(g + 0.044715·g³))).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        GU: Fused gate+up GEMV output [T, 2*inter] (f32); row = gate(inter) ++ up(inter).
        Y: Output [T, inter] (f32) = gelu_tanh(gate) · up.
        T: Number of rows (tokens).
        inter: Intermediate size (gate/up width).
    """
    var T = Int(T_arg)
    var inter = Int(inter_arg)
    comptime assert GU.flat_rank == 1
    var idx = global_idx.x
    if idx >= T * inter:
        return
    var t = idx // inter
    var i = idx % inter
    var g = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i])
    var u = rebind[Scalar[DType.float32]](GU[t * 2 * inter + i + inter])
    var gelu = 0.5 * g * (1.0 + tanh(_GELU_C * (g + 0.044715 * g * g * g)))
    Y[idx] = rebind[Y.ElementType](gelu * u)


def gelu_mul_kernel[
    LT: TensorLayout
](
    A: TileTensor[DType.float32, LT, MutAnyOrigin],  # [n] gate
    B: TileTensor[DType.float32, LT, MutAnyOrigin],  # [n] multiplier
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],  # [n] out
    n_arg: Int32,
):
    """Y = gelu_tanh(A)·B over two SEPARATE [n] buffers (Gemma3n per-layer-input
    gate: gelu(gate(h)) ⊙ per_layer_input). Sibling of gelu_mul_cat_kernel, which
    reads one fused gate++up buffer instead.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        A: Gate buffer [n] (f32).
        B: Multiplier buffer [n] (f32).
        Y: Output [n] (f32) = gelu_tanh(A) · B.
        n: Element count.
    """
    var n = Int(n_arg)
    comptime assert A.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var g = rebind[Scalar[DType.float32]](A[i])
    var u = rebind[Scalar[DType.float32]](B[i])
    var gelu = 0.5 * g * (1.0 + tanh(_GELU_C * (g + 0.044715 * g * g * g)))
    Y[i] = rebind[Y.ElementType](gelu * u)


def gelu_mul_strided_kernel[
    LT: TensorLayout
](
    A: TileTensor[DType.float32, LT, MutAnyOrigin],  # [T, n] gate
    P: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [T, stride] (per-layer-input table)
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],  # [T, n] out
    T_arg: Int32,
    n_arg: Int32,
    stride_arg: Int32,
    off_arg: Int32,
):
    """Y[t,j] = gelu_tanh(A[t,j]) · P[t, off+j] — the Gemma3n PLE gate fused with the
    strided slice of the per-layer-input table (copy_strided + gelu_mul → 1 launch).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        A: Gate [T, n] (f32).
        P: Per-layer-input table [T, stride] (f32).
        Y: Output [T, n] (f32) = gelu_tanh(A[t,j]) · P[t, off+j].
        T: Number of rows (tokens).
        n: Slice width per row.
        stride: Row stride of P.
        off: Column offset into P's row.
    """
    var T = Int(T_arg)
    var n = Int(n_arg)
    var stride = Int(stride_arg)
    var off = Int(off_arg)
    comptime assert A.flat_rank == 1
    var i = global_idx.x
    if i >= T * n:
        return
    var t = i // n
    var j = i % n
    var g = rebind[Scalar[DType.float32]](A[i])
    var u = rebind[Scalar[DType.float32]](P[t * stride + off + j])
    var gelu = 0.5 * g * (1.0 + tanh(_GELU_C * (g + 0.044715 * g * g * g)))
    Y[i] = rebind[Y.ElementType](gelu * u)


def softcap_kernel[
    LT: TensorLayout
](
    X: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # in-place: X ← cap·tanh(X/cap)
    n_arg: Int32,
    cap: Float32,
):
    """Logit soft-capping (Gemma): X ← cap·tanh(X/cap), in place. Used for the
    final-logit softcap (cap=30) and reusable for the attention-logit softcap.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: In/out buffer (f32, length `n`); overwritten with cap·tanh(X/cap).
        n: Element count.
        cap: Soft-cap magnitude.
    """
    var n = Int(n_arg)
    comptime assert X.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    var v = rebind[Scalar[DType.float32]](X[i])
    X[i] = rebind[X.ElementType](cap * tanh(v / cap))


def add_scalar_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],  # in-place: X ← X + c
    n_arg: Int32,
    c: Float32,
):
    """In-place add a scalar to every element. Gemma bakes (1+w) into every
    RMSNorm weight at load (c=1.0) so the existing `x/rms*w` kernels are exact for
    Gemma's (1+w) RMSNorm without touching the hot norm kernels.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: In/out buffer (f32, length `n`); each element becomes X[i] + c.
        n: Element count.
        c: Scalar to add.
    """
    var n = Int(n_arg)
    comptime assert X.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    X[i] = rebind[X.ElementType](rebind[Scalar[DType.float32]](X[i]) + c)


def mul_scalar_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    n_arg: Int32,
    c: Float32,
):
    """Y = X * c (elementwise). Gemma scales embeddings by √hidden (input path) and
    applies the per-layer learned scalar to the layer output.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Input buffer (f32, length `n`).
        Y: Output buffer (f32, length `n`) = X · c.
        n: Element count.
        c: Scalar multiplier.
    """
    var n = Int(n_arg)
    comptime assert X.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    Y[i] = rebind[Y.ElementType](rebind[Scalar[DType.float32]](X[i]) * c)


def vnorm_kernel[
    LT: TensorLayout, HKV: Int, HEAD_DIM: Int, KV_DT: DType = DType.float32
](
    In: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # V source (row = in_stride, V at in_off)
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],  # [max, HKV, HEAD_DIM] V cache
    Tq_arg: Int32,
    q_offset_arg: Int32,
    in_stride_arg: Int32,
    in_off_arg: Int32,
):
    """Gemma per-head SCALE-FREE RMSNorm over V (v_norm has no weight): one thread
    per (token, kv-head) normalizes that head's HEAD_DIM values by 1/sqrt(mean(x²)+
    eps) and writes them into the V cache at the absolute-position row (like
    rope_k writes K). Used by Gemma's full-attention layers where V = v_norm(k_proj).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.

    Args:
        In: V source (f32); each row has stride in_stride, the V slice at column in_off.
        Vc: V cache [max, HKV, HEAD_DIM] (f32); normalized V written at row q_offset+t.
        Tq: Number of tokens this launch.
        q_offset: Absolute position of the first token (cache row base).
        in_stride: Source row stride.
        in_off: Column offset of V within the source row.
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var in_stride = Int(in_stride_arg)
    var in_off = Int(in_off_arg)
    comptime assert In.flat_rank == 1
    var nkv = HKV * HEAD_DIM
    var idx = Int(global_idx.x)
    if idx >= Tq * HKV:
        return
    var t = idx // HKV
    var kvh = idx % HKV
    var inbase = t * in_stride + in_off + kvh * HEAD_DIM
    var outbase = (q_offset + t) * nkv + kvh * HEAD_DIM
    var rrms = _head_rrms(In, inbase, HEAD_DIM)
    for d in range(HEAD_DIM):
        var v = rebind[Scalar[DType.float32]](In[inbase + d])
        Vc[outbase + d] = rebind[Vc.ElementType]((v * rrms).cast[KV_DT]())


def copy_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst_offset_arg: Int32,
    n_arg: Int32,
):
    """Copy `n` contiguous f32 elements from `src` into `dst` at `dst_offset`.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        src: Source buffer (f32); `n` contiguous elements from index 0.
        dst: Destination buffer (f32); written starting at `dst_offset`.
        dst_offset: Start index in `dst`.
        n: Element count.
    """
    var dst_offset = Int(dst_offset_arg)
    var n = Int(n_arg)
    comptime assert dst.flat_rank == 1
    var i = global_idx.x
    if i >= n:
        return
    dst[dst_offset + i] = rebind[dst.ElementType](src[i])


def copy_strided_kernel[
    LT: TensorLayout
](
    src: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # row = in_stride, slice at in_off
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    T_arg: Int32,
    in_stride_arg: Int32,
    in_off_arg: Int32,
    dst_off_arg: Int32,
    n_arg: Int32,  # slice width per row
):
    """Copy a [T, n] column-slice out of a strided source (e.g. the V part of a
    fused [q|k|v] buffer) into dst[dst_off:] contiguously — V into its cache rows.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        src: Strided source (f32); each row has stride in_stride, slice at column in_off.
        dst: Destination buffer (f32); the slice is written contiguously at dst_off.
        T: Number of rows.
        in_stride: Source row stride.
        in_off: Column offset of the slice within the source row.
        dst_off: Start index in `dst`.
        n: Slice width per row.
    """
    var T = Int(T_arg)
    var in_stride = Int(in_stride_arg)
    var in_off = Int(in_off_arg)
    var dst_off = Int(dst_off_arg)
    var n = Int(n_arg)
    comptime assert dst.flat_rank == 1
    var idx = global_idx.x
    if idx >= T * n:
        return
    var t = idx // n
    var j = idx % n
    dst[dst_off + t * n + j] = rebind[dst.ElementType](
        src[t * in_stride + in_off + j]
    )


def slice_row_kernel[
    LT: TensorLayout
](
    src: TileTensor[DType.float32, LT, MutAnyOrigin],
    dst: TileTensor[DType.float32, LT, MutAnyOrigin],
    src_offset_arg: Int32,
    n_arg: Int32,
):
    """Copy n contiguous elements from src starting at src_offset into dst[0:n].
    Used to lift the last token's hidden row out before the LM head, so prefill
    runs the (VOCAB-wide) head on one row instead of all T (§11 #12).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        src: Source buffer (f32); `n` contiguous elements read from `src_offset`.
        dst: Destination buffer (f32); written at [0:n].
        src_offset: Start index in `src`.
        n: Element count.
    """
    var src_offset = Int(src_offset_arg)
    var n = Int(n_arg)
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
    1 / sqrt(mean(x²) + EPS). Qwen3 QK-RMSNorm normalizes per head before RoPE.
    """
    var ss = Float32(0.0)
    for d in range(HEAD_DIM):
        var v = rebind[Scalar[DType.float32]](src[base + d])
        ss += v * v
    return 1.0 / sqrt(ss / Float32(HEAD_DIM) + EPS)


def rope_k_kernel[
    LT: TensorLayout,
    HKV: Int,
    HEAD_DIM: Int,
    QK_NORM: Bool = False,
    KV_DT: DType = DType.float32,
](
    Kin: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # K source (row = in_stride, K at in_off)
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] cache (rotated)
    Kn: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [HEAD_DIM] k_norm weight (dummy if !QK_NORM)
    Tq_arg: Int32,
    q_offset_arg: Int32,
    in_stride_arg: Int32,  # source row stride (= nkv unfused, = hd+2nkv when reading fused qkv)
    in_off_arg: Int32,  # source column offset of K within the row (0 unfused, hd when fused)
    theta: Float32 = THETA,  # RoPE base (Qwen passes THETA; Gemma per-layer 1e4/1e6)
    rot_pairs_arg: Int32 = -1,  # # of d-pairs rotated (<0 = HALF = full rotary; Gemma partial = 64)
):
    """Apply RoPE to freshly-projected K and write it into the cache at its
    absolute-position rows. Doing this once on write (one thread per token×kv-
    head) replaces recomputing K's RoPE for every past key on every decode step
    inside attention (§11 #12). Same split-half rotation/θ as the Q path. Kin may
    be a strided slice of a fused [q|k|v] buffer (in_stride/in_off).

    When QK_NORM (Qwen3), each head is first RMS-normalized over HEAD_DIM and
    scaled by Kn[d] before the rotation (HF: q_norm/k_norm applied pre-RoPE).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.
        QK_NORM: Whether to apply per-head QK-RMSNorm before RoPE (Qwen3).

    Args:
        Kin: K source (f32); each row has stride in_stride, the K slice at column in_off.
        Kc: K cache [max, HKV, HEAD_DIM] (f32); rotated K written at row q_offset+t.
        Kn: K_norm weight [HEAD_DIM] (f32); used only when QK_NORM (dummy otherwise).
        Tq: Number of tokens this launch.
        q_offset: Absolute position of the first token (cache row base).
        in_stride: Source row stride (= nkv unfused, = hd+2·nkv reading fused qkv).
        in_off: Column offset of K within the source row (0 unfused, hd when fused).
        theta: RoPE base frequency (Qwen passes THETA; Gemma per-layer 1e4/1e6).
        rot_pairs: Number of d-pairs rotated (<0 = HEAD_DIM/2 = full rotary).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var in_stride = Int(in_stride_arg)
    var in_off = Int(in_off_arg)
    var rot_pairs = Int(rot_pairs_arg)
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
    var outbase = (
        q_offset + t
    ) * nkv + kvh * HEAD_DIM  # cache row = absolute position
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(Kin, inbase, HEAD_DIM)
    var lt = log(theta)
    var npair = rot_pairs if rot_pairs >= 0 else HALF
    for d in range(HALF):
        var x0 = rebind[Scalar[DType.float32]](Kin[inbase + d])
        var x1 = rebind[Scalar[DType.float32]](Kin[inbase + d + HALF])
        comptime if QK_NORM:
            var g0 = rebind[Scalar[DType.float32]](Kn[d])
            var g1 = rebind[Scalar[DType.float32]](Kn[d + HALF])
            x0 = x0 * rrms * g0
            x1 = x1 * rrms * g1
        if d < npair:  # partial rotary: only the first npair pairs rotate
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * lt)
            var ang = Float32(pos) * freq
            var c = cos(ang)
            var s = sin(ang)
            Kc[outbase + d] = rebind[Kc.ElementType](
                (x0 * c - x1 * s).cast[KV_DT]()
            )
            Kc[outbase + d + HALF] = rebind[Kc.ElementType](
                (x1 * c + x0 * s).cast[KV_DT]()
            )
        else:
            Kc[outbase + d] = rebind[Kc.ElementType]((x0).cast[KV_DT]())
            Kc[outbase + d + HALF] = rebind[Kc.ElementType]((x1).cast[KV_DT]())


def rope_kv_kernel[
    LT: TensorLayout,
    HKV: Int,
    HEAD_DIM: Int,
    QK_NORM: Bool = False,
    KV_DT: DType = DType.float32,
](
    In: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # fused [q|k|v] buffer (row = in_stride)
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] K cache (rotated)
    Vc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] V cache (copied)
    Kn: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [HEAD_DIM] k_norm weight (dummy if !QK_NORM)
    Tq_arg: Int32,
    q_offset_arg: Int32,
    in_stride_arg: Int32,  # source row stride (= hd + 2*nkv when reading fused qkv)
    k_off_arg: Int32,  # column offset of K within the row
    v_off_arg: Int32,  # column offset of V within the row
    theta: Float32 = THETA,
    rot_pairs_arg: Int32 = -1,
):
    """Rope_k_kernel + the V cache-copy in ONE launch: one thread per token×kv-head
    rotates that head's K into the cache AND copies its V into the cache. Both
    already walked the same (token, kv-head) grid and wrote a per-head cache slice,
    so merging halves the K/V write dispatches (rope_k + copy_strided → 1).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.
        QK_NORM: Whether to apply per-head QK-RMSNorm before RoPE (Qwen3).

    Args:
        In: Fused [q|k|v] source (f32); each row has stride in_stride.
        Kc: K cache [max, HKV, HEAD_DIM] (f32); rotated K written at row q_offset+t.
        Vc: V cache [max, HKV, HEAD_DIM] (f32); V copied (unrotated) at the same row.
        Kn: K_norm weight [HEAD_DIM] (f32); used only when QK_NORM (dummy otherwise).
        Tq: Number of tokens this launch.
        q_offset: Absolute position of the first token (cache row base).
        in_stride: Source row stride (= hd + 2·nkv when reading fused qkv).
        k_off: Column offset of K within the source row.
        v_off: Column offset of V within the source row.
        theta: RoPE base frequency.
        rot_pairs: Number of d-pairs rotated (<0 = HEAD_DIM/2 = full rotary).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var in_stride = Int(in_stride_arg)
    var k_off = Int(k_off_arg)
    var v_off = Int(v_off_arg)
    var rot_pairs = Int(rot_pairs_arg)
    comptime assert In.flat_rank == 1
    comptime HALF = HEAD_DIM // 2
    var nkv = HKV * HEAD_DIM
    var idx = Int(global_idx.x)
    if idx >= Tq * HKV:
        return
    var t = idx // HKV
    var kvh = idx % HKV
    var pos = q_offset + t
    var kin = t * in_stride + k_off + kvh * HEAD_DIM
    var vin = t * in_stride + v_off + kvh * HEAD_DIM
    var outbase = pos * nkv + kvh * HEAD_DIM  # cache row = absolute position
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(In, kin, HEAD_DIM)
    var lt = log(theta)
    var npair = rot_pairs if rot_pairs >= 0 else HALF
    for d in range(HALF):
        var x0 = rebind[Scalar[DType.float32]](In[kin + d])
        var x1 = rebind[Scalar[DType.float32]](In[kin + d + HALF])
        comptime if QK_NORM:
            var g0 = rebind[Scalar[DType.float32]](Kn[d])
            var g1 = rebind[Scalar[DType.float32]](Kn[d + HALF])
            x0 = x0 * rrms * g0
            x1 = x1 * rrms * g1
        if d < npair:
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * lt)
            var ang = Float32(pos) * freq
            var c = cos(ang)
            var s = sin(ang)
            Kc[outbase + d] = rebind[Kc.ElementType](
                (x0 * c - x1 * s).cast[KV_DT]()
            )
            Kc[outbase + d + HALF] = rebind[Kc.ElementType](
                (x1 * c + x0 * s).cast[KV_DT]()
            )
        else:
            Kc[outbase + d] = rebind[Kc.ElementType]((x0).cast[KV_DT]())
            Kc[outbase + d + HALF] = rebind[Kc.ElementType]((x1).cast[KV_DT]())
    # V is copied unrotated (HEAD_DIM contiguous values for this head).
    for d in range(HEAD_DIM):
        Vc[outbase + d] = rebind[Vc.ElementType]((In[vin + d]).cast[KV_DT]())


def rope_q_kernel[
    LT: TensorLayout, HQ: Int, HEAD_DIM: Int, QK_NORM: Bool = False
](
    Q: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # Q source (row = in_stride, Q at in_off)
    Qr: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [Tq, HQ, HEAD_DIM] rotated out (contiguous)
    Qn: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [HEAD_DIM] q_norm weight (dummy if !QK_NORM)
    Tq_arg: Int32,
    q_offset_arg: Int32,
    in_stride_arg: Int32,  # source row stride (= hd unfused, = hd+2nkv when reading fused qkv)
    in_off_arg: Int32,  # source column offset of Q within the row (0; Q is first in [q|k|v])
    theta: Float32 = THETA,
    rot_pairs_arg: Int32 = -1,
):
    """Apply RoPE to Q (one thread per query×head) into a rotated buffer, so the
    attention kernel itself does no transcendentals — same as K is rotated on
    write (§11 #12). Position = absolute query position q_offset+t. Q may be a
    strided slice of a fused [q|k|v] buffer (in_stride/in_off); Qr is contiguous.

    When QK_NORM (Qwen3), each head is first RMS-normalized over HEAD_DIM and
    scaled by Qn[d] before the rotation (HF: q_norm applied pre-RoPE).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HEAD_DIM: Per-head dimension.
        QK_NORM: Whether to apply per-head QK-RMSNorm before RoPE (Qwen3).

    Args:
        Q: Q source (f32); each row has stride in_stride, the Q slice at column in_off.
        Qr: Rotated output [Tq, HQ, HEAD_DIM] (f32), contiguous.
        Qn: Q_norm weight [HEAD_DIM] (f32); used only when QK_NORM (dummy otherwise).
        Tq: Number of tokens this launch.
        q_offset: Absolute position of the first query (cache row base).
        in_stride: Source row stride (= hd unfused, = hd+2·nkv reading fused qkv).
        in_off: Column offset of Q within the source row (0; Q is first in [q|k|v]).
        theta: RoPE base frequency.
        rot_pairs: Number of d-pairs rotated (<0 = HEAD_DIM/2 = full rotary).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var in_stride = Int(in_stride_arg)
    var in_off = Int(in_off_arg)
    var rot_pairs = Int(rot_pairs_arg)
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
    var lt = log(theta)
    var npair = rot_pairs if rot_pairs >= 0 else HALF
    for d in range(HALF):
        var x0 = rebind[Scalar[DType.float32]](Q[inb + d])
        var x1 = rebind[Scalar[DType.float32]](Q[inb + d + HALF])
        comptime if QK_NORM:
            var g0 = rebind[Scalar[DType.float32]](Qn[d])
            var g1 = rebind[Scalar[DType.float32]](Qn[d + HALF])
            x0 = x0 * rrms * g0
            x1 = x1 * rrms * g1
        if d < npair:
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * lt)
            var ang = Float32(pos) * freq
            var c = cos(ang)
            var s = sin(ang)
            Qr[base + d] = rebind[Qr.ElementType](x0 * c - x1 * s)
            Qr[base + d + HALF] = rebind[Qr.ElementType](x1 * c + x0 * s)
        else:
            Qr[base + d] = rebind[Qr.ElementType](x0)
            Qr[base + d + HALF] = rebind[Qr.ElementType](x1)


def attn_cached_kernel[
    LT: TensorLayout,
    HQ: Int,
    HKV: Int,
    HEAD_DIM: Int,
    KV_DT: DType = DType.float32,
](
    Q: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [Tq, HQ, HEAD_DIM] *RoPE-rotated*
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] RoPE-rotated, row = abs position
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],  # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],  # [Tq, HQ, HEAD_DIM]
    Tq_arg: Int32,
    q_offset_arg: Int32,
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
    under the forward tolerance, and greedy decode stays token-for-token (§11 #12).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.

    Args:
        Q: RoPE-rotated queries [Tq, HQ, HEAD_DIM] (f32).
        Kc: RoPE-rotated K cache [max, HKV, HEAD_DIM] (f32), row = absolute position.
        Vc: V cache [max, HKV, HEAD_DIM] (f32).
        O: Attention output [Tq, HQ, HEAD_DIM] (f32).
        Tq: Number of query tokens.
        q_offset: Absolute position of the first query (causal horizon per row).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    comptime GROUP = HQ // HKV
    var qh = Int(global_idx.x) // WARP_SIZE  # one warp per (query, head)
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
        comptime for c2 in range(NVEC // 2):
            # 16-wide f16 K load = the same 32 bytes/load as the old f32 path
            # (the loop is load-issue bound, so halving BYTES per load without
            # halving LOADS was a measured regression), widened once.
            var kw = Kc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            s += qreg[2 * c2] * kw.slice[VEC, offset=0]()
            s += qreg[2 * c2 + 1] * kw.slice[VEC, offset=VEC]()
        var score = s.reduce_add() * scale
        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        comptime for c2 in range(NVEC // 2):
            var vw = Vc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            accv[2 * c2] = accv[2 * c2] * corr + p * vw.slice[VEC, offset=0]()
            accv[2 * c2 + 1] = (
                accv[2 * c2 + 1] * corr + p * vw.slice[VEC, offset=VEC]()
            )
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


def attn_cached_rope_kernel[
    LT: TensorLayout,
    HQ: Int,
    HKV: Int,
    HEAD_DIM: Int,
    QK_NORM: Bool = False,
    KV_DT: DType = DType.float32,
](
    Q: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # RAW Q slice (row = q_stride, Q at q_off) — NOT pre-rotated
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] RoPE-rotated cache
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],  # [max, HKV, HEAD_DIM]
    Qn: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [HEAD_DIM] q_norm weight (dummy if !QK_NORM)
    O: TileTensor[DType.float32, LT, MutAnyOrigin],  # [Tq, HQ, HEAD_DIM]
    Tq_arg: Int32,
    q_offset_arg: Int32,
    q_stride_arg: Int32,
    q_off_arg: Int32,
):
    """Attn_cached_kernel with RoPE applied to Q *on load* — folds the rope_q launch
    (and its rotated-Q buffer) into attention at decode. With HALF=HEAD_DIM/2 the Q
    register chunk c pairs element-wise with chunk c+NVEC/2, so qreg is rotated in
    place with vectorized cos/sin (no extra registers). Qwen3 applies q_norm/RMS per
    head pre-rotation. Bit-parity with rope_q + attn_cached (same split-half θ).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.
        QK_NORM: Whether to apply per-head Q-RMSNorm before RoPE (Qwen3).

    Args:
        Q: RAW (un-rotated) Q slice (f32); each row has stride q_stride, Q at column q_off.
        Kc: RoPE-rotated K cache [max, HKV, HEAD_DIM] (f32), row = absolute position.
        Vc: V cache [max, HKV, HEAD_DIM] (f32).
        Qn: Q_norm weight [HEAD_DIM] (f32); used only when QK_NORM (dummy otherwise).
        O: Attention output [Tq, HQ, HEAD_DIM] (f32).
        Tq: Number of query tokens.
        q_offset: Absolute position of the first query (causal horizon per row).
        q_stride: Source row stride of Q.
        q_off: Column offset of Q within the source row.
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var q_stride = Int(q_stride_arg)
    var q_off = Int(q_off_arg)
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    comptime HALFC = NVEC // 2  # chunks per half (HALF = HALFC*VEC)
    comptime GROUP = HQ // HKV
    var qh = Int(global_idx.x) // WARP_SIZE
    var lane = Int(global_idx.x) % WARP_SIZE
    var h = qh % HQ
    var t = qh // HQ
    if t >= Tq:
        return
    var kvh = h // GROUP
    var qpos = q_offset + t
    var qbase = t * q_stride + q_off + h * HEAD_DIM
    var scale = 1.0 / sqrt(Float32(HEAD_DIM))

    # Load raw Q into registers, then rotate (RoPE) in place.
    var qreg = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for c in range(NVEC):
        qreg[c] = Q.raw_load[VEC](qbase + c * VEC)
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(Q, qbase, HEAD_DIM)
    comptime HALF = HEAD_DIM // 2
    for c in range(HALFC):
        var lo = qreg[c]
        var hi = qreg[c + HALFC]
        comptime if QK_NORM:
            lo = lo * rrms * Qn.raw_load[VEC](c * VEC)
            hi = hi * rrms * Qn.raw_load[VEC](c * VEC + HALF)
        var ang = SIMD[DType.float32, VEC](0.0)
        for e in range(VEC):
            var d = c * VEC + e
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
            ang[e] = Float32(qpos) * freq
        var cosv = cos(ang)
        var sinv = sin(ang)
        qreg[c] = lo * cosv - hi * sinv
        qreg[c + HALFC] = hi * cosv + lo * sinv

    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var accv = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for j in range(lane, qpos + 1, WARP_SIZE):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        var s = SIMD[DType.float32, VEC](0.0)
        comptime for c2 in range(NVEC // 2):
            # 16-wide f16 K load = the same 32 bytes/load as the old f32 path
            # (the loop is load-issue bound, so halving BYTES per load without
            # halving LOADS was a measured regression), widened once.
            var kw = Kc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            s += qreg[2 * c2] * kw.slice[VEC, offset=0]()
            s += qreg[2 * c2 + 1] * kw.slice[VEC, offset=VEC]()
        var score = s.reduce_add() * scale
        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        comptime for c2 in range(NVEC // 2):
            var vw = Vc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            accv[2 * c2] = accv[2 * c2] * corr + p * vw.slice[VEC, offset=0]()
            accv[2 * c2 + 1] = (
                accv[2 * c2 + 1] * corr + p * vw.slice[VEC, offset=VEC]()
            )
        m = m_new

    var m_g = warp_max(m)
    var f = exp(m - m_g)
    var l_g = warp_sum(l * f)
    var obase = (t * HQ + h) * HEAD_DIM
    for c in range(NVEC):
        for e in range(VEC):
            var a = warp_sum(accv[c][e] * f)
            if lane == 0:
                O[obase + c * VEC + e] = rebind[O.ElementType](a / l_g)


def attn_cached_rope_kw_kernel[
    LT: TensorLayout,
    HQ: Int,
    HKV: Int,
    HEAD_DIM: Int,
    QK_NORM: Bool = False,
    W: Int = 4,
    KV_DT: DType = DType.float32,
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],
    Kc: TileTensor[KV_DT, LT, MutAnyOrigin],
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],
    Qn: TileTensor[DType.float32, LT, MutAnyOrigin],
    O: TileTensor[DType.float32, LT, MutAnyOrigin],
    Tq_arg: Int32,
    q_offset_arg: Int32,
    q_stride_arg: Int32,
    q_off_arg: Int32,
):
    """`attn_cached_rope_kernel` with the key scan K-SPLIT across W warps — the
    same starved-parallelism fix as the GEMV rw kernel, applied to attention.
    One threadgroup per (query, head); each of its W warps runs the flash scan
    over a disjoint key comb (j = w·32+lane, stride W·32), does the existing
    cross-LANE merge, then the W per-warp partials (m, l, acc[HEAD_DIM]) meet
    in threadgroup memory for a cross-WARP flash merge by warp 0. Each warp
    redundantly loads+rotates the same Q (cheap: HEAD_DIM work vs the key
    scan). Motivation: at 1570-ctx decode the old kernel gave each (query,
    head) only 32 lanes → ~50 SEQUENTIAL flash steps per lane; long-context
    attention is scan-latency bound (measured — f16 KV made it slower), so
    more warps per query is the lever, not fewer bytes.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.
        QK_NORM: Whether to apply per-head Q-RMSNorm before RoPE (Qwen3).
        W: Key-scan warps per (query, head).
        KV_DT: KV-cache element dtype.

    Args:
        Q: RAW (un-rotated) Q slice (f32); rows of stride q_stride, Q at q_off.
        Kc: RoPE-rotated K cache [max, HKV, HEAD_DIM], row = absolute position.
        Vc: V cache [max, HKV, HEAD_DIM].
        Qn: Q_norm weight [HEAD_DIM] (f32); used only when QK_NORM.
        O: Attention output [Tq, HQ, HEAD_DIM] (f32).
        Tq: Number of query tokens.
        q_offset: Absolute position of the first query.
        q_stride: Source row stride of Q.
        q_off: Column offset of Q within the source row.
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var q_stride = Int(q_stride_arg)
    var q_off = Int(q_off_arg)
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    comptime HALFC = NVEC // 2
    comptime GROUP = HQ // HKV
    var qh = Int(block_idx.x)  # one threadgroup per (query, head)
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var h = qh % HQ
    var t = qh // HQ
    if t >= Tq:
        return
    var kvh = h // GROUP
    var qpos = q_offset + t
    var qbase = t * q_stride + q_off + h * HEAD_DIM
    var scale = 1.0 / sqrt(Float32(HEAD_DIM))

    # Every warp loads + rotates the same Q (identical math, no sync needed).
    var qreg = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for c in range(NVEC):
        qreg[c] = Q.raw_load[VEC](qbase + c * VEC)
    var rrms = Float32(1.0)
    comptime if QK_NORM:
        rrms = _head_rrms(Q, qbase, HEAD_DIM)
    comptime HALF = HEAD_DIM // 2
    for c in range(HALFC):
        var lo = qreg[c]
        var hi = qreg[c + HALFC]
        comptime if QK_NORM:
            lo = lo * rrms * Qn.raw_load[VEC](c * VEC)
            hi = hi * rrms * Qn.raw_load[VEC](c * VEC + HALF)
        var ang = SIMD[DType.float32, VEC](0.0)
        for e in range(VEC):
            var d = c * VEC + e
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
            ang[e] = Float32(qpos) * freq
        var cosv = cos(ang)
        var sinv = sin(ang)
        qreg[c] = lo * cosv - hi * sinv
        qreg[c + HALFC] = hi * cosv + lo * sinv

    # This warp's comb of keys: j = w*32+lane, stride W*32.
    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var accv = InlineArray[SIMD[DType.float32, VEC], NVEC](fill=0.0)
    for j in range(w * WARP_SIZE + lane, qpos + 1, W * WARP_SIZE):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        var s = SIMD[DType.float32, VEC](0.0)
        comptime for c2 in range(NVEC // 2):
            var kw = Kc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            s += qreg[2 * c2] * kw.slice[VEC, offset=0]()
            s += qreg[2 * c2 + 1] * kw.slice[VEC, offset=VEC]()
        var score = s.reduce_add() * scale
        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        comptime for c2 in range(NVEC // 2):
            var vw = Vc.raw_load[2 * VEC](kbase + c2 * 2 * VEC).cast[
                DType.float32
            ]()
            accv[2 * c2] = accv[2 * c2] * corr + p * vw.slice[VEC, offset=0]()
            accv[2 * c2 + 1] = (
                accv[2 * c2 + 1] * corr + p * vw.slice[VEC, offset=VEC]()
            )
        m = m_new

    # Tier 1: the existing cross-lane merge → one (m, l, acc) per warp.
    var m_w = warp_max(m)
    var f = exp(m - m_w)
    var l_w = warp_sum(l * f)
    # Tier 2: per-warp partials meet in threadgroup memory.
    var sm = stack_allocation[
        W * (HEAD_DIM + 2), Float32, address_space=AddressSpace.SHARED
    ]()
    for c in range(NVEC):
        for e in range(VEC):
            var a = warp_sum(accv[c][e] * f)
            if lane == 0:
                sm[unsafe_offset=w * (HEAD_DIM + 2) + c * VEC + e] = a
    if lane == 0:
        sm[unsafe_offset=w * (HEAD_DIM + 2) + HEAD_DIM] = m_w
        sm[unsafe_offset=w * (HEAD_DIM + 2) + HEAD_DIM + 1] = l_w
    barrier()
    if w == 0:
        # Global flash merge over the W partials (empty combs have l=0 and
        # m=-1e30, so their weight collapses to zero — no special-casing).
        var m_g = Float32(-1.0e30)
        comptime for i in range(W):
            m_g = max(m_g, sm[unsafe_offset=i * (HEAD_DIM + 2) + HEAD_DIM])
        var l_g = Float32(0.0)
        comptime for i in range(W):
            l_g += sm[unsafe_offset=i * (HEAD_DIM + 2) + HEAD_DIM + 1] * exp(
                sm[unsafe_offset=i * (HEAD_DIM + 2) + HEAD_DIM] - m_g
            )
        var obase = (t * HQ + h) * HEAD_DIM
        for d in range(lane, HEAD_DIM, WARP_SIZE):
            var a = Float32(0.0)
            comptime for i in range(W):
                a += sm[unsafe_offset=i * (HEAD_DIM + 2) + d] * exp(
                    sm[unsafe_offset=i * (HEAD_DIM + 2) + HEAD_DIM] - m_g
                )
            O[obase + d] = rebind[O.ElementType](a / l_g)


comptime FLASH_BK = WARP_SIZE  # flash keys per tile = one per lane
"""Flash-attention keys per tile: one per lane (= WARP_SIZE)."""


def flash_attn_kernel[
    LT: TensorLayout,
    HQ: Int,
    HKV: Int,
    HEAD_DIM: Int,
    PW: Int,
    KV_DT: DType = DType.float32,
](
    Q: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [Tq, HQ, HEAD_DIM] *RoPE-rotated*
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] rotated, row = abs pos
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],  # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],  # [Tq, HQ, HEAD_DIM]
    Tq_arg: Int32,
    q_offset_arg: Int32,
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
    the same (verified max|diff|=0). Only the read path differs.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.
        PW: Query positions per block (warps/block = PW · GROUP).

    Args:
        Q: RoPE-rotated queries [Tq, HQ, HEAD_DIM] (f32).
        Kc: RoPE-rotated K cache [max, HKV, HEAD_DIM] (f32), row = absolute position.
        Vc: V cache [max, HKV, HEAD_DIM] (f32).
        O: Attention output [Tq, HQ, HEAD_DIM] (f32).
        Tq: Number of query tokens.
        q_offset: Absolute position of the first query (causal horizon per row).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    comptime assert Q.flat_rank == 1
    comptime VEC = 8
    comptime NVEC = HEAD_DIM // VEC
    comptime GROUP = HQ // HKV
    comptime NWARP = PW * GROUP
    comptime NTHREAD = NWARP * WARP_SIZE
    var Ks = stack_allocation[
        FLASH_BK * HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var Vs = stack_allocation[
        FLASH_BK * HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()

    var tib = Int(thread_idx.x)
    var warp = tib // WARP_SIZE
    var lane = tib % WARP_SIZE
    var blk = Int(block_idx.x)
    var kvh = blk % HKV
    var q0 = (blk // HKV) * PW
    var qi = warp // GROUP  # query position within the tile (0 … PW-1)
    var gi = warp % GROUP  # head within the kv-group (0 … GROUP-1)
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
                Ks[unsafe_offset=idx] = rebind[Scalar[KV_DT]](Kc[src]).cast[
                    DType.float32
                ]()
                Vs[unsafe_offset=idx] = rebind[Scalar[KV_DT]](Vc[src]).cast[
                    DType.float32
                ]()
            else:
                Ks[unsafe_offset=idx] = Float32(0.0)
                Vs[unsafe_offset=idx] = Float32(0.0)
        barrier()

        if active:
            var j = kt0 + lane
            if j <= qpos:
                var kb = lane * HEAD_DIM
                var s = SIMD[DType.float32, VEC](0.0)
                for c in range(NVEC):
                    s += qreg[c] * Ks.unsafe_load[width=VEC](kb + c * VEC)
                var score = s.reduce_add() * scale
                var m_new = max(m, score)
                var corr = exp(m - m_new)
                var p = exp(score - m_new)
                lsum = lsum * corr + p
                for c in range(NVEC):
                    accv[c] = accv[c] * corr + p * Vs.unsafe_load[width=VEC](
                        kb + c * VEC
                    )
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


def tc_attn_kernel[
    LT: TensorLayout,
    HQ: Int,
    HKV: Int,
    HEAD_DIM: Int,
    KV_DT: DType = DType.float32,
](
    Q: TileTensor[
        DType.float32, LT, MutAnyOrigin
    ],  # [Tq, HQ, HEAD_DIM] *RoPE-rotated*
    Kc: TileTensor[
        KV_DT, LT, MutAnyOrigin
    ],  # [max, HKV, HEAD_DIM] rotated, row = abs pos
    Vc: TileTensor[KV_DT, LT, MutAnyOrigin],  # [max, HKV, HEAD_DIM]
    O: TileTensor[DType.float32, LT, MutAnyOrigin],  # [Tq, HQ, HEAD_DIM]
    Tq_arg: Int32,
    q_offset_arg: Int32,
    scale_in: Float32 = -1.0,  # softmax scale; <0 = 1/sqrt(HEAD_DIM) (Qwen), Gemma passes 1.0
    window_arg: Int32 = 0,  # >0 = sliding window: attend only to the last `window`
    #   keys (Gemma's sliding layers, 1024); 0 = full causal.
):
    """TENSOR-CORE causal GQA attention for *prefill* — the big long-prefill lever.

    `attn_cached_kernel` gives each (query,head) one warp doing SCALAR per-key dot
    products; attention then dominates prefill and grows ~quadratically (the 8×
    gap to MLX at long context). This kernel instead computes S = Q·Kᵀ and O = P·V
    on the Apple 8×8 simdgroup-matrix units (`_mma8x8`), the same intrinsic the
    GEMM uses. One simdgroup (32 lanes, 1 warp) owns an 8-query tile of one head:
    Q lives in ND=HEAD_DIM/8 A-fragments, O in ND C-fragments, and K/V stream in
    8-key tiles. The online softmax reduces the S fragment ALONG keys with
    `_frag_row_max`/`_frag_row_sum` (butterfly shuffles over the 4 lanes that share
    a fragment row). Measured ~27× (P=512) … ~32× (P=1536) over attn_cached on the
    M4, bit-exact vs the scalar kernel (max|Δ|≈1e-7).

    Causal masking zeros P past the diagonal, so a masked key's V contribution is
    killed in the MMA regardless — V is read with a branchless clamped row index
    (a per-lane divergent guard around the unrolled MMA accumulate miscompiles on
    this toolchain, silently corrupting the first output d-tile).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        HQ: Number of query heads.
        HKV: Number of key/value heads.
        HEAD_DIM: Per-head dimension.

    Args:
        Q: RoPE-rotated queries [Tq, HQ, HEAD_DIM] (f32).
        Kc: RoPE-rotated K cache [max, HKV, HEAD_DIM] (f32), row = absolute position.
        Vc: V cache [max, HKV, HEAD_DIM] (f32).
        O: Attention output [Tq, HQ, HEAD_DIM] (f32).
        Tq: Number of query tokens.
        q_offset: Absolute position of the first query (causal horizon per row).
        scale_in: Softmax scale; <0 selects 1/sqrt(HEAD_DIM) (Qwen; Gemma passes 1.0).
        window: Sliding-window width (>0 = attend only the last `window` keys;
            0 = full causal).
    """
    var Tq = Int(Tq_arg)
    var q_offset = Int(q_offset_arg)
    var window = Int(window_arg)
    comptime assert Q.flat_rank == 1
    comptime ND = HEAD_DIM // _MMA8  # d-tiles
    comptime GROUP = HQ // HKV
    var lane = Int(thread_idx.x) % 32
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]

    var bx = Int(block_idx.x)
    var qt = bx // HQ
    var h = bx % HQ
    var q0 = qt * _MMA8
    var kvh = h // GROUP
    var scale = scale_in if scale_in >= 0.0 else 1.0 / sqrt(Float32(HEAD_DIM))

    var row = q0 + frow
    var qpos = q_offset + row  # this lane's row abs position
    # max abs position any active query in this tile attends to (block-uniform)
    var last_row = q0 + _MMA8 - 1
    if last_row > Tq - 1:
        last_row = Tq - 1
    var kmax = q_offset + last_row

    var qp = Q.ptr
    var kp = Kc.ptr
    var vp = Vc.ptr

    # Q resident in A-fragments: afrag[dt][s] = Q[row, h, dt*8 + fcol + s]
    var afrag = InlineArray[SIMD[DType.float32, _FRAG8], ND](uninitialized=True)
    comptime for dt in range(ND):
        var af = SIMD[DType.float32, _FRAG8](0)
        if row < Tq:
            af = qp.unsafe_offset(
                (row * HQ + h) * HEAD_DIM + dt * _MMA8 + fcol
            ).unsafe_load[width=_FRAG8]()
        afrag[dt] = af

    var ofrag = InlineArray[SIMD[DType.float32, _FRAG8], ND](
        fill=SIMD[DType.float32, _FRAG8](0)
    )
    var m = Float32(-1.0e30)
    var l = Float32(0.0)

    # Sliding window: every active query in this 8-tile attends back at most
    # `window` keys, so skip whole key-tiles older than the earliest in-window key
    # of the first query (block-uniform start, floored to an 8-key tile). The
    # per-key mask below still trims the partial boundary tile per query.
    var kt = 0
    if window > 0:
        var kfloor = q_offset + q0 - window + 1
        if kfloor < 0:
            kfloor = 0
        kt = (kfloor // _MMA8) * _MMA8
    while kt <= kmax:
        # S[8q × 8k] = Q · Kᵀ over HEAD_DIM, accumulated on the MMA.
        var sfrag = SIMD[DType.float32, _FRAG8](0)
        comptime for dt in range(ND):
            var bk = SIMD[DType.float32, _FRAG8](0)
            var kd = dt * _MMA8 + frow  # d index for the B (Kᵀ) operand
            comptime for s in range(_FRAG8):
                var key = kt + fcol + s
                if key <= kmax:
                    bk[s] = kp[
                        unsafe_offset=(key * HKV + kvh) * HEAD_DIM + kd
                    ].cast[DType.float32]()
            sfrag = _mma8x8(afrag[dt], bk, sfrag)

        sfrag *= scale
        # causal mask: key past this row's abs position → −inf; with a sliding
        # window, also mask keys older than `qpos − window + 1`.
        comptime for s in range(_FRAG8):
            var keyp = kt + fcol + s
            if keyp > qpos or (window > 0 and qpos - keyp >= window):
                sfrag[s] = Float32(-1.0e30)

        var rmax = _frag_row_max(max(sfrag[0], sfrag[1]))
        var m_new = max(m, rmax)
        var corr = exp(m - m_new)
        var p0 = exp(sfrag[0] - m_new)
        var p1 = exp(sfrag[1] - m_new)
        var pfrag = SIMD[DType.float32, _FRAG8](p0, p1)
        l = l * corr + _frag_row_sum(p0 + p1)

        # O = O*corr + P·V (V non-transposed: bv[s]=V[key=kt+frow, d=col]). Masked
        # keys carry P=0, so clamp the row index (no divergent branch — see above).
        var key_a = kt + frow
        if key_a > kmax:
            key_a = kmax
        comptime for dt in range(ND):
            ofrag[dt] = ofrag[dt] * corr
        comptime for dt in range(ND):
            var bv = SIMD[DType.float32, _FRAG8](0)
            comptime for s in range(_FRAG8):
                bv[s] = vp[
                    unsafe_offset=(key_a * HKV + kvh) * HEAD_DIM
                    + dt * _MMA8
                    + fcol
                    + s
                ].cast[DType.float32]()
            ofrag[dt] = _mma8x8(pfrag, bv, ofrag[dt])

        m = m_new
        kt += _MMA8

    if row < Tq:
        var ob = (row * HQ + h) * HEAD_DIM
        comptime for dt in range(ND):
            comptime for s in range(_FRAG8):
                O[ob + dt * _MMA8 + fcol + s] = rebind[O.ElementType](
                    ofrag[dt][s] / l
                )


# ── token-selection kernels (GPU argmax / top-k; decode's host-loop fix) ───────
# The server used to download all `vocab` logits (600 KB at 3B) every token and
# scan them on the host (argmax, and `process_logits`' k×vocab selection loop) —
# ~33 ms/token of host work once the MAX 26.5 driver made the GPU step cheap.
# These kernels keep selection on-device; the host downloads 2·k floats.
# Index values are carried as Float32 (exact for vocab < 2^24) so the partial
# and output buffers stay single-dtype. Ties resolve to the LOWEST index, byte-
# matching the host scan's first-wins `>` comparison.

comptime AMAX_NEG = Float32(-1.0e30)
"""Masked-out / identity value for the argmax reductions (mirrors the host
sentinel in `sampling.process_logits`)."""


def penalize_copy_kernel[
    LT: TensorLayout
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    W: TileTensor[DType.float32, LT, MutAnyOrigin],
    mask: TileTensor[DType.uint8, LT, MutAnyOrigin],
    n_arg: Int32,
    base_arg: Int32,
    pen: Float32,
    last_id_arg: Int32,
):
    """W[i] = X[base+i] with the HF repetition penalty applied to seen tokens.

    A token counts as seen when `mask[i] != 0` OR `i == last_id` — the latter so
    the just-generated token is penalized THIS step without a cross-block write
    ordering dependency; the thread owning `last_id` also persists `mask` for
    later steps. `last_id < 0` marks nothing (prompt-only / first step).

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.

    Args:
        X: Source logits (read at offset `base`).
        W: Destination work buffer (penalized copy, length n).
        mask: Per-vocab seen mask (uint8; 1 = token in context).
        n_arg: Vocab size.
        base_arg: Row offset into X.
        pen: Repetition penalty (1.0 = no-op).
        last_id_arg: Token id generated by the previous step, or -1.
    """
    comptime assert X.flat_rank == 1
    comptime assert W.flat_rank == 1
    comptime assert mask.flat_rank == 1
    var n = Int(n_arg)
    var base = Int(base_arg)
    var last_id = Int(last_id_arg)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var v = rebind[Scalar[DType.float32]](X[base + i])
    var seen = rebind[Scalar[DType.uint8]](mask[i]) != 0 or i == last_id
    if seen and pen != 1.0:
        v = v / pen if v > 0 else v * pen
    W[i] = rebind[W.ElementType](v)
    if i == last_id:
        mask[i] = rebind[mask.ElementType](UInt8(1))


def mask_mark_kernel[
    LT: TensorLayout
](
    ids: TileTensor[DType.int32, LT, MutAnyOrigin],
    mask: TileTensor[DType.uint8, LT, MutAnyOrigin],
    n_ids_arg: Int32,
):
    """mask[ids[t]] = 1 for each of the n_ids prompt/context tokens."""
    comptime assert ids.flat_rank == 1
    comptime assert mask.flat_rank == 1
    var n_ids = Int(n_ids_arg)
    var t = Int(global_idx.x)
    if t >= n_ids:
        return
    var id = Int(rebind[Scalar[DType.int32]](ids[t]))
    mask[id] = rebind[mask.ElementType](UInt8(1))


def argmax_part_kernel[
    LT: TensorLayout
](
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.float32, LT, MutAnyOrigin],
    n_arg: Int32,
    base_arg: Int32,
):
    """Stage 1: per-warp argmax partials over S[base .. base+n).

    Every thread strides the row by the total thread count, keeping its best
    (value, index) — later elements replace only on strictly-greater, so each
    thread's best is its LOWEST-index maximum. A shuffle-down warp reduction
    (preferring the lower index on ties) leaves the warp best on lane 0, which
    writes (index, value) to P[2·warp .. 2·warp+1].

    Args:
        S: Source row (read-only here; the merge stage masks winners out).
        P: Per-warp partials, 2 floats per warp (index, value).
        n_arg: Row length.
        base_arg: Row offset into S.
    """
    comptime assert S.flat_rank == 1
    comptime assert P.flat_rank == 1
    var n = Int(n_arg)
    var base = Int(base_arg)
    var total = Int(grid_dim.x) * Int(block_dim.x)
    var t = Int(global_idx.x)
    var best_v = AMAX_NEG
    var best_i = Float32(0)
    var i = t
    while i < n:
        var v = rebind[Scalar[DType.float32]](S[base + i])
        if v > best_v:
            best_v = v
            best_i = Float32(i)
        i += total
    comptime for j in range(5):
        comptime off = 16 >> j
        var ov = warp_shuffle_down(best_v, UInt32(off))
        var oi = warp_shuffle_down(best_i, UInt32(off))
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
    var lane = Int(thread_idx.x) % WARP_SIZE
    if lane == 0:
        var w = t // WARP_SIZE
        P[2 * w] = rebind[P.ElementType](best_i)
        P[2 * w + 1] = rebind[P.ElementType](best_v)


def argmax_merge_kernel[
    LT: TensorLayout
](
    P: TileTensor[DType.float32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    Out: TileTensor[DType.float32, LT, MutAnyOrigin],
    nparts_arg: Int32,
    slot_arg: Int32,
    mask_out_arg: Int32,
):
    """Stage 2 (ONE warp): merge the per-warp partials to the row argmax.

    Writes (index, value) to Out[2·slot .. 2·slot+1]. With `mask_out` nonzero
    the winning element of S is overwritten with `AMAX_NEG`, so repeated
    part+merge rounds yield the top-k in descending order (the iterative
    selection the host `process_logits` used, moved on-device).

    Args:
        P: Stage-1 partials (index, value) pairs.
        S: The work row (mutated only when mask_out != 0).
        Out: Result slots, 2 floats per k.
        nparts_arg: Number of partial pairs.
        slot_arg: Output slot to write.
        mask_out_arg: Nonzero → mask the winner out of S for the next round.
    """
    comptime assert P.flat_rank == 1
    comptime assert S.flat_rank == 1
    comptime assert Out.flat_rank == 1
    var nparts = Int(nparts_arg)
    var slot = Int(slot_arg)
    var t = Int(thread_idx.x)
    var best_v = AMAX_NEG
    var best_i = Float32(0)
    var p = t
    while p < nparts:
        var v = rebind[Scalar[DType.float32]](P[2 * p + 1])
        var fi = rebind[Scalar[DType.float32]](P[2 * p])
        if v > best_v or (v == best_v and fi < best_i):
            best_v = v
            best_i = fi
        p += WARP_SIZE
    comptime for j in range(5):
        comptime off = 16 >> j
        var ov = warp_shuffle_down(best_v, UInt32(off))
        var oi = warp_shuffle_down(best_i, UInt32(off))
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
    if t == 0:
        Out[2 * slot] = rebind[Out.ElementType](best_i)
        Out[2 * slot + 1] = rebind[Out.ElementType](best_v)
        if Int(mask_out_arg) != 0:
            S[Int(best_i)] = rebind[S.ElementType](AMAX_NEG)


# ── mrw decode GEMVs (R rows/threadgroup × W K-slice warps; 2026-08-07) ────────
# Promoted from tests/manual/q4_gemv_mr.mojo after REAL-WEIGHT chain benches:
# on 36-layer streaming data (no cache reuse) the plain one-warp-per-output
# GEMV runs ~35 GB/s while mrw[2,W] hits 90 GB/s on gate_up (K=2048,N=22016)
# and 72 GB/s on down (K=11008,N=2048) — each warp loads an x oct once and
# reuses it for R weight streams, and W K-slice warps per row tile keep loads
# in flight. Small shapes (qkv/o, ≤2.6 MB) are launch-latency-bound and stay
# on the plain kernel. Synthetic same-buffer sweeps OVERSTATE all of these
# (cache-resident weights) — re-tune only against real checkpoint streams.


def matmul_q4_norm_mrw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    LNW: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """RMSNorm-fused mrw decode GEMV (M=1): `Y = (RMSNorm(x)·lnw)·Wq4ᵀ (+B)`.

    The norm folds in linearly (matmul_q4_norm_kernel's trick): every warp
    accumulates its K-slice's unnormalized dot (lnw folded) AND its partial
    Σx², both meet in threadgroup memory, and the writer lane divides by the
    full-row rms once.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Rows per threadgroup (each warp carries all R).
        W: K-slice warps.

    Args:
        X: Input activations [1, K] (f32, un-normalized).
        LNW: RMSNorm weight [K].
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [1, N] (f32).
        K_arg: Contraction (input) dimension.
        N_arg: Number of output channels.
        NG_arg: Groups per row (K / Q4_GROUP).
        use_bias_arg: Add B when nonzero.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var lwp = LNW.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var sspart = stack_allocation[
        W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    var ss = Float32(0.0)
    for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
        var k0 = q * 64
        var w8 = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:  # warp-uniform guard
                w8[r] = pp.unsafe_offset((n0 + r) * words + q * 8).unsafe_load[
                    width=8
                ]()
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            var xr = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
            var lw = lwp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
            ss += (xr * xr).reduce_add()
            xv[j] = xr * lw
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                var racc = Float32(0.0)
                comptime for j in range(8):
                    var nibs = (
                        SIMD[DType.uint32, 8](w8[r][j]) >> _Q4_SHIFTS
                    ) & 0xF
                    var qf = (nibs.cast[DType.int32]() - 8).cast[
                        DType.float32
                    ]()
                    racc += (qf * xv[j]).reduce_add()
                acc[r] += racc * rebind[Float32](s)
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    var pss = warp_sum(ss)
    if lane == 0:
        sspart[unsafe_offset=w] = pss
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var ss_all = Float32(0.0)
        comptime for i in range(W):
            ss_all += sspart[unsafe_offset=i]
        var rms = sqrt(ss_all / Float32(K) + EPS)
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        total = total / rms
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def matmul_q4_silu_resid_mrw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    GU: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    Resid: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
):
    """SwiGLU-input + residual-output mrw decode GEMV (M=1):
    `Y = (silu(gate)·up)·Wq4ᵀ + Resid` with gate = GU[:K], up = GU[K:2K].

    Args:
        GU: Concatenated [gate|up] activations [1, 2K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        Resid: Residual input [1, N] added to the output.
        Y: Output [1, N] (f32).
        K_arg: Contraction (input) dimension.
        N_arg: Number of output channels.
        NG_arg: Groups per row (K / Q4_GROUP).
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    comptime assert GU.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var gp = GU.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
        var k0 = q * 64
        var w8 = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:  # warp-uniform guard
                w8[r] = pp.unsafe_offset((n0 + r) * words + q * 8).unsafe_load[
                    width=8
                ]()
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            var g = gp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
            var u = gp.unsafe_offset(k0 + j * 8 + K).unsafe_load[width=8]()
            xv[j] = (g / (1.0 + exp(-g))) * u
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                var racc = Float32(0.0)
                comptime for j in range(8):
                    var nibs = (
                        SIMD[DType.uint32, 8](w8[r][j]) >> _Q4_SHIFTS
                    ) & 0xF
                    var qf = (nibs.cast[DType.int32]() - 8).cast[
                        DType.float32
                    ]()
                    racc += (qf * xv[j]).reduce_add()
                acc[r] += racc * rebind[Float32](s)
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        total += rebind[Scalar[DType.float32]](Resid[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)


def matmul_q4_mrw_kernel[
    LT: TensorLayout, R: Int, W: Int
](
    X: TileTensor[DType.float32, LT, MutAnyOrigin],
    P: TileTensor[DType.uint32, LT, MutAnyOrigin],
    S: TileTensor[DType.float32, LT, MutAnyOrigin],
    B: TileTensor[DType.float32, LT, MutAnyOrigin],
    Y: TileTensor[DType.float32, LT, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    NG_arg: Int32,
    use_bias_arg: Int32,
):
    """Decode GEMV (M=1): the mr and rw ideas COMPOSED. A threadgroup owns R
    adjacent rows; its W warps each sweep a disjoint K-slice carrying ALL R
    rows — so each x oct is loaded once per warp and reused for R weight
    streams (the square-shape fix: x-issue traffic /R), while W warps per row
    tile keep enough warps in flight to hide load latency (the rw fix). Warp
    partials for each row meet in threadgroup memory.

    Parameters:
        LT: Tensor layout type for the flat 1D buffers.
        R: Rows per threadgroup (each warp carries all R).
        W: K-slice warps.

    Args:
        X: Input activations [1, K] (f32).
        P: Packed int4 weights (u32, 8 nibbles/word) for [N, K].
        S: Per-group f32 scales [N, NG].
        B: Bias [N] (f32), added when use_bias != 0.
        Y: Output [1, N] (f32).
        K: Contraction (input) dimension.
        N: Number of output channels.
        NG: Groups per row (K / Q4_GROUP).
        use_bias: Add B when nonzero.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var NG = Int(NG_arg)
    var use_bias = Int(use_bias_arg)
    comptime assert X.flat_rank == 1
    var n0 = Int(block_idx.x) * R
    var w = Int(thread_idx.x) // WARP_SIZE  # K-slice index (warp id)
    var lane = Int(thread_idx.x) % WARP_SIZE
    var words = K // 8
    var octs = words // 8
    var pp = P.ptr
    var xp = X.ptr
    var sp = S.ptr
    var part = stack_allocation[
        R * W, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, R](fill=0.0)
    for q in range(w * WARP_SIZE + lane, octs, W * WARP_SIZE):
        var k0 = q * 64
        var w8 = InlineArray[SIMD[DType.uint32, 8], R](fill=0)
        comptime for r in range(R):
            if n0 + r < N:  # warp-uniform guard
                w8[r] = pp.unsafe_offset((n0 + r) * words + q * 8).unsafe_load[
                    width=8
                ]()
        var xv = InlineArray[SIMD[DType.float32, 8], 8](fill=0)
        comptime for j in range(8):
            xv[j] = xp.unsafe_offset(k0 + j * 8).unsafe_load[width=8]()
        comptime for r in range(R):
            if n0 + r < N:
                var s = sp[unsafe_offset=(n0 + r) * NG + (k0 >> Q4_SHIFT)]
                var racc = Float32(0.0)
                comptime for j in range(8):
                    var nibs = (
                        SIMD[DType.uint32, 8](w8[r][j]) >> _Q4_SHIFTS
                    ) & 0xF
                    var qf = (nibs.cast[DType.int32]() - 8).cast[
                        DType.float32
                    ]()
                    racc += (qf * xv[j]).reduce_add()
                acc[r] += racc * rebind[Float32](s)
    comptime for r in range(R):
        var p = warp_sum(acc[r])
        if lane == 0:
            part[unsafe_offset=r * W + w] = p
    barrier()
    if w == 0 and lane < R and n0 + lane < N:
        var total = Float32(0.0)
        comptime for i in range(W):
            total += part[unsafe_offset=lane * W + i]
        if use_bias != 0:
            total += rebind[Scalar[DType.float32]](B[n0 + lane])
        Y[n0 + lane] = rebind[Y.ElementType](total)
