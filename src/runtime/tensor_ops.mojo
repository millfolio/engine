"""Model-agnostic tensor op launchers over src/kernels.mojo. Each function runs
one kernel and returns a fresh device buffer (or writes into a provided one).
The weight representation types (DevBuf/WBuf/PBuf, QMat) live here too, so both
the loader (safetensors.mojo) and the model can share them without a cycle.

No model-specific constants: every op takes its dims at the call site."""

from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.gpu import WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from runtime.kernel_cache import cached_enqueue

from kernels import (
    penalize_copy_kernel,
    mask_mark_kernel,
    argmax_part_kernel,
    argmax_merge_kernel,
    matmul_kernel,
    matmul_simd_kernel,
    matmul_tiled_kernel,
    matmul_q4_kernel,
    matmul_q4_rw_kernel,
    matmul_q4_norm_rw_kernel,
    matmul_q4_resid_rw_kernel,
    matmul_q4_batch_kernel,
    matmul_q4_small_kernel,
    matmul_simd_q4_kernel,
    matmul_tiled_q4_kernel,
    matmul_hkn_kernel,
    dequant_q4_ht_kernel,
    cast_f32_h_kernel,
    matmul_resid_kernel,
    matmul_q4_resid_kernel,
    matmul_norm_kernel,
    matmul_q4_norm_kernel,
    matmul_q4_mrw_kernel,
    matmul_q4_norm_mrw_kernel,
    matmul_q4_silu_resid_mrw_kernel,
    matmul_silu_resid_kernel,
    matmul_q4_silu_resid_kernel,
    rmsnorm_kernel,
    add_kernel,
    silu_mul_kernel,
    silu_mul_cat_kernel,
    gelu_mul_cat_kernel,
    gelu_mul_kernel,
    gelu_mul_strided_kernel,
    rmsnorm_add_kernel,
    nll_gather_kernel,
    softcap_kernel,
    add_scalar_kernel,
    mul_scalar_kernel,
    vnorm_kernel,
    embed_kernel,
    slice_row_kernel,
    copy_kernel,
    copy_strided_kernel,
    SG_BM,
    SG_BN,
    SG_TPB,
    Q4_GROUP,
    SPEC_MAX_M,
    SPEC_SMALL_MIN,
    _SM_BN,
    _SM_TPB,
)

comptime BLOCK = 256
"""Default GPU threads-per-block for the 1D op launches."""

comptime DQ_MIN_M = 128
"""Minimum M to route prefill through the dequant-once half-precision GEMM. Below this
the fused int4 GEMM wins: dequant-once pays ~4.5× the weight bytes (0.5 read +
2 write + 2 re-read vs 0.5 fused), which only amortizes once the GEMM is
firmly compute-bound. Crossover re-measured post-ktail-fix on M4
(.scratch/dq_crossover.mojo): parity at M=128, +10-17% at 256, +30-41% at 512+."""

comptime DevBuf = DeviceBuffer[DType.float32]
"""An f32 device buffer — the common activation/working type."""
comptime WBuf = DeviceBuffer[
    DType.uint16
]  # bf16 weights kept on-device as raw u16
"""A u16 device buffer holding raw bf16 weights (widened per element)."""
comptime KV_DTYPE = DType.float32
"""THE KV-cache dtype switch — KVBuf and every KV-kernel instantiation follow it.
f16 was MEASURED AND REVERTED (2026-07-13, M4): decode attention is scan-latency
bound, not bandwidth bound (KV is ~4% of decode traffic at 1570 ctx), so f16 cost
more in widen-casts than it saved in bytes — long-code decode 15.4 -> 12.2 tok/s.
Prefill LIKED it (tc_attn 5.65 -> 5.17 s), so revisit if the caches ever split."""
comptime KVBuf = DeviceBuffer[KV_DTYPE]
"""An f16 device buffer — the KV-cache type (half the attention bandwidth
and half the resident bytes of f32; MLX ships f16 KV as default)."""
comptime PBuf = DeviceBuffer[
    DType.uint32
]  # packed group-128 int4 weights (8 nibbles/word)
"""A u32 device buffer of packed group-128 int4 weights (8 nibbles/word)."""


struct QMat(Movable):
    """A projection weight in *either* representation: bf16 (`q4=False`, uses
    `bf16`) or group-128 int4 (`q4=True`, uses `packed`+`scales`). The unused
    representation holds a size-1 dummy buffer so the struct stays optional-free.
    mm() dispatches on `q4`, so the bf16 path is byte-for-byte unchanged and a
    model can mix (here: bf16 0.5B, int4 3B; either is selectable at load)."""

    var bf16: WBuf
    """Raw bf16 weights (used when `q4` is False); else a size-1 dummy."""
    var packed: PBuf
    """Packed group-128 int4 weights (used when `q4` is True); else a dummy."""
    var scales: DevBuf
    """Per-group f32 scales for the int4 weights (used when `q4` is True)."""
    var q4: Bool
    """Selects the representation: True = group-128 int4, False = bf16."""

    def __init__(
        out self, var bf16: WBuf, var packed: PBuf, var scales: DevBuf, q4: Bool
    ):
        """Construct a QMat taking ownership of its three buffers and `q4` flag.

        Args:
            bf16: Raw bf16 weight buffer (used when `q4` is False; else a dummy).
            packed: Packed group-128 int4 weight buffer (used when `q4` is True).
            scales: Per-group f32 scales for the int4 weights.
            q4: Representation flag — True = group-128 int4, False = bf16.
        """
        self.bf16 = bf16^
        self.packed = packed^
        self.scales = scales^
        self.q4 = q4


def download_f32(mut buf: DevBuf, n: Int) raises -> List[Float32]:
    """The first `n` f32 elements of `buf`, copied to the host in ONE memcpy.

    The previous idiom — a per-element `append(mt[i])` loop over the mapped
    buffer — cost milliseconds per decode step at vocab size (151936 appends
    with a TileTensor index each) and dominated the lm_head+logits phase once
    the MAX 26.5 driver batched kernel dispatch. Callers must `synchronize()`
    first; `map_to_host` only guarantees the mapping, not kernel completion.

    Args:
        buf: The device buffer to copy from.
        n: Number of leading f32 elements to copy.

    Returns:
        A host `List[Float32]` of length `n`.

    Raises:
        If mapping the buffer to the host fails.
    """
    var out = List[Float32](length=n, fill=0.0)
    with buf.map_to_host() as m:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=m.unsafe_ptr(), count=n)
    return out^


def qmat_bf16(ctx: DeviceContext, var buf: WBuf) raises -> QMat:
    """Wrap raw bf16 weights in a QMat (q4=False), with dummy int4 buffers.

    Args:
        ctx: The GPU device context (allocates the size-1 dummy int4 buffers).
        buf: The raw bf16 weight buffer to take ownership of.

    Returns:
        A QMat holding `buf` with q4=False and size-1 dummy packed/scales buffers.

    Raises:
        If allocating the dummy device buffers fails.
    """
    return QMat(
        buf^,
        ctx.enqueue_create_buffer[DType.uint32](1),
        ctx.enqueue_create_buffer[DType.float32](1),
        False,
    )


# ── op launchers (each runs one kernel, returns a new device buffer) ───────────


def mm(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut w: WBuf,
    mut b: DevBuf,
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """Matmul x[M,K]·wᵀ[N,K] (+ optional bias) → y[M,N] for bf16 weights. Dispatches
    on M: GEMV at M=1 (decode), simdgroup-matrix GEMM when `simd_ok` (prefill),
    else the scalar register-tiled GEMM.

    Args:
        ctx: The GPU device context.
        x: Input activations [M, K].
        w: Bf16 weights [N, K] (transposed in the matmul).
        b: Bias [N] (read only when `use_bias` is nonzero).
        M: Number of input rows (tokens).
        K: Input / contraction dimension.
        N: Number of output columns.
        use_bias: Nonzero to add `b`, zero to skip it.
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding x·wᵀ (+ bias).

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    if M == 1:
        # decode: memory-bound GEMV, one warp per output element (M*N warps). The
        # simdgroup-matrix path is for prefill only — at M=1 its 8-row tiles waste
        # 7/8 of every fragment, so decode always uses the GEMV.
        comptime k = matmul_kernel[type_of(lay)]
        cached_enqueue[k](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(use_bias),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif simd_ok:
        # prefill, fast path: simdgroup-matrix GEMM (~4.5× the scalar tiled kernel
        # on the M4). Gated by the startup probe; the scalar path below is the
        # fallback if this toolchain rejects the AIR intrinsics.
        comptime ks = matmul_simd_kernel[type_of(lay)]
        cached_enqueue[ks](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(use_bias),
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
    else:
        # prefill, scalar fallback: 2D register-tiled GEMM, one warp per (CN-column,
        # TM-token) block, so each weight is reused across TM tokens and each X value
        # across CN columns — cutting the dominant X traffic CN-fold (§11 #12). TM=CN=8
        # measured ~2× a token-only tiling (~210 GFLOP/s) on the M4.
        comptime TM = 8
        comptime CN = 8
        comptime kt = matmul_tiled_kernel[type_of(lay), TM, CN]
        cached_enqueue[kt](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(use_bias),
            grid_dim=ceildiv(
                ceildiv(N, CN) * ceildiv(M, TM) * WARP_SIZE, BLOCK
            ),
            block_dim=BLOCK,
        )
    return y^


def mm_w(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut w: QMat,
    mut b: DevBuf,
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """Mm() for a QMat weight: bf16 path (delegates to mm) or group-128 int4. The
    int4 dispatch mirrors mm — GEMV at M=1 (decode), simdgroup-matrix GEMM at
    M>1 with the probe on (prefill), scalar-tiled fallback otherwise.

    Args:
        ctx: The GPU device context.
        x: Input activations [M, K].
        w: The projection weight (bf16 or group-128 int4).
        b: Bias [N] (read only when `use_bias` is nonzero).
        M: Number of input rows (tokens).
        K: Input / contraction dimension.
        N: Number of output columns.
        use_bias: Nonzero to add `b`, zero to skip it.
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding x·wᵀ (+ bias).

    Raises:
        If buffer allocation or kernel launch fails.
    """
    if not w.q4:
        return mm(ctx, x, w.bf16, b, M, K, N, use_bias, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    var NG = K // Q4_GROUP
    var xt = TileTensor(x, row_major(M * K))
    var pt = TileTensor(w.packed, row_major(N * K // 8))
    var st = TileTensor(w.scales, row_major(N * NG))
    var bt = TileTensor(b, row_major(N if use_bias != 0 else 1))
    var yt = TileTensor(y, lay)
    if M == 1:
        # Decode GEMV: plain one-warp-per-output. HISTORY: pre-MAX-26.5 this
        # was shape-routed to K-split rw variants (short-K shapes starved for
        # warps in flight: 2048×11008 51.9→92.9 GB/s on the old driver).
        # RE-TUNE 2026-08-07: the 26.5 driver's command-buffer batching gives
        # that concurrency for free and the split's cross-warp reduction makes
        # it LOSE at every decode shape (qkv 18→38 GB/s, gate_up 21→47,
        # lm_head 21→46; .scratch/layer_variants.mojo + lmhead_variants.mojo).
        # The rw kernels stay in src/kernels for the next driver-era re-tune.
        comptime k = matmul_q4_kernel[type_of(lay)]
        cached_enqueue[k](
            ctx,
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif M >= SPEC_SMALL_MIN and M <= SPEC_MAX_M and simd_ok:
        # Mid-small M (≈5..8, larger speculative verify): dedicated int4 GEMM with a
        # single 8-row MMA tile — MMA-efficient like the prefill GEMM but without
        # its wasted 56-row padding at small M. Flat ~M-independent here and well
        # below both the batched GEMV (linear, loses past Q≈5) and the 64-row GEMM.
        comptime ks = matmul_q4_small_kernel[type_of(lay)]
        cached_enqueue[ks](
            ctx,
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=(ceildiv(N, _SM_BN), 1),
            block_dim=_SM_TPB,
        )
    elif M <= SPEC_MAX_M:
        # Tiny M (2..4) — or no simdgroup intrinsic: batched GEMV (weights read once,
        # M rows accumulated in registers); cheapest at the smallest batch sizes.
        comptime kb = matmul_q4_batch_kernel[type_of(lay)]
        cached_enqueue[kb](
            ctx,
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif simd_ok and M >= DQ_MIN_M:
        # Large-M prefill: dequant-once half-precision pipeline (MAX linalg bake-off
        # 2026-08-07 — bf16 MMA operands + contiguous [K,N] B-side vector
        # loads reach ~3.0 TFLOP/s effective incl. the dequant + X cast, vs
        # 2.15 for the fused int4 GEMM below; +37% at M=1570). Scratch is
        # transient per call (stream-ordered free; largest = gate_up 90 MB).
        var wdq = ctx.enqueue_create_buffer[DType.float16](K * N)
        var xb = ctx.enqueue_create_buffer[DType.float16](M * K)
        var wdt = TileTensor(wdq, row_major(K * N))
        var xbt = TileTensor(xb, row_major(M * K))
        comptime kd = dequant_q4_ht_kernel[type_of(lay)]
        cached_enqueue[kd](
            ctx,
            pt,
            st,
            wdt,
            Int32(K),
            Int32(N),
            Int32(NG),
            grid_dim=(ceildiv(N, 32), ceildiv(K // 8, 4)),
            block_dim=(32, 4),
        )
        comptime kc = cast_f32_h_kernel[type_of(lay)]
        cached_enqueue[kc](
            ctx,
            xt,
            xbt,
            Int32(M * K),
            grid_dim=ceildiv(ceildiv(M * K, 4), BLOCK),
            block_dim=BLOCK,
        )
        # WIDE (32×128) when ceildiv(M,32) is odd — saves half an M-tile of
        # edge waste (bit-identical; 5-16% on 87-250-token suffixes/chunks,
        # neutral on even parity — small-M race 2026-08-08).
        if (ceildiv(M, 32) & 1) == 1:
            comptime kgw = matmul_hkn_kernel[type_of(lay), True]
            cached_enqueue[kgw](
                ctx,
                xbt,
                wdt,
                bt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                Int32(use_bias),
                grid_dim=(ceildiv(N, 128), ceildiv(M, 32)),
                block_dim=SG_TPB,
            )
        else:
            comptime kg = matmul_hkn_kernel[type_of(lay)]
            cached_enqueue[kg](
                ctx,
                xbt,
                wdt,
                bt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                Int32(use_bias),
                grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
                block_dim=SG_TPB,
            )
    elif simd_ok:
        if (ceildiv(M, 32) & 1) == 1:
            comptime ksw = matmul_simd_q4_kernel[type_of(lay), True]
            cached_enqueue[ksw](
                ctx,
                xt,
                pt,
                st,
                bt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                Int32(NG),
                Int32(use_bias),
                grid_dim=(ceildiv(N, 128), ceildiv(M, 32)),
                block_dim=SG_TPB,
            )
        else:
            comptime ks = matmul_simd_q4_kernel[type_of(lay)]
            cached_enqueue[ks](
                ctx,
                xt,
                pt,
                st,
                bt,
                yt,
                Int32(M),
                Int32(K),
                Int32(N),
                Int32(NG),
                Int32(use_bias),
                grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
                block_dim=SG_TPB,
            )
    else:
        comptime TM = 8
        comptime CN = 8
        comptime kt = matmul_tiled_q4_kernel[type_of(lay), TM, CN]
        cached_enqueue[kt](
            ctx,
            xt,
            pt,
            st,
            bt,
            yt,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(
                ceildiv(N, CN) * ceildiv(M, TM) * WARP_SIZE, BLOCK
            ),
            block_dim=BLOCK,
        )
    return y^


def mm_w_add(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut w: QMat,
    mut b: DevBuf,
    mut resid: DevBuf,
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """Y = x·Wᵀ(+bias) + resid. At decode (M=1) the residual add is fused into the
    proj GEMV epilogue (one launch instead of GEMV+add); prefill (M>1) falls back
    to mm_w + add (the simd GEMM has no residual variant), so prefill is unchanged.

    Args:
        ctx: The GPU device context.
        x: Input activations [M, K].
        w: The projection weight (bf16 or group-128 int4).
        b: Bias [N] (read only when `use_bias` is nonzero).
        resid: Residual [M, N] added to the matmul output.
        M: Number of input rows (tokens).
        K: Input / contraction dimension.
        N: Number of output columns.
        use_bias: Nonzero to add `b`, zero to skip it.
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding x·wᵀ (+ bias) + resid.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    if M != 1:
        var y0 = mm_w(ctx, x, w, b, M, K, N, use_bias, simd_ok)
        return add(ctx, resid, y0, M * N)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    var bt = TileTensor(b, row_major(N if use_bias != 0 else 1))
    if w.q4:
        var NG = K // Q4_GROUP
        # Shape-routed like mm_w's decode GEMV: short-K rows starve for warps
        # (the o-proj square is the 1.37× case); long-K stays on the shipping
        # kernel, which already saturates there.
        # Decode resid-fused GEMV: plain one-warp-per-output (see the
        # mm_w_norm re-tune note — K-split loses at decode shapes under the
        # MAX 26.5 driver).
        comptime kq = matmul_q4_resid_kernel[type_of(lay)]
        cached_enqueue[kq](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(w.packed, row_major(N * K // 8)),
            TileTensor(w.scales, row_major(N * NG)),
            bt,
            TileTensor(resid, lay),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime kb = matmul_resid_kernel[type_of(lay)]
        cached_enqueue[kb](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(w.bf16, row_major(N * K)),
            bt,
            TileTensor(resid, lay),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(use_bias),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    return y^


def mm_norm(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut lnw: DevBuf,
    mut w: WBuf,
    mut b: DevBuf,
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """RMSNorm(x)·Wᵀ for a bf16 weight, with the norm fused into the GEMV at decode
    (M=1) — one launch instead of rmsnorm+matmul. Prefill (M>1) falls back to a
    separate rmsnorm then mm (the GEMM has no fused-norm variant).

    Args:
        ctx: The GPU device context.
        x: Input activations [M, K].
        lnw: RMSNorm weight [K] applied before the projection.
        w: Bf16 weights [N, K] (transposed in the matmul).
        b: Bias [N] (read only when `use_bias` is nonzero).
        M: Number of input rows (tokens).
        K: Input / contraction dimension.
        N: Number of output columns.
        use_bias: Nonzero to add `b`, zero to skip it.
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding RMSNorm(x)·wᵀ (+ bias).

    Raises:
        If buffer allocation or kernel launch fails.
    """
    if M != 1:
        var xn = rmsnorm(ctx, x, lnw, M, K)
        return mm(ctx, xn, w, b, M, K, N, use_bias, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    comptime k = matmul_norm_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, row_major(M * K)),
        TileTensor(lnw, row_major(K)),
        TileTensor(w, row_major(N * K)),
        TileTensor(b, row_major(N if use_bias != 0 else 1)),
        TileTensor(y, lay),
        Int32(M),
        Int32(K),
        Int32(N),
        Int32(use_bias),
        grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def mm_w_norm(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut lnw: DevBuf,
    mut w: QMat,
    mut b: DevBuf,
    M: Int,
    K: Int,
    N: Int,
    use_bias: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """Mm_norm() for a QMat weight: fused-norm int4 GEMV at decode (M=1), else
    rmsnorm + mm_w at prefill. Folds the pre-projection RMSNorm into qkv/gate_up.

    Args:
        ctx: The GPU device context.
        x: Input activations [M, K].
        lnw: RMSNorm weight [K] applied before the projection.
        w: The projection weight (bf16 or group-128 int4).
        b: Bias [N] (read only when `use_bias` is nonzero).
        M: Number of input rows (tokens).
        K: Input / contraction dimension.
        N: Number of output columns.
        use_bias: Nonzero to add `b`, zero to skip it.
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding RMSNorm(x)·wᵀ (+ bias).

    Raises:
        If buffer allocation or kernel launch fails.
    """
    if not w.q4:
        return mm_norm(ctx, x, lnw, w.bf16, b, M, K, N, use_bias, simd_ok)
    if M != 1:
        var xn = rmsnorm(ctx, x, lnw, M, K)
        return mm_w(ctx, xn, w, b, M, K, N, use_bias, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    var NG = K // Q4_GROUP
    # Shape-routed like mm_w's decode GEMV. The norm-fused GEMV carries decode's
    # qkv AND the merged gate_up — gate_up (K=hidden, N=2·inter) is exactly the
    # short-K/huge-N shape where K-splitting measured 1.79× (92.9 GB/s on M4).
    # Decode norm-fused GEMV: plain one-warp-per-output. RE-TUNE 2026-08-07
    # under the MAX 26.5 driver: the K-split rw variants' win was hiding
    # dispatch gaps, which the driver's command-buffer batching now does for
    # free, leaving only their cross-warp reduction overhead — plain measured
    # faster at EVERY decode shape (lm_head N=151936: 3.4 ms vs 7.4; qkv
    # N=2560: 38 vs 18 GB/s; gate_up N=22016: 47 vs 21 GB/s; benches in
    # .scratch/lmhead_variants.mojo + layer_variants.mojo). The rw kernels
    # stay in src/kernels for the next driver-era re-tune.
    if N >= 4096:
        # Big-N decode (gate_up, LM head): UNFUSED mrw streams real weights at
        # ~90 GB/s vs the plain kernel's ~35; folding the norm INTO the mrw
        # loop costs 2.2x (extra x-side streams serialize its load pipeline),
        # so RMSNorm runs as its own tiny launch (~0.14 ms) instead.
        var xn = rmsnorm(ctx, x, lnw, M, K)
        comptime km = matmul_q4_mrw_kernel[type_of(lay), 2, 4]
        cached_enqueue[km](
            ctx,
            TileTensor(xn, row_major(M * K)),
            TileTensor(w.packed, row_major(N * K // 8)),
            TileTensor(w.scales, row_major(N * NG)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)),
            TileTensor(y, lay),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(N, 2),
            block_dim=4 * WARP_SIZE,
        )
    else:
        # Small-N decode (qkv, o): launch-latency-bound — plain wins.
        comptime kv_ = matmul_q4_norm_kernel[type_of(lay)]
        cached_enqueue[kv_](
            ctx,
            TileTensor(x, row_major(M * K)),
            TileTensor(lnw, row_major(K)),
            TileTensor(w.packed, row_major(N * K // 8)),
            TileTensor(w.scales, row_major(N * NG)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)),
            TileTensor(y, lay),
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(NG),
            Int32(use_bias),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    return y^


def mm_w_silu_add(
    ctx: DeviceContext,
    mut gu: DevBuf,
    mut w: QMat,
    mut resid: DevBuf,
    M: Int,
    inter: Int,
    N: Int,
    simd_ok: Bool = False,
) raises -> DevBuf:
    """Down-proj with SwiGLU fused on the input + residual on the output, at decode
    (M=1): Y = silu(gate)·up · Wᵀ + resid, reading the fused gate|up buffer directly
    — drops the silu_mul_cat launch and its `act` buffer. Prefill (M>1) falls back
    to silu_mul_cat + mm_w_add (unchanged).

    Args:
        ctx: The GPU device context.
        gu: The fused gate|up activations [M, 2*inter].
        w: The down-projection weight (bf16 or group-128 int4) [N, inter].
        resid: Residual [M, N] added to the down-projection output.
        M: Number of input rows (tokens).
        inter: The intermediate (gate/up) dimension.
        N: Number of output columns (the hidden dimension).
        simd_ok: True to allow the simdgroup-matrix GEMM at M>1 (prefill).

    Returns:
        A fresh [M, N] f32 buffer holding silu(gate)·up · wᵀ + resid.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    if M != 1:
        var act = silu_mul_cat(ctx, gu, M, inter)
        var dummy = ctx.enqueue_create_buffer[DType.float32](1)
        return mm_w_add(ctx, act, w, dummy, resid, M, inter, N, 0, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    if w.q4:
        # Long-K down-proj: UNFUSED mrw streams real weights at ~72 GB/s vs
        # ~35 plain; the SwiGLU folds cost 2.2x inside mrw's load pipeline,
        # so silu(gate)·up runs as its own tiny launch and the residual rides
        # the mrw kernel's bias input (exact: bias is += B[n]).
        var NG = inter // Q4_GROUP
        var act = silu_mul_cat(ctx, gu, M, inter)
        comptime kq = matmul_q4_mrw_kernel[type_of(lay), 2, 4]
        cached_enqueue[kq](
            ctx,
            TileTensor(act, row_major(M * inter)),
            TileTensor(w.packed, row_major(N * inter // 8)),
            TileTensor(w.scales, row_major(N * NG)),
            TileTensor(resid, lay),
            TileTensor(y, lay),
            Int32(inter),
            Int32(N),
            Int32(NG),
            Int32(1),
            grid_dim=ceildiv(N, 2),
            block_dim=4 * WARP_SIZE,
        )
    else:
        comptime kb = matmul_silu_resid_kernel[type_of(lay)]
        cached_enqueue[kb](
            ctx,
            TileTensor(gu, row_major(M * 2 * inter)),
            TileTensor(w.bf16, row_major(N * inter)),
            TileTensor(resid, lay),
            TileTensor(y, lay),
            Int32(M),
            Int32(inter),
            Int32(N),
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    return y^


def probe_simd_gemm(ctx: DeviceContext) raises -> Bool:
    """Runtime capability gate for the simdgroup-matrix GEMM. Runs a tiny
    matmul_simd_kernel and checks it against a CPU reference. Returns False — so
    mm() uses the scalar fallback — if this Metal toolchain rejects the AIR
    intrinsics: that surfaces as a catchable pipeline-state error (not a crash),
    and the DeviceContext stays usable afterward.

    Args:
        ctx: The GPU device context the probe kernel runs on.

    Returns:
        True if the simdgroup-matrix GEMM ran and matched the CPU reference;
        False if the toolchain rejected it or the result was wrong.

    Raises:
        Never — every device error is caught internally and reported as False.
    """
    try:
        var M = 8
        var K = 16
        var N = 8
        var xb = ctx.enqueue_create_buffer[DType.float32](M * K)
        var wb = ctx.enqueue_create_buffer[DType.uint16](N * K)
        var bb = ctx.enqueue_create_buffer[DType.float32](1)
        var yb = ctx.enqueue_create_buffer[DType.float32](M * N)
        var hx = List[Float32]()
        for i in range(M * K):
            hx.append(Float32((i * 3) % 7) * 0.25 - 0.75)
        with xb.map_to_host() as h:
            for i in range(M * K):
                h[i] = hx[i]
        var hw = List[Float32]()  # bf16-truncated weight values (host ref)
        with wb.map_to_host() as h:
            for i in range(N * K):
                var f = Float32((i * 2) % 5) * 0.5 - 1.0
                var bits = Pointer(to=f).unsafe_bitcast[UInt32]()[
                    unsafe_offset=0
                ]
                var top = UInt16(bits >> 16)
                h[i] = top
                var re: UInt32 = UInt32(top) << 16
                hw.append(
                    Pointer(to=re).unsafe_bitcast[Float32]()[unsafe_offset=0]
                )
        var lay = row_major(M * N)
        var xt = TileTensor(xb, row_major(M * K))
        var wt = TileTensor(wb, row_major(N * K))
        var bt = TileTensor(bb, row_major(1))
        var yt = TileTensor(yb, lay)
        comptime ks = matmul_simd_kernel[type_of(lay)]
        cached_enqueue[ks](
            ctx,
            xt,
            wt,
            bt,
            yt,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(0),
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)),
            block_dim=SG_TPB,
        )
        ctx.synchronize()
        var ok = True
        with yb.map_to_host() as h:
            for m in range(M):
                for n in range(N):
                    var acc = Float32(0.0)
                    for k in range(K):
                        acc += hx[m * K + k] * hw[n * K + k]
                    var e = h[m * N + n] - acc
                    if e < 0:
                        e = -e
                    if e > 1.0e-3:
                        ok = False
        return ok
    except:
        return False


def rmsnorm(
    ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, T: Int, dim: Int
) raises -> DevBuf:
    """RMSNorm each of the T rows of x[T,dim] scaled by weight w[dim], one warp per
    row → fresh [T,dim] buffer.

    Args:
        ctx: The GPU device context.
        x: Input activations [T, dim].
        w: The RMSNorm scale weight [dim].
        T: Number of rows (tokens).
        dim: The row width.

    Returns:
        A fresh [T, dim] f32 buffer holding the normalized, scaled rows.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](T * dim)
    var lay = row_major(T * dim)
    comptime k = rmsnorm_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, lay),
        TileTensor(w, row_major(dim)),
        TileTensor(y, lay),
        Int32(T),
        Int32(dim),
        grid_dim=ceildiv(T * WARP_SIZE, BLOCK),
        block_dim=BLOCK,  # one warp per row
    )
    return y^


def rmsnorm_add(
    ctx: DeviceContext,
    mut x: DevBuf,
    mut w: DevBuf,
    mut resid: DevBuf,
    T: Int,
    dim: Int,
    scale: Float32 = 1.0,
) raises -> DevBuf:
    """(RMSNorm(x)·w + resid)·scale in one launch — fuses Gemma's post-proj norm +
    residual add (+ optional layer scalar).

    Args:
        ctx: The GPU device context.
        x: Input activations [T, dim].
        w: The RMSNorm scale weight [dim].
        resid: Residual [T, dim] added after the norm.
        T: Number of rows (tokens).
        dim: The row width.
        scale: Optional per-layer scalar applied to the summed result.

    Returns:
        A fresh [T, dim] f32 buffer holding (RMSNorm(x)·w + resid)·scale.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](T * dim)
    var lay = row_major(T * dim)
    comptime k = rmsnorm_add_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, lay),
        TileTensor(w, row_major(dim)),
        TileTensor(resid, lay),
        TileTensor(y, lay),
        Int32(T),
        Int32(dim),
        scale,
        grid_dim=ceildiv(T * WARP_SIZE, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def gelu_mul_strided(
    ctx: DeviceContext,
    mut a: DevBuf,
    mut p: DevBuf,
    T: Int,
    n: Int,
    stride: Int,
    off: Int,
) raises -> DevBuf:
    """Gelu(a[t,j])·p[t, off+j] — PLE gate fused with the strided per-layer-input slice.

    Args:
        ctx: The GPU device context.
        a: The gate input [T, n] passed through gelu.
        p: The strided source [T, stride] sliced at column offset `off`.
        T: Number of rows (tokens).
        n: The output (gate) width.
        stride: The row stride of `p`.
        off: The column offset into each row of `p`.

    Returns:
        A fresh [T, n] f32 buffer holding gelu(a)·p[:, off:off+n].

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](T * n)
    var lay = row_major(T * n)
    comptime k = gelu_mul_strided_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(a, lay),
        TileTensor(p, row_major(T * stride)),
        TileTensor(y, lay),
        Int32(T),
        Int32(n),
        Int32(stride),
        Int32(off),
        grid_dim=ceildiv(T * n, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def add(
    ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int
) raises -> DevBuf:
    """Elementwise a + b over n elements → fresh buffer.

    Args:
        ctx: The GPU device context.
        a: The first operand [n].
        b: The second operand [n].
        n: Number of elements.

    Returns:
        A fresh [n] f32 buffer holding a + b.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = add_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(a, lay),
        TileTensor(b, lay),
        TileTensor(y, lay),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def silu_mul(
    ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int
) raises -> DevBuf:
    """SwiGLU elementwise SiLU(a)·b over n elements → fresh buffer.

    Args:
        ctx: The GPU device context.
        a: The gate operand [n] passed through SiLU.
        b: The up operand [n].
        n: Number of elements.

    Returns:
        A fresh [n] f32 buffer holding SiLU(a)·b.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = silu_mul_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(a, lay),
        TileTensor(b, lay),
        TileTensor(y, lay),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def silu_mul_cat(
    ctx: DeviceContext, mut gu: DevBuf, T: Int, inter: Int
) raises -> DevBuf:
    """SwiGLU on the fused gate+up GEMV output [T, 2*inter] → [T, inter].

    Args:
        ctx: The GPU device context.
        gu: The fused gate|up activations [T, 2*inter].
        T: Number of rows (tokens).
        inter: The intermediate (gate/up) dimension.

    Returns:
        A fresh [T, inter] f32 buffer holding SiLU(gate)·up.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](T * inter)
    var lay = row_major(T * inter)
    comptime k = silu_mul_cat_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(gu, row_major(T * 2 * inter)),
        TileTensor(y, lay),
        Int32(T),
        Int32(inter),
        grid_dim=ceildiv(T * inter, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def gelu_mul_cat(
    ctx: DeviceContext, mut gu: DevBuf, T: Int, inter: Int
) raises -> DevBuf:
    """GeGLU on the fused gate+up GEMV output [T, 2*inter] → [T, inter] (Gemma).

    Args:
        ctx: The GPU device context.
        gu: The fused gate|up activations [T, 2*inter].
        T: Number of rows (tokens).
        inter: The intermediate (gate/up) dimension.

    Returns:
        A fresh [T, inter] f32 buffer holding gelu(gate)·up.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](T * inter)
    var lay = row_major(T * inter)
    comptime k = gelu_mul_cat_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(gu, row_major(T * 2 * inter)),
        TileTensor(y, lay),
        Int32(T),
        Int32(inter),
        grid_dim=ceildiv(T * inter, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def gelu_mul(
    ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int
) raises -> DevBuf:
    """Y = gelu_tanh(a)·b over two separate [n] buffers (Gemma3n PLE gate).

    Args:
        ctx: The GPU device context.
        a: The gate operand [n] passed through gelu_tanh.
        b: The up operand [n].
        n: Number of elements.

    Returns:
        A fresh [n] f32 buffer holding gelu_tanh(a)·b.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = gelu_mul_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(a, lay),
        TileTensor(b, lay),
        TileTensor(y, lay),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def nll_gather(
    ctx: DeviceContext,
    mut logits: DevBuf,
    targets: List[Int],
    n: Int,
    vocab: Int,
) raises -> List[Float32]:
    """Per-position log P(target) from [n×vocab] GPU logits — one GPU pass, returns
    n host floats (no n×vocab host copy). For perplexity / echo logprobs.

    Args:
        ctx: The GPU device context.
        logits: The [n, vocab] logits buffer.
        targets: The n target token ids (one per position).
        n: Number of positions.
        vocab: The vocabulary size (row width of `logits`).

    Returns:
        A host List of n floats — log P(target) at each position.

    Raises:
        If buffer allocation, host mapping, or kernel launch fails.
    """
    var tgt = ctx.enqueue_create_buffer[DType.int32](n)
    with tgt.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(targets[i]))
    var out = ctx.enqueue_create_buffer[DType.float32](n)
    var nlay = row_major(n)
    comptime k = nll_gather_kernel[type_of(nlay)]
    cached_enqueue[k](
        ctx,
        TileTensor(logits, row_major(n * vocab)),
        TileTensor(tgt, nlay),
        TileTensor(out, nlay),
        Int32(n),
        Int32(vocab),
        grid_dim=ceildiv(n * WARP_SIZE, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()
    var res = List[Float32]()
    with out.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            res.append(rebind[Scalar[DType.float32]](mt[i]))
    return res^


def softcap(ctx: DeviceContext, mut x: DevBuf, n: Int, cap: Float32) raises:
    """In-place logit soft-capping x ← cap·tanh(x/cap) (Gemma).

    Args:
        ctx: The GPU device context.
        x: The [n] buffer soft-capped in place.
        n: Number of elements.
        cap: The soft-cap magnitude.

    Raises:
        If the kernel launch fails.
    """
    var lay = row_major(n)
    comptime k = softcap_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, lay),
        Int32(n),
        cap,
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )


def add_scalar(ctx: DeviceContext, mut x: DevBuf, n: Int, c: Float32) raises:
    """In-place x ← x + c. Gemma bakes (1+w) into RMSNorm weights at load (c=1).

    Args:
        ctx: The GPU device context.
        x: The [n] buffer updated in place.
        n: Number of elements.
        c: The scalar added to each element.

    Raises:
        If the kernel launch fails.
    """
    var lay = row_major(n)
    comptime k = add_scalar_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, lay),
        Int32(n),
        c,
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )


def mul_scalar(
    ctx: DeviceContext, mut x: DevBuf, n: Int, c: Float32
) raises -> DevBuf:
    """Y = x * c. Gemma embedding ×√hidden and per-layer learned scalar.

    Args:
        ctx: The GPU device context.
        x: The input buffer [n].
        n: Number of elements.
        c: The scalar each element is multiplied by.

    Returns:
        A fresh [n] f32 buffer holding x * c.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = mul_scalar_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(x, lay),
        TileTensor(y, lay),
        Int32(n),
        c,
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def embed_tokens(
    ctx: DeviceContext,
    mut ids: DeviceBuffer[DType.int32],
    mut emb: WBuf,
    T: Int,
    hidden: Int,
    vocab: Int,
) raises -> DevBuf:
    """Gather the bf16 embedding rows for T token `ids` from emb[vocab,hidden],
    widening to a fresh f32 [T,hidden] buffer.

    Args:
        ctx: The GPU device context.
        ids: The T token ids to gather.
        emb: The bf16 embedding table [vocab, hidden].
        T: Number of tokens.
        hidden: The embedding (row) width.
        vocab: The vocabulary size (number of rows in `emb`).

    Returns:
        A fresh [T, hidden] f32 buffer of the gathered, widened embedding rows.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var h = ctx.enqueue_create_buffer[DType.float32](T * hidden)
    var lay = row_major(T * hidden)
    comptime k = embed_kernel[
        type_of(lay)
    ]  # dimension-agnostic (runtime T, hidden)
    cached_enqueue[k](
        ctx,
        TileTensor(ids, row_major(T)),
        TileTensor(emb, row_major(vocab * hidden)),
        TileTensor(h, lay),
        Int32(T),
        Int32(hidden),
        grid_dim=ceildiv(T * hidden, BLOCK),
        block_dim=BLOCK,
    )
    return h^


def last_row(
    ctx: DeviceContext, mut src: DevBuf, T: Int, dim: Int
) raises -> DevBuf:
    """Lift row T-1 (dim elements) of src[T,dim] into a fresh 1×dim buffer.

    Args:
        ctx: The GPU device context.
        src: The source buffer [T, dim].
        T: Number of rows (the last, T-1, is taken).
        dim: The row width.

    Returns:
        A fresh [dim] f32 buffer holding row T-1 of `src`.

    Raises:
        If buffer allocation or kernel launch fails.
    """
    var y = ctx.enqueue_create_buffer[DType.float32](dim)
    var lay = row_major(dim)
    comptime k = slice_row_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(src, row_major(T * dim)),
        TileTensor(y, lay),
        Int32((T - 1) * dim),
        Int32(dim),
        grid_dim=ceildiv(dim, BLOCK),
        block_dim=BLOCK,
    )
    return y^


def copy_into(
    ctx: DeviceContext,
    mut src: DevBuf,
    mut dst: DevBuf,
    dst_offset: Int,
    n: Int,
    dst_len: Int,
) raises:
    """Copy n elements from src into dst starting at `dst_offset` (dst has capacity
    `dst_len`).

    Args:
        ctx: The GPU device context.
        src: The source buffer (first n elements copied).
        dst: The destination buffer (capacity `dst_len`).
        dst_offset: The start index in `dst` to copy into.
        n: Number of elements to copy.
        dst_len: The total capacity of `dst`.

    Raises:
        If the kernel launch fails.
    """
    var lay = row_major(n)
    comptime k = copy_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(src, lay),
        TileTensor(dst, row_major(dst_len)),
        Int32(dst_offset),
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )


def copy_strided(
    ctx: DeviceContext,
    mut src: DevBuf,
    mut dst: DevBuf,
    T: Int,
    in_stride: Int,
    in_off: Int,
    dst_off: Int,
    n: Int,
    dst_len: Int,
) raises:
    """Copy a [T, n] slice out of a strided source (V part of a fused [q|k|v]) into
    dst[dst_off:] — V into its cache rows.

    Args:
        ctx: The GPU device context.
        src: The strided source buffer [T, in_stride].
        dst: The destination buffer (capacity `dst_len`).
        T: Number of rows.
        in_stride: The row stride of `src`.
        in_off: The column offset of the slice within each `src` row.
        dst_off: The start index in `dst` to copy into.
        n: The slice width (columns per row).
        dst_len: The total capacity of `dst`.

    Raises:
        If the kernel launch fails.
    """
    var lay = row_major(T * in_stride)
    comptime k = copy_strided_kernel[type_of(lay)]
    cached_enqueue[k](
        ctx,
        TileTensor(src, lay),
        TileTensor(dst, row_major(dst_len)),
        Int32(T),
        Int32(in_stride),
        Int32(in_off),
        Int32(dst_off),
        Int32(n),
        grid_dim=ceildiv(T * n, BLOCK),
        block_dim=BLOCK,
    )


# ── GPU token selection (argmax / top-k) ───────────────────────────────────────

comptime AMAX_BLOCKS = 64
"""Stage-1 grid for the argmax reductions (64 blocks × 256 threads = 512
warps → 1024 partial floats; plenty of occupancy for vocab-sized rows)."""

comptime AMAX_PARTS = AMAX_BLOCKS * (BLOCK // 32)
"""Number of per-warp (index, value) partial pairs stage 1 emits."""


@fieldwise_init
struct SelectBufs(Movable):
    """Reusable device scratch for token selection: the penalized work row,
    stage-1 partials, the (index, value) output slots, and the seen-token
    mask that carries the repetition-penalty context across decode steps."""

    var work: DevBuf
    """Penalized copy of the logits row (vocab f32; top-k mutates it)."""
    var partials: DevBuf
    """Stage-1 per-warp partials (2·AMAX_PARTS f32)."""
    var outb: DevBuf
    """Selection results, 2 f32 (index, value) per slot."""
    var mask: DeviceBuffer[DType.uint8]
    """Per-vocab seen mask for the repetition penalty (uint8)."""


def make_select_bufs(
    ctx: DeviceContext, vocab: Int, max_slots: Int
) raises -> SelectBufs:
    """Allocate `SelectBufs` for a `vocab`-wide model and ≤ `max_slots`
    selection slots (max of top-k k and the spec-verify row count)."""
    return SelectBufs(
        ctx.enqueue_create_buffer[DType.float32](vocab),
        ctx.enqueue_create_buffer[DType.float32](2 * AMAX_PARTS),
        ctx.enqueue_create_buffer[DType.float32](2 * max_slots),
        ctx.enqueue_create_buffer[DType.uint8](vocab),
    )


def mask_reset(
    ctx: DeviceContext,
    mut b: SelectBufs,
    mut ids: DeviceBuffer[DType.int32],
    n_ids: Int,
) raises:
    """Clear the seen mask and mark the `n_ids` prompt/context token ids."""
    b.mask.enqueue_fill(0)
    if n_ids == 0:
        return
    comptime km = mask_mark_kernel[type_of(row_major(1))]
    cached_enqueue[km](
        ctx,
        TileTensor(ids, row_major(n_ids)),
        TileTensor(b.mask, row_major(1)),
        Int32(n_ids),
        grid_dim=ceildiv(n_ids, BLOCK),
        block_dim=BLOCK,
    )


def _amax_round(
    ctx: DeviceContext,
    mut src: DevBuf,
    mut partials: DevBuf,
    mut outb: DevBuf,
    n: Int,
    base: Int,
    slot: Int,
    mask_out: Int,
) raises:
    """One part+merge argmax round over src[base .. base+n) into slot."""
    comptime kp = argmax_part_kernel[type_of(row_major(1))]
    cached_enqueue[kp](
        ctx,
        TileTensor(src, row_major(1)),
        TileTensor(partials, row_major(1)),
        Int32(n),
        Int32(base),
        grid_dim=AMAX_BLOCKS,
        block_dim=BLOCK,
    )
    comptime km = argmax_merge_kernel[type_of(row_major(1))]
    cached_enqueue[km](
        ctx,
        TileTensor(partials, row_major(1)),
        TileTensor(src, row_major(1)),
        TileTensor(outb, row_major(1)),
        Int32(AMAX_PARTS),
        Int32(slot),
        Int32(mask_out),
        grid_dim=1,
        block_dim=32,
    )


def sel_amax_part(ctx: DeviceContext, mut b: SelectBufs, n: Int) raises:
    """One bare stage-1 argmax enqueue over the work row (diagnostics)."""
    comptime kp = argmax_part_kernel[type_of(row_major(1))]
    cached_enqueue[kp](
        ctx,
        TileTensor(b.work, row_major(1)),
        TileTensor(b.partials, row_major(1)),
        Int32(n),
        Int32(0),
        grid_dim=AMAX_BLOCKS,
        block_dim=BLOCK,
    )


def gpu_argmax(
    ctx: DeviceContext, mut logits: DevBuf, mut b: SelectBufs, n: Int
) raises -> Int:
    """Argmax of a device logits row: two tiny kernels + an 8-byte download
    (replaces downloading the whole row and scanning it on the host)."""
    _amax_round(ctx, logits, b.partials, b.outb, n, 0, 0, 0)
    ctx.synchronize()
    var pair = download_f32(b.outb, 2)
    return Int(pair[0])


def gpu_argmax_rows(
    ctx: DeviceContext, mut G: DevBuf, mut b: SelectBufs, rows: Int, n: Int
) raises -> List[Int]:
    """Per-row argmax of a [rows, n] device logits matrix (spec-decode verify):
    2·rows tiny kernels + one 8·rows-byte download."""
    for r in range(rows):
        _amax_round(ctx, G, b.partials, b.outb, n, r * n, r, 0)
    ctx.synchronize()
    var pairs = download_f32(b.outb, 2 * rows)
    var out = List[Int](capacity=rows)
    for r in range(rows):
        out.append(Int(pairs[2 * r]))
    return out^


def gpu_topk(
    ctx: DeviceContext,
    mut logits: DevBuf,
    mut b: SelectBufs,
    n: Int,
    k: Int,
    pen: Float32,
    last_id: Int,
) raises -> Tuple[List[Int], List[Float32]]:
    """Top-k (ids, PENALIZED logits) of a device logits row, descending —
    exactly `process_logits`' selection stage, on-device. The repetition
    penalty uses the persistent seen mask plus `last_id` (also marked into
    the mask for later steps); temperature/softmax/top-p stay on the host
    over just these k candidates (temperature is order-preserving, and the
    host softmax normalizes over the top-k only, so this is exact)."""
    comptime kc = penalize_copy_kernel[type_of(row_major(1))]
    cached_enqueue[kc](
        ctx,
        TileTensor(logits, row_major(1)),
        TileTensor(b.work, row_major(1)),
        TileTensor(b.mask, row_major(1)),
        Int32(n),
        Int32(0),
        pen,
        Int32(last_id),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    for slot in range(k):
        _amax_round(ctx, b.work, b.partials, b.outb, n, 0, slot, 1)
    ctx.synchronize()
    var pairs = download_f32(b.outb, 2 * k)
    var ids = List[Int](capacity=k)
    var vals = List[Float32](capacity=k)
    for s in range(k):
        ids.append(Int(pairs[2 * s]))
        vals.append(pairs[2 * s + 1])
    return (ids^, vals^)
