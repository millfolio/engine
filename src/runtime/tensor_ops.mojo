"""Model-agnostic tensor op launchers over src/kernels.mojo. Each function runs
one kernel and returns a fresh device buffer (or writes into a provided one).
The weight representation types (DevBuf/WBuf/PBuf, QMat) live here too, so both
the loader (safetensors.mojo) and the model can share them without a cycle.

No model-specific constants: every op takes its dims at the call site."""

from std.math import ceildiv
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import (
    matmul_kernel, matmul_simd_kernel, matmul_tiled_kernel,
    matmul_q4_kernel, matmul_q4_batch_kernel, matmul_q4_small_kernel,
    matmul_simd_q4_kernel, matmul_tiled_q4_kernel,
    matmul_resid_kernel, matmul_q4_resid_kernel,
    matmul_norm_kernel, matmul_q4_norm_kernel,
    matmul_silu_resid_kernel, matmul_q4_silu_resid_kernel,
    rmsnorm_kernel, add_kernel, silu_mul_kernel, silu_mul_cat_kernel,
    gelu_mul_cat_kernel, gelu_mul_kernel, gelu_mul_strided_kernel, rmsnorm_add_kernel,
    nll_gather_kernel, softcap_kernel, add_scalar_kernel, mul_scalar_kernel, vnorm_kernel,
    embed_kernel, slice_row_kernel, copy_kernel, copy_strided_kernel,
    SG_BM, SG_BN, SG_TPB, Q4_GROUP, SPEC_MAX_M, SPEC_SMALL_MIN, _SM_BN, _SM_TPB,
)

comptime BLOCK = 256

comptime DevBuf = DeviceBuffer[DType.float32]
comptime WBuf = DeviceBuffer[DType.uint16]   # bf16 weights kept on-device as raw u16
comptime PBuf = DeviceBuffer[DType.uint32]   # packed group-128 int4 weights (8 nibbles/word)


struct QMat(Movable):
    """A projection weight in *either* representation: bf16 (`q4=False`, uses
    `bf16`) or group-128 int4 (`q4=True`, uses `packed`+`scales`). The unused
    representation holds a size-1 dummy buffer so the struct stays optional-free.
    mm() dispatches on `q4`, so the bf16 path is byte-for-byte unchanged and a
    model can mix (here: bf16 0.5B, int4 3B; either is selectable at load)."""
    var bf16: WBuf
    var packed: PBuf
    var scales: DevBuf
    var q4: Bool

    def __init__(out self, var bf16: WBuf, var packed: PBuf, var scales: DevBuf, q4: Bool):
        self.bf16 = bf16^
        self.packed = packed^
        self.scales = scales^
        self.q4 = q4


def qmat_bf16(ctx: DeviceContext, var buf: WBuf) raises -> QMat:
    return QMat(buf^, ctx.enqueue_create_buffer[DType.uint32](1),
                ctx.enqueue_create_buffer[DType.float32](1), False)


# ── op launchers (each runs one kernel, returns a new device buffer) ───────────

def mm(ctx: DeviceContext, mut x: DevBuf, mut w: WBuf, mut b: DevBuf,
       M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    if M == 1:
        # decode: memory-bound GEMV, one warp per output element (M*N warps). The
        # simdgroup-matrix path is for prefill only — at M=1 its 8-row tiles waste
        # 7/8 of every fragment, so decode always uses the GEMV.
        comptime k = matmul_kernel[type_of(lay)]
        ctx.enqueue_function[k](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    elif simd_ok:
        # prefill, fast path: simdgroup-matrix GEMM (~4.5× the scalar tiled kernel
        # on the M4). Gated by the startup probe; the scalar path below is the
        # fallback if this toolchain rejects the AIR intrinsics.
        comptime ks = matmul_simd_kernel[type_of(lay)]
        ctx.enqueue_function[ks](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)), block_dim=SG_TPB,
        )
    else:
        # prefill, scalar fallback: 2D register-tiled GEMM, one warp per (CN-column,
        # TM-token) block, so each weight is reused across TM tokens and each X value
        # across CN columns — cutting the dominant X traffic CN-fold (§11 #12). TM=CN=8
        # measured ~2× a token-only tiling (~210 GFLOP/s) on the M4.
        comptime TM = 8
        comptime CN = 8
        comptime kt = matmul_tiled_kernel[type_of(lay), TM, CN]
        ctx.enqueue_function[kt](
            TileTensor(x, row_major(M * K)), TileTensor(w, row_major(N * K)),
            TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
            M, K, N, use_bias,
            grid_dim=ceildiv(ceildiv(N, CN) * ceildiv(M, TM) * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    return y^


def mm_w(ctx: DeviceContext, mut x: DevBuf, mut w: QMat, mut b: DevBuf,
         M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    """mm() for a QMat weight: bf16 path (delegates to mm) or group-128 int4. The
    int4 dispatch mirrors mm — GEMV at M=1 (decode), simdgroup-matrix GEMM at
    M>1 with the probe on (prefill), scalar-tiled fallback otherwise."""
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
        comptime k = matmul_q4_kernel[type_of(lay)]
        ctx.enqueue_function[k](xt, pt, st, bt, yt, M, K, N, NG, use_bias,
            grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK)
    elif M >= SPEC_SMALL_MIN and M <= SPEC_MAX_M and simd_ok:
        # Mid-small M (≈5..8, larger speculative verify): dedicated int4 GEMM with a
        # single 8-row MMA tile — MMA-efficient like the prefill GEMM but without
        # its wasted 56-row padding at small M. Flat ~M-independent here and well
        # below both the batched GEMV (linear, loses past Q≈5) and the 64-row GEMM.
        comptime ks = matmul_q4_small_kernel[type_of(lay)]
        ctx.enqueue_function[ks](xt, pt, st, bt, yt, M, K, N, NG, use_bias,
            grid_dim=(ceildiv(N, _SM_BN), 1), block_dim=_SM_TPB)
    elif M <= SPEC_MAX_M:
        # Tiny M (2..4) — or no simdgroup intrinsic: batched GEMV (weights read once,
        # M rows accumulated in registers); cheapest at the smallest batch sizes.
        comptime kb = matmul_q4_batch_kernel[type_of(lay)]
        ctx.enqueue_function[kb](xt, pt, st, bt, yt, M, K, N, NG, use_bias,
            grid_dim=ceildiv(N * WARP_SIZE, BLOCK), block_dim=BLOCK)
    elif simd_ok:
        comptime ks = matmul_simd_q4_kernel[type_of(lay)]
        ctx.enqueue_function[ks](xt, pt, st, bt, yt, M, K, N, NG, use_bias,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)), block_dim=SG_TPB)
    else:
        comptime TM = 8
        comptime CN = 8
        comptime kt = matmul_tiled_q4_kernel[type_of(lay), TM, CN]
        ctx.enqueue_function[kt](xt, pt, st, bt, yt, M, K, N, NG, use_bias,
            grid_dim=ceildiv(ceildiv(N, CN) * ceildiv(M, TM) * WARP_SIZE, BLOCK), block_dim=BLOCK)
    return y^

def mm_w_add(ctx: DeviceContext, mut x: DevBuf, mut w: QMat, mut b: DevBuf, mut resid: DevBuf,
             M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    """Y = x·Wᵀ(+bias) + resid. At decode (M=1) the residual add is fused into the
    proj GEMV epilogue (one launch instead of GEMV+add); prefill (M>1) falls back
    to mm_w + add (the simd GEMM has no residual variant), so prefill is unchanged."""
    if M != 1:
        var y0 = mm_w(ctx, x, w, b, M, K, N, use_bias, simd_ok)
        return add(ctx, resid, y0, M * N)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    var bt = TileTensor(b, row_major(N if use_bias != 0 else 1))
    if w.q4:
        var NG = K // Q4_GROUP
        comptime kq = matmul_q4_resid_kernel[type_of(lay)]
        ctx.enqueue_function[kq](
            TileTensor(x, row_major(M * K)), TileTensor(w.packed, row_major(N * K // 8)),
            TileTensor(w.scales, row_major(N * NG)), bt, TileTensor(resid, lay), TileTensor(y, lay),
            M, K, N, NG, use_bias, grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    else:
        comptime kb = matmul_resid_kernel[type_of(lay)]
        ctx.enqueue_function[kb](
            TileTensor(x, row_major(M * K)), TileTensor(w.bf16, row_major(N * K)), bt,
            TileTensor(resid, lay), TileTensor(y, lay),
            M, K, N, use_bias, grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    return y^


def mm_norm(ctx: DeviceContext, mut x: DevBuf, mut lnw: DevBuf, mut w: WBuf, mut b: DevBuf,
            M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    """RMSNorm(x)·Wᵀ for a bf16 weight, with the norm fused into the GEMV at decode
    (M=1) — one launch instead of rmsnorm+matmul. Prefill (M>1) falls back to a
    separate rmsnorm then mm (the GEMM has no fused-norm variant)."""
    if M != 1:
        var xn = rmsnorm(ctx, x, lnw, M, K)
        return mm(ctx, xn, w, b, M, K, N, use_bias, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    comptime k = matmul_norm_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, row_major(M * K)), TileTensor(lnw, row_major(K)),
        TileTensor(w, row_major(N * K)), TileTensor(b, row_major(N if use_bias != 0 else 1)),
        TileTensor(y, lay), M, K, N, use_bias,
        grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
    )
    return y^


def mm_w_norm(ctx: DeviceContext, mut x: DevBuf, mut lnw: DevBuf, mut w: QMat, mut b: DevBuf,
              M: Int, K: Int, N: Int, use_bias: Int, simd_ok: Bool = False) raises -> DevBuf:
    """mm_norm() for a QMat weight: fused-norm int4 GEMV at decode (M=1), else
    rmsnorm + mm_w at prefill. Folds the pre-projection RMSNorm into qkv/gate_up."""
    if not w.q4:
        return mm_norm(ctx, x, lnw, w.bf16, b, M, K, N, use_bias, simd_ok)
    if M != 1:
        var xn = rmsnorm(ctx, x, lnw, M, K)
        return mm_w(ctx, xn, w, b, M, K, N, use_bias, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    var NG = K // Q4_GROUP
    comptime k = matmul_q4_norm_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, row_major(M * K)), TileTensor(lnw, row_major(K)),
        TileTensor(w.packed, row_major(N * K // 8)), TileTensor(w.scales, row_major(N * NG)),
        TileTensor(b, row_major(N if use_bias != 0 else 1)), TileTensor(y, lay),
        M, K, N, NG, use_bias, grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
    )
    return y^


def mm_w_silu_add(ctx: DeviceContext, mut gu: DevBuf, mut w: QMat, mut resid: DevBuf,
                  M: Int, inter: Int, N: Int, simd_ok: Bool = False) raises -> DevBuf:
    """down-proj with SwiGLU fused on the input + residual on the output, at decode
    (M=1): Y = silu(gate)·up · Wᵀ + resid, reading the fused gate|up buffer directly
    — drops the silu_mul_cat launch and its `act` buffer. Prefill (M>1) falls back
    to silu_mul_cat + mm_w_add (unchanged)."""
    if M != 1:
        var act = silu_mul_cat(ctx, gu, M, inter)
        var dummy = ctx.enqueue_create_buffer[DType.float32](1)
        return mm_w_add(ctx, act, w, dummy, resid, M, inter, N, 0, simd_ok)
    var y = ctx.enqueue_create_buffer[DType.float32](M * N)
    var lay = row_major(M * N)
    if w.q4:
        var NG = inter // Q4_GROUP
        comptime kq = matmul_q4_silu_resid_kernel[type_of(lay)]
        ctx.enqueue_function[kq](
            TileTensor(gu, row_major(M * 2 * inter)), TileTensor(w.packed, row_major(N * inter // 8)),
            TileTensor(w.scales, row_major(N * NG)), TileTensor(resid, lay), TileTensor(y, lay),
            M, inter, N, NG, grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    else:
        comptime kb = matmul_silu_resid_kernel[type_of(lay)]
        ctx.enqueue_function[kb](
            TileTensor(gu, row_major(M * 2 * inter)), TileTensor(w.bf16, row_major(N * inter)),
            TileTensor(resid, lay), TileTensor(y, lay),
            M, inter, N, grid_dim=ceildiv(M * N * WARP_SIZE, BLOCK), block_dim=BLOCK,
        )
    return y^


def probe_simd_gemm(ctx: DeviceContext) raises -> Bool:
    """Runtime capability gate for the simdgroup-matrix GEMM. Runs a tiny
    matmul_simd_kernel and checks it against a CPU reference. Returns False — so
    mm() uses the scalar fallback — if this Metal toolchain rejects the AIR
    intrinsics: that surfaces as a catchable pipeline-state error (not a crash),
    and the DeviceContext stays usable afterward."""
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
        var hw = List[Float32]()        # bf16-truncated weight values (host ref)
        with wb.map_to_host() as h:
            for i in range(N * K):
                var f = Float32((i * 2) % 5) * 0.5 - 1.0
                var bits = UnsafePointer(to=f).bitcast[UInt32]()[0]
                var top = UInt16(bits >> 16)
                h[i] = top
                var re: UInt32 = UInt32(top) << 16
                hw.append(UnsafePointer(to=re).bitcast[Float32]()[0])
        var lay = row_major(M * N)
        var xt = TileTensor(xb, row_major(M * K))
        var wt = TileTensor(wb, row_major(N * K))
        var bt = TileTensor(bb, row_major(1))
        var yt = TileTensor(yb, lay)
        comptime ks = matmul_simd_kernel[type_of(lay)]
        ctx.enqueue_function[ks](
            xt, wt, bt, yt, M, K, N, 0,
            grid_dim=(ceildiv(N, SG_BN), ceildiv(M, SG_BM)), block_dim=SG_TPB,
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


def rmsnorm(ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, T: Int, dim: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](T * dim)
    var lay = row_major(T * dim)
    comptime k = rmsnorm_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, lay), TileTensor(w, row_major(dim)), TileTensor(y, lay), T, dim,
        grid_dim=ceildiv(T * WARP_SIZE, BLOCK), block_dim=BLOCK,   # one warp per row
    )
    return y^

def rmsnorm_add(ctx: DeviceContext, mut x: DevBuf, mut w: DevBuf, mut resid: DevBuf,
                T: Int, dim: Int, scale: Float32 = 1.0) raises -> DevBuf:
    """(RMSNorm(x)·w + resid)·scale in one launch — fuses Gemma's post-proj norm +
    residual add (+ optional layer scalar)."""
    var y = ctx.enqueue_create_buffer[DType.float32](T * dim)
    var lay = row_major(T * dim)
    comptime k = rmsnorm_add_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(x, lay), TileTensor(w, row_major(dim)), TileTensor(resid, lay),
        TileTensor(y, lay), T, dim, scale,
        grid_dim=ceildiv(T * WARP_SIZE, BLOCK), block_dim=BLOCK,
    )
    return y^

def gelu_mul_strided(ctx: DeviceContext, mut a: DevBuf, mut p: DevBuf,
                     T: Int, n: Int, stride: Int, off: Int) raises -> DevBuf:
    """gelu(a[t,j])·p[t, off+j] — PLE gate fused with the strided per-layer-input slice."""
    var y = ctx.enqueue_create_buffer[DType.float32](T * n)
    var lay = row_major(T * n)
    comptime k = gelu_mul_strided_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(p, row_major(T * stride)), TileTensor(y, lay),
        T, n, stride, off, grid_dim=ceildiv(T * n, BLOCK), block_dim=BLOCK,
    )
    return y^

def add(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = add_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def silu_mul(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = silu_mul_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def silu_mul_cat(ctx: DeviceContext, mut gu: DevBuf, T: Int, inter: Int) raises -> DevBuf:
    """SwiGLU on the fused gate+up GEMV output [T, 2*inter] → [T, inter]."""
    var y = ctx.enqueue_create_buffer[DType.float32](T * inter)
    var lay = row_major(T * inter)
    comptime k = silu_mul_cat_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(gu, row_major(T * 2 * inter)), TileTensor(y, lay), T, inter,
        grid_dim=ceildiv(T * inter, BLOCK), block_dim=BLOCK,
    )
    return y^

def gelu_mul_cat(ctx: DeviceContext, mut gu: DevBuf, T: Int, inter: Int) raises -> DevBuf:
    """GeGLU on the fused gate+up GEMV output [T, 2*inter] → [T, inter] (Gemma)."""
    var y = ctx.enqueue_create_buffer[DType.float32](T * inter)
    var lay = row_major(T * inter)
    comptime k = gelu_mul_cat_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(gu, row_major(T * 2 * inter)), TileTensor(y, lay), T, inter,
        grid_dim=ceildiv(T * inter, BLOCK), block_dim=BLOCK,
    )
    return y^

def gelu_mul(ctx: DeviceContext, mut a: DevBuf, mut b: DevBuf, n: Int) raises -> DevBuf:
    """Y = gelu_tanh(a)·b over two separate [n] buffers (Gemma3n PLE gate)."""
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = gelu_mul_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(a, lay), TileTensor(b, lay), TileTensor(y, lay), n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    return y^

def nll_gather(ctx: DeviceContext, mut logits: DevBuf, targets: List[Int],
               n: Int, vocab: Int) raises -> List[Float32]:
    """Per-position log P(target) from [n×vocab] GPU logits — one GPU pass, returns
    n host floats (no n×vocab host copy). For perplexity / echo logprobs."""
    var tgt = ctx.enqueue_create_buffer[DType.int32](n)
    with tgt.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(targets[i]))
    var out = ctx.enqueue_create_buffer[DType.float32](n)
    var nlay = row_major(n)
    comptime k = nll_gather_kernel[type_of(nlay)]
    ctx.enqueue_function[k](
        TileTensor(logits, row_major(n * vocab)), TileTensor(tgt, nlay),
        TileTensor(out, nlay), n, vocab,
        grid_dim=ceildiv(n * WARP_SIZE, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()
    var res = List[Float32]()
    with out.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            res.append(rebind[Scalar[DType.float32]](mt[i]))
    return res^

def softcap(ctx: DeviceContext, mut x: DevBuf, n: Int, cap: Float32) raises:
    """In-place logit soft-capping x ← cap·tanh(x/cap) (Gemma)."""
    var lay = row_major(n)
    comptime k = softcap_kernel[type_of(lay)]
    ctx.enqueue_function[k](TileTensor(x, lay), n, cap, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

def add_scalar(ctx: DeviceContext, mut x: DevBuf, n: Int, c: Float32) raises:
    """In-place x ← x + c. Gemma bakes (1+w) into RMSNorm weights at load (c=1)."""
    var lay = row_major(n)
    comptime k = add_scalar_kernel[type_of(lay)]
    ctx.enqueue_function[k](TileTensor(x, lay), n, c, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

def mul_scalar(ctx: DeviceContext, mut x: DevBuf, n: Int, c: Float32) raises -> DevBuf:
    """y = x * c. Gemma embedding ×√hidden and per-layer learned scalar."""
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var lay = row_major(n)
    comptime k = mul_scalar_kernel[type_of(lay)]
    ctx.enqueue_function[k](TileTensor(x, lay), TileTensor(y, lay), n, c, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
    return y^

def embed_tokens(ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], mut emb: WBuf, T: Int,
                 hidden: Int, vocab: Int) raises -> DevBuf:
    var h = ctx.enqueue_create_buffer[DType.float32](T * hidden)
    var lay = row_major(T * hidden)
    comptime k = embed_kernel[type_of(lay)]   # dimension-agnostic (runtime T, hidden)
    ctx.enqueue_function[k](
        TileTensor(ids, row_major(T)), TileTensor(emb, row_major(vocab * hidden)),
        TileTensor(h, lay), T, hidden,
        grid_dim=ceildiv(T * hidden, BLOCK), block_dim=BLOCK,
    )
    return h^

def last_row(ctx: DeviceContext, mut src: DevBuf, T: Int, dim: Int) raises -> DevBuf:
    """Lift row T-1 (dim elements) of src[T,dim] into a fresh 1×dim buffer."""
    var y = ctx.enqueue_create_buffer[DType.float32](dim)
    var lay = row_major(dim)
    comptime k = slice_row_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, row_major(T * dim)), TileTensor(y, lay), (T - 1) * dim, dim,
        grid_dim=ceildiv(dim, BLOCK), block_dim=BLOCK,
    )
    return y^

def copy_into(ctx: DeviceContext, mut src: DevBuf, mut dst: DevBuf, dst_offset: Int, n: Int, dst_len: Int) raises:
    var lay = row_major(n)
    comptime k = copy_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, lay), TileTensor(dst, row_major(dst_len)), dst_offset, n,
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )

def copy_strided(ctx: DeviceContext, mut src: DevBuf, mut dst: DevBuf,
                 T: Int, in_stride: Int, in_off: Int, dst_off: Int, n: Int, dst_len: Int) raises:
    """Copy a [T, n] slice out of a strided source (V part of a fused [q|k|v]) into
    dst[dst_off:] — V into its cache rows."""
    var lay = row_major(T * in_stride)
    comptime k = copy_strided_kernel[type_of(lay)]
    ctx.enqueue_function[k](
        TileTensor(src, lay), TileTensor(dst, row_major(dst_len)), T, in_stride, in_off, dst_off, n,
        grid_dim=ceildiv(T * n, BLOCK), block_dim=BLOCK,
    )
