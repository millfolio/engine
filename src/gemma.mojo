"""Gemma 4 12B-it TEXT model family (dense, 48 layers).

Mirrors `qwen.mojo`: a `GemmaWeights(Movable, ModelWeights)` struct + a loader
(`load_gemma_weights`) + the decoder-layer forward (`gemma_layer`) + the RoPE/
attention dispatch. Composes the shared `tensor_ops`/`kernels`/`safetensors`
libraries; the engine drives it through the `ModelWeights` trait.

Gemma 4 12B specifics handled here (confirmed from the checkpoint config +
transformers' Gemma4 source — the per-layer fixtures are the contract):
- RMSNorm is the PLAIN `x/rms * weight` variant (Gemma4RMSNorm does `* weight`,
  NOT `* (1+weight)` — the q_norm weights are stored ~1.02, input_layernorm ~6.6).
  So the shared rmsnorm/mm_norm kernels are used as-is; v_norm is scale-free.
  (The Gemma2/3 (1+w) convention does NOT apply to Gemma4 — verified vs the fixture.)
- Two layer types with DIFFERENT attention geometry:
  * sliding_attention (default): head_dim 256, 16 q / 8 kv heads, v_proj present,
    RoPE θ=1e4 FULL rotary (all 128 pairs).
  * full_attention (proportional): head_dim = global_head_dim = 512, 16 q / 1 kv
    head, NO v_proj (V = v_norm(k_proj), K = k_norm(k_proj) share the projection
    output — attention_k_eq_v=True), RoPE θ=1e6 PARTIAL rotary (first 64 of 256
    pairs over the 512-dim head).
  Both use softmax scaling = 1.0 (NOT 1/sqrt(head_dim) — HF sets self.scaling=1.0).
- per-head q_norm/k_norm (RMSNorm over head_dim) + scale-free v_norm, all applied
  BEFORE RoPE. GeGLU MLP. Embeddings ×√hidden (input path only); tied LM head uses
  RAW embeds; final-logit softcap = 30. layer_scalar [1] per layer.
- seq < sliding_window (1024) at prefill ⇒ sliding == full-causal here.
"""

from std.math import ceildiv
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import rope_q_kernel, rope_k_kernel, tc_attn_kernel, vnorm_kernel
from tensor_ops import (
    BLOCK, DevBuf, WBuf, QMat, qmat_bf16, mm_w, mm_norm, mm_w_norm,
    embed_tokens, last_row, rmsnorm, rmsnorm_add, add, gelu_mul_cat, softcap, mul_scalar,
)
from safetensors import (
    TensorEntry, gather_tensors, load_named, load_named_bf16, load_proj, fuse_pair,
)
from model_iface import ModelConfig, ModelWeights, FAMILY_GEMMA, ACT_GELU, TOOL_GEMMA

# Gemma 4 12B-it dims (from text_config).
comptime G_HIDDEN = 3840
comptime G_INTER = 15360
comptime G_NLAYERS = 48
comptime G_VOCAB = 262144
comptime G_HQ = 16
comptime G_EOS1 = 1
comptime G_EOS2 = 106
comptime G_EMBED_SCALE = Float32(61.96773353931867)   # sqrt(3840)
comptime G_FINAL_SOFTCAP = Float32(30.0)

# sliding (default rope): head_dim 256, 8 kv heads, θ=1e4, full rotary (128 pairs).
comptime SL_HEAD_DIM = 256
comptime SL_HKV = 8
comptime SL_NKV = SL_HKV * SL_HEAD_DIM        # 2048
comptime SL_THETA = Float32(10000.0)
comptime SL_ROT_PAIRS = SL_HEAD_DIM // 2      # 128 (full)
comptime G_SLIDING_WINDOW = 1024              # sliding layers attend only the last 1024 keys
# full (proportional rope): head_dim 512 (global), 1 kv head, θ=1e6, partial 64 pairs.
comptime FU_HEAD_DIM = 512
comptime FU_HKV = 1
comptime FU_NKV = FU_HKV * FU_HEAD_DIM        # 512
comptime FU_THETA = Float32(1000000.0)
comptime FU_ROT_PAIRS = 64                    # int(0.25 * 512 // 2)


@fieldwise_init
struct GemmaWeights(Movable, ModelWeights):
    var embed: WBuf                  # bf16, tied LM head (RAW — embed scale is input-only)
    var final_norm: DevBuf           # final RMSNorm weight
    var ln1: List[DevBuf]            # input_layernorm
    var ln_post_attn: List[DevBuf]   # post_attention_layernorm
    var ln_pre_ff: List[DevBuf]      # pre_feedforward_layernorm
    var ln_post_ff: List[DevBuf]     # post_feedforward_layernorm
    var qkv: List[QMat]              # full layers: [q|k] (no separate v); sliding: [q|k|v]
    var ow: List[QMat]               # o_proj
    var qnorm: List[DevBuf]          # per-head q_norm [head_dim]
    var knorm: List[DevBuf]          # per-head k_norm [head_dim]
    var gate_up: List[QMat]          # gate|up fused
    var down: List[QMat]
    var layer_scalar: List[Float32]  # [1] per-layer learned scalar
    var is_full: List[Bool]          # True = full_attention layer
    var hidden: Int
    var inter: Int
    var nlayers: Int
    var vocab: Int
    var hq: Int
    var simd_ok: Bool
    var cfg: ModelConfig

    # ── ModelWeights conformance ─────────────────────────────────────────────
    def config(self) -> ModelConfig:
        return self.cfg

    def embed_prompt(mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int) raises -> DevBuf:
        var h = embed_tokens(ctx, ids, self.embed, T, self.hidden, self.vocab)
        # Gemma scales the embeddings by √hidden on the INPUT path only.
        return mul_scalar(ctx, h, T * self.hidden, G_EMBED_SCALE)

    def run_layer(mut self, ctx: DeviceContext, l: Int, mut h: DevBuf,
                 mut kcs: List[DevBuf], mut vcs: List[DevBuf],
                 Tq: Int, q_offset: Int, cache_len: Int, mut dummy: DevBuf) raises -> DevBuf:
        return gemma_layer(ctx, self, l, h, kcs[l], vcs[l], Tq, q_offset, cache_len, dummy)

    def lm_logits(mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
        # Final (1+w) RMSNorm + tied LM head over the last row, then softcap=30.
        var hl = last_row(ctx, h, T, self.hidden)
        var logits = mm_norm(ctx, hl, self.final_norm, self.embed, dummy, 1, self.hidden, self.vocab, 0)
        softcap(ctx, logits, self.vocab, G_FINAL_SOFTCAP)
        ctx.synchronize()
        var out = List[Float32]()
        with logits.map_to_host() as m:
            var mt = TileTensor(m, row_major(self.vocab))
            for i in range(self.vocab):
                out.append(rebind[Scalar[DType.float32]](mt[i]))
        return out^

    def lm_logits_all(mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
        # All-row logits for spec-decode verification, with the final softcap=30
        # applied across every position.
        var n = T * self.vocab
        var logits = mm_norm(ctx, h, self.final_norm, self.embed, dummy, T, self.hidden, self.vocab, 0)
        softcap(ctx, logits, n, G_FINAL_SOFTCAP)
        ctx.synchronize()
        var out = List[Float32]()
        with logits.map_to_host() as m:
            var mt = TileTensor(m, row_major(n))
            for i in range(n):
                out.append(rebind[Scalar[DType.float32]](mt[i]))
        return out^


def load_gemma_weights(ctx: DeviceContext, path: String, layers: List[Int], q4: Bool = False) raises -> GemmaWeights:
    """Load the Gemma text decoder. `layers` selects which decoder layers to
    actually load (the rest get size-1 placeholders) so the per-layer validation
    stays tiny in memory; pass range(48) for the full model. embed/final_norm are
    always loaded (embed stays bf16). With q4=True the projection weights are
    group-128 int4 (the full bf16 model is ~24 GB and won't fit a 24 GB GPU; int4
    is ~7 GB). Skips vision/audio embedder tensors."""
    var gathered = gather_tensors(path)
    var entries = gathered[0].copy()
    var paths = gathered[1].copy()
    var name2idx = Dict[String, Int]()
    for e in range(len(entries)):
        name2idx[entries[e].name] = e
    var pfx = String("language_model.model.")

    var embed = load_named_bf16(ctx, paths, entries, name2idx, pfx + "embed_tokens.weight")
    var final_norm = load_named(ctx, paths, entries, name2idx, pfx + "norm.weight")

    var want = Dict[Int, Bool]()
    for i in range(len(layers)):
        want[layers[i]] = True

    var ln1 = List[DevBuf]()
    var ln_post_attn = List[DevBuf]()
    var ln_pre_ff = List[DevBuf]()
    var ln_post_ff = List[DevBuf]()
    var qkv = List[QMat]()
    var ow = List[QMat]()
    var qnorm = List[DevBuf]()
    var knorm = List[DevBuf]()
    var gate_up = List[QMat]()
    var down = List[QMat]()
    var layer_scalar = List[Float32]()
    var is_full = List[Bool]()

    for l in range(G_NLAYERS):
        var full = _is_full_layer(l)
        is_full.append(full)
        if l not in want:
            # placeholder buffers so list indices line up; never read for skipped layers.
            ln1.append(ctx.enqueue_create_buffer[DType.float32](1))
            ln_post_attn.append(ctx.enqueue_create_buffer[DType.float32](1))
            ln_pre_ff.append(ctx.enqueue_create_buffer[DType.float32](1))
            ln_post_ff.append(ctx.enqueue_create_buffer[DType.float32](1))
            qkv.append(qmat_bf16(ctx, ctx.enqueue_create_buffer[DType.uint16](1)))
            ow.append(qmat_bf16(ctx, ctx.enqueue_create_buffer[DType.uint16](1)))
            qnorm.append(ctx.enqueue_create_buffer[DType.float32](1))
            knorm.append(ctx.enqueue_create_buffer[DType.float32](1))
            gate_up.append(qmat_bf16(ctx, ctx.enqueue_create_buffer[DType.uint16](1)))
            down.append(qmat_bf16(ctx, ctx.enqueue_create_buffer[DType.uint16](1)))
            layer_scalar.append(1.0)
            continue

        var p = pfx + "layers." + String(l) + "."
        var head_dim = FU_HEAD_DIM if full else SL_HEAD_DIM
        var hkv = FU_HKV if full else SL_HKV
        var nkv = hkv * head_dim
        var q_dim = G_HQ * head_dim

        ln1.append(load_named(ctx, paths, entries, name2idx, p + "input_layernorm.weight"))
        ln_post_attn.append(load_named(ctx, paths, entries, name2idx, p + "post_attention_layernorm.weight"))
        ln_pre_ff.append(load_named(ctx, paths, entries, name2idx, p + "pre_feedforward_layernorm.weight"))
        ln_post_ff.append(load_named(ctx, paths, entries, name2idx, p + "post_feedforward_layernorm.weight"))

        # q|k|v projections. Sliding: fuse q|k|v. Full: only q|k (V reuses k_proj).
        var qpw = load_proj(ctx, paths, entries, name2idx, p + "self_attn.q_proj.weight", G_HIDDEN, q4)
        var kpw = load_proj(ctx, paths, entries, name2idx, p + "self_attn.k_proj.weight", G_HIDDEN, q4)
        if full:
            qkv.append(fuse_pair(ctx, qpw^, kpw^, q_dim, nkv, G_HIDDEN, q4))   # [q|k]
        else:
            var vpw = load_proj(ctx, paths, entries, name2idx, p + "self_attn.v_proj.weight", G_HIDDEN, q4)
            var qk = fuse_pair(ctx, qpw^, kpw^, q_dim, nkv, G_HIDDEN, q4)
            qkv.append(fuse_pair(ctx, qk^, vpw^, q_dim + nkv, nkv, G_HIDDEN, q4))  # [q|k|v]

        ow.append(load_proj(ctx, paths, entries, name2idx, p + "self_attn.o_proj.weight", q_dim, q4))
        qnorm.append(load_named(ctx, paths, entries, name2idx, p + "self_attn.q_norm.weight"))
        knorm.append(load_named(ctx, paths, entries, name2idx, p + "self_attn.k_norm.weight"))

        var gp = load_proj(ctx, paths, entries, name2idx, p + "mlp.gate_proj.weight", G_HIDDEN, q4)
        var upj = load_proj(ctx, paths, entries, name2idx, p + "mlp.up_proj.weight", G_HIDDEN, q4)
        gate_up.append(fuse_pair(ctx, gp^, upj^, G_INTER, G_INTER, G_HIDDEN, q4))
        down.append(load_proj(ctx, paths, entries, name2idx, p + "mlp.down_proj.weight", G_INTER, q4))
        layer_scalar.append(_load_scalar(ctx, paths, entries, name2idx, p + "layer_scalar"))

    # nkv for the engine's cache sizing: use the MAX of the two layer types
    # (SL_NKV=2048 > FU_NKV=512) so a single per-layer cache row fits either.
    # (The per-layer test sizes its own caches.) norm_offset=0 (plain RMSNorm).
    var cfg = ModelConfig(
        FAMILY_GEMMA, G_NLAYERS, SL_NKV, False, True, ACT_GELU, 0.0, G_FINAL_SOFTCAP,
        1024, SL_THETA, G_EMBED_SCALE, 0.0, G_EOS1, G_EOS2, TOOL_GEMMA, 50,   # 50 = <|tool_response>
    )
    return GemmaWeights(
        embed^, final_norm^, ln1^, ln_post_attn^, ln_pre_ff^, ln_post_ff^, qkv^, ow^,
        qnorm^, knorm^, gate_up^, down^, layer_scalar^, is_full^,
        G_HIDDEN, G_INTER, G_NLAYERS, G_VOCAB, G_HQ, False, cfg^,
    )


def _is_full_layer(l: Int) -> Bool:
    # layer_types: full_attention at 5,11,17,23,29,35,41,47 (every 6th, 1-indexed).
    return ((l + 1) % 6) == 0


def _load_scalar(ctx: DeviceContext, paths: List[String], entries: List[TensorEntry],
                 name2idx: Dict[String, Int], name: String) raises -> Float32:
    """Read a [1] bf16 layer_scalar to a host f32 value."""
    var b = load_named(ctx, paths, entries, name2idx, name)
    ctx.synchronize()
    var out = List[Float32]()
    with b.map_to_host() as m:
        out.append(m[0])
    return out[0]


# ── Gemma RoPE / attention dispatch (comptime-specialized per layer type) ───────

def gemma_attn(ctx: DeviceContext, mut qkv: DevBuf, mut kc: DevBuf, mut vc: DevBuf,
               mut qnw: DevBuf, mut knw: DevBuf, l_full: Bool,
               Tq: Int, q_offset: Int, cache_len: Int) raises -> DevBuf:
    """Project→norm→RoPE→attend for one Gemma layer. `qkv` is the fused projection
    output ([q|k|v] for sliding, [q|k] for full). Writes rotated K + normalized V
    into the caches, rotates Q, runs tensor-core causal attention (scale=1.0)."""
    var head_dim = FU_HEAD_DIM if l_full else SL_HEAD_DIM
    var hkv = FU_HKV if l_full else SL_HKV
    var hq = G_HQ
    var nkv = hkv * head_dim
    var q_dim = hq * head_dim
    var theta = FU_THETA if l_full else SL_THETA
    var rot = FU_ROT_PAIRS if l_full else SL_ROT_PAIRS
    # fused row width: full = q_dim + nkv ([q|k]); sliding = q_dim + 2*nkv ([q|k|v]).
    var W = q_dim + (1 if l_full else 2) * nkv
    var k_off = q_dim
    var v_off = q_dim + nkv     # only valid for sliding ([q|k|v]); full reuses k_off for V

    # Q: per-head q_norm + RoPE → contiguous rotated buffer.
    var qr = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var qslay = row_major(Tq * W)
    var qrlay = row_major(Tq * q_dim)
    var qnlay = row_major(head_dim)
    if l_full:
        comptime kq = rope_q_kernel[type_of(qrlay), G_HQ, FU_HEAD_DIM, True]
        ctx.enqueue_function[kq](TileTensor(qkv, qslay), TileTensor(qr, qrlay), TileTensor(qnw, qnlay),
            Tq, q_offset, W, 0, theta, rot, grid_dim=ceildiv(Tq * hq, BLOCK), block_dim=BLOCK)
    else:
        comptime kq = rope_q_kernel[type_of(qrlay), G_HQ, SL_HEAD_DIM, True]
        ctx.enqueue_function[kq](TileTensor(qkv, qslay), TileTensor(qr, qrlay), TileTensor(qnw, qnlay),
            Tq, q_offset, W, 0, theta, rot, grid_dim=ceildiv(Tq * hq, BLOCK), block_dim=BLOCK)

    # K: per-head k_norm + RoPE → cache rows (absolute position).
    var clay = row_major(cache_len)
    var knlay = row_major(head_dim)
    if l_full:
        comptime kk = rope_k_kernel[type_of(qslay), FU_HKV, FU_HEAD_DIM, True]
        ctx.enqueue_function[kk](TileTensor(qkv, qslay), TileTensor(kc, clay), TileTensor(knw, knlay),
            Tq, q_offset, W, k_off, theta, rot, grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK)
    else:
        comptime kk = rope_k_kernel[type_of(qslay), SL_HKV, SL_HEAD_DIM, True]
        ctx.enqueue_function[kk](TileTensor(qkv, qslay), TileTensor(kc, clay), TileTensor(knw, knlay),
            Tq, q_offset, W, k_off, theta, rot, grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK)

    # V: scale-free v_norm → cache rows. Full reuses k_proj output (V=k_proj@k_off);
    # sliding reads the V slice at v_off.
    var vsrc_off = k_off if l_full else v_off
    if l_full:
        comptime kv = vnorm_kernel[type_of(qslay), FU_HKV, FU_HEAD_DIM]
        ctx.enqueue_function[kv](TileTensor(qkv, qslay), TileTensor(vc, clay),
            Tq, q_offset, W, vsrc_off, grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK)
    else:
        comptime kv = vnorm_kernel[type_of(qslay), SL_HKV, SL_HEAD_DIM]
        ctx.enqueue_function[kv](TileTensor(qkv, qslay), TileTensor(vc, clay),
            Tq, q_offset, W, vsrc_off, grid_dim=ceildiv(Tq * hkv, BLOCK), block_dim=BLOCK)

    # Tensor-core causal GQA attention, softmax scale = 1.0 (Gemma).
    var o = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var olay = row_major(Tq * q_dim)
    var grid = ceildiv(Tq, 8) * hq
    if l_full:
        comptime ka = tc_attn_kernel[type_of(olay), G_HQ, FU_HKV, FU_HEAD_DIM]
        ctx.enqueue_function[ka](TileTensor(qr, qrlay), TileTensor(kc, clay), TileTensor(vc, clay),
            TileTensor(o, olay), Tq, q_offset, Float32(1.0), 0, grid_dim=grid, block_dim=WARP_SIZE)
    else:
        comptime ka = tc_attn_kernel[type_of(olay), G_HQ, SL_HKV, SL_HEAD_DIM]
        ctx.enqueue_function[ka](TileTensor(qr, qrlay), TileTensor(kc, clay), TileTensor(vc, clay),
            TileTensor(o, olay), Tq, q_offset, Float32(1.0), G_SLIDING_WINDOW,
            grid_dim=grid, block_dim=WARP_SIZE)
    return o^


# ── decoder layer forward ───────────────────────────────────────────────────────

def gemma_layer(ctx: DeviceContext, mut w: GemmaWeights, l: Int, mut h: DevBuf,
                mut kc: DevBuf, mut vc: DevBuf, Tq: Int, q_offset: Int,
                cache_len: Int, mut dummy: DevBuf) raises -> DevBuf:
    """One Gemma decoder layer (dense). Mirrors the HF Gemma4TextDecoderLayer:
      r=h; h=in_ln(h); h=attn(h); h=post_attn_ln(h); h=r+h
      r=h; h=pre_ff_ln(h); h=geglu(h); h=post_ff_ln(h); h=r+h
      h=h*layer_scalar
    All RMSNorms are the plain `x/rms*weight` variant (no (1+w) offset)."""
    var full = w.is_full[l]
    var head_dim = FU_HEAD_DIM if full else SL_HEAD_DIM
    var hkv = FU_HKV if full else SL_HKV
    var nkv = hkv * head_dim
    var q_dim = w.hq * head_dim
    var hd = w.hidden

    # ── attention block ──
    # input_layernorm fused into the qkv GEMV (prefill: rmsnorm then mm).
    var W = q_dim + (1 if full else 2) * nkv
    var qkv = mm_w_norm(ctx, h, w.ln1[l], w.qkv[l], dummy, Tq, hd, W, 0, w.simd_ok)
    var o = gemma_attn(ctx, qkv, kc, vc, w.qnorm[l], w.knorm[l], full, Tq, q_offset, cache_len)
    # o_proj(o)[q_dim→hd]
    var ao = mm_w(ctx, o, w.ow[l], dummy, Tq, q_dim, hd, 0, w.simd_ok)
    # post_attention_layernorm + residual add fused (Gemma norms the attn out).
    var h1 = rmsnorm_add(ctx, ao, w.ln_post_attn[l], h, Tq, hd)

    # ── feed-forward block ──
    var gu = mm_w_norm(ctx, h1, w.ln_pre_ff[l], w.gate_up[l], dummy, Tq, hd, 2 * w.inter, 0, w.simd_ok)
    var act = gelu_mul_cat(ctx, gu, Tq, w.inter)
    var ff = mm_w(ctx, act, w.down[l], dummy, Tq, w.inter, hd, 0, w.simd_ok)
    # post_feedforward_layernorm + residual + the per-layer scalar, all fused.
    return rmsnorm_add(ctx, ff, w.ln_post_ff[l], h1, Tq, hd, w.layer_scalar[l])
