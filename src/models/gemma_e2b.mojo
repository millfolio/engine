"""Gemma-4 e2b-it TEXT model (dense, 35 layers) — the speculative-decode DRAFT
model for the Gemma-4 12B target. Same tokenizer/vocab (262144) as the 12B, so its
greedy tokens are directly verifiable by the 12B.

Architecturally it is the 12B's dense Gemma-4 decoder (see gemma.mojo) with:
  * smaller dims: hidden 1536, inter 6144, 35 layers, 8 q heads.
  * full_attention at every 5th layer ([4,9,…,34]); sliding elsewhere. Geometry
    mirrors the 12B: sliding = head_dim 256 / θ=1e4 / full rotary / window 512;
    full = head_dim 512 / θ=1e6 / partial 64-pair rotary. Both 1 kv head; ALL
    layers have a separate v_proj (the 12B's full layers shared k as v).
  * Per-Layer Embeddings (PLE): a second embedding table feeds each layer a 256-d
    per-token signal — gated by the layer's hidden, multiplied in, projected back
    to hidden, normed, and added. (No AltUp / LAuReL — this checkpoint has neither.)

NOTE: `gemma4` has no public HF/MLX reference in this environment, so the PLE
composition, partial-rotary, scale-free v_norm, and layer_scalar placement below
are reconstructed from the tensor shapes + the Gemma3n PLE source and validated by
greedy COHERENCE + agreement with the 12B, not per-layer numeric parity.

num_kv_shared_layers=20 (the real model shares KV across the last 20 layers) is
NOT modelled: every layer computes its own KV (the checkpoint keeps all k/v_proj),
which the generic engine cache already expects. As a draft that only costs a little
acceptance, never correctness (the 12B verifies every token)."""

from std.math import ceildiv, sqrt
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import rope_q_kernel, rope_k_kernel, tc_attn_kernel, vnorm_kernel
from runtime.tensor_ops import (
    BLOCK,
    DevBuf,
    WBuf,
    QMat,
    qmat_bf16,
    mm_w,
    mm_norm,
    mm_w_norm,
    embed_tokens,
    last_row,
    rmsnorm,
    rmsnorm_add,
    add,
    gelu_mul_cat,
    gelu_mul,
    gelu_mul_strided,
    softcap,
    mul_scalar,
    copy_strided,
    nll_gather,
)
from runtime.safetensors import (
    TensorEntry,
    gather_tensors,
    load_named,
    load_named_bf16,
    load_proj,
    fuse_pair,
)
from runtime.model_iface import (
    ModelConfig,
    ModelWeights,
    FAMILY_GEMMA,
    ACT_GELU,
    TOOL_GEMMA,
)

# ── e2b text dims (from text_config) ─────────────────────────────────────────
comptime E_HIDDEN = 1536
comptime E_INTER = 6144
comptime E_NLAYERS = 35
comptime E_FIRST_KV_SHARED = E_NLAYERS - 20  # 15 — layers ≥ this share KV AND use
# a DOUBLE-WIDE MLP (use_double_wide_mlp).
comptime E_VOCAB = 262144
comptime E_HQ = 8
comptime E_EOS1 = 1
comptime E_EOS2 = 106
comptime E_EMBED_SCALE = Float32(39.19183588453085)  # sqrt(1536)
comptime E_FINAL_SOFTCAP = Float32(30.0)

# sliding (default rope): head_dim 256, 1 kv head, θ=1e4, full rotary (128 pairs).
comptime ESL_HEAD_DIM = 256
comptime ESL_HKV = 1
comptime ESL_NKV = ESL_HKV * ESL_HEAD_DIM  # 256
comptime ESL_THETA = Float32(10000.0)
comptime ESL_ROT_PAIRS = ESL_HEAD_DIM // 2  # 128 (full)
comptime E_SLIDING_WINDOW = 512
# full (proportional rope): head_dim 512, 1 kv head, θ=1e6, partial 64 pairs.
comptime EFU_HEAD_DIM = 512
comptime EFU_HKV = 1
comptime EFU_NKV = EFU_HKV * EFU_HEAD_DIM  # 512
comptime EFU_THETA = Float32(1000000.0)
comptime EFU_ROT_PAIRS = 64  # int(0.25 * 512 // 2)

# ── Per-Layer Embeddings (PLE) ───────────────────────────────────────────────
comptime E_HPLI = 256  # hidden_size_per_layer_input
comptime E_PLE_WIDTH = E_NLAYERS * E_HPLI  # 8960 = embed_tokens_per_layer row
comptime E_PLE_EMBED_SCALE = Float32(
    16.0
)  # sqrt(256), embed_tokens_per_layer scale
comptime E_PLE_PROJ_SCALE = Float32(0.02551551815399144)  # 1536**-0.5
comptime E_PLE_INPUT_SCALE = Float32(0.7071067811865476)  # 2**-0.5


@fieldwise_init
struct GemmaE2bWeights(ModelWeights, Movable):
    var embed: WBuf  # bf16, tied LM head (RAW — embed scale is input-only)
    var embed_pl: WBuf  # bf16 embed_tokens_per_layer [vocab, 8960]
    var final_norm: DevBuf
    var ln1: List[DevBuf]
    var ln_post_attn: List[DevBuf]
    var ln_pre_ff: List[DevBuf]
    var ln_post_ff: List[DevBuf]
    var qkv: List[QMat]  # fused [q|k|v] (all layers have v_proj)
    var ow: List[QMat]
    var qnorm: List[DevBuf]
    var knorm: List[DevBuf]
    var gate_up: List[QMat]
    var down: List[QMat]
    var layer_scalar: List[Float32]
    # PLE
    var plm_proj: QMat  # per_layer_model_projection [8960, 1536]
    var plp_norm: DevBuf  # per_layer_projection_norm [256]
    var pli_gate: List[QMat]  # per_layer_input_gate [256, 1536]
    var pli_proj: List[QMat]  # per_layer_projection [1536, 256]
    var pli_post_norm: List[DevBuf]  # post_per_layer_input_norm [1536]
    var ple: DevBuf  # per-forward per-layer inputs [T, 8960]; set in embed_prompt
    var is_full: List[Bool]
    var kv_src: List[Int]  # KV-share source layer per layer (own l for l<15)
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

    def embed_prompt(
        mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int
    ) raises -> DevBuf:
        var h = embed_tokens(ctx, ids, self.embed, T, self.hidden, self.vocab)
        h = mul_scalar(
            ctx, h, T * self.hidden, E_EMBED_SCALE
        )  # input path: ×√hidden
        # PLE per-layer inputs: (norm(model_proj(h)·hidden^-0.5) + embed_pl·√256)·2^-0.5
        var pl = embed_tokens(
            ctx, ids, self.embed_pl, T, E_PLE_WIDTH, self.vocab
        )
        pl = mul_scalar(ctx, pl, T * E_PLE_WIDTH, E_PLE_EMBED_SCALE)
        var dummy = ctx.enqueue_create_buffer[DType.float32](1)
        var proj = mm_w(
            ctx,
            h,
            self.plm_proj,
            dummy,
            T,
            self.hidden,
            E_PLE_WIDTH,
            0,
            self.simd_ok,
        )
        proj = mul_scalar(ctx, proj, T * E_PLE_WIDTH, E_PLE_PROJ_SCALE)
        proj = rmsnorm(
            ctx, proj, self.plp_norm, T * E_NLAYERS, E_HPLI
        )  # norm each 256 group
        var ple = add(ctx, proj, pl, T * E_PLE_WIDTH)
        ple = mul_scalar(ctx, ple, T * E_PLE_WIDTH, E_PLE_INPUT_SCALE)
        self.ple = ple^
        return h^

    def run_layer(
        mut self,
        ctx: DeviceContext,
        l: Int,
        mut h: DevBuf,
        mut kcs: List[DevBuf],
        mut vcs: List[DevBuf],
        Tq: Int,
        q_offset: Int,
        cache_len: Int,
        mut dummy: DevBuf,
    ) raises -> DevBuf:
        return e2b_layer(
            ctx, self, l, h, kcs, vcs, Tq, q_offset, cache_len, dummy
        )

    def lm_logits(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        var hl = last_row(ctx, h, T, self.hidden)
        var logits = mm_norm(
            ctx,
            hl,
            self.final_norm,
            self.embed,
            dummy,
            1,
            self.hidden,
            self.vocab,
            0,
        )
        softcap(ctx, logits, self.vocab, E_FINAL_SOFTCAP)
        ctx.synchronize()
        var out = List[Float32]()
        with logits.map_to_host() as m:
            var mt = TileTensor(m, row_major(self.vocab))
            for i in range(self.vocab):
                out.append(rebind[Scalar[DType.float32]](mt[i]))
        return out^

    def lm_logits_all(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        var n = T * self.vocab
        var logits = mm_norm(
            ctx,
            h,
            self.final_norm,
            self.embed,
            dummy,
            T,
            self.hidden,
            self.vocab,
            0,
        )
        softcap(ctx, logits, n, E_FINAL_SOFTCAP)
        ctx.synchronize()
        var out = List[Float32]()
        with logits.map_to_host() as m:
            var mt = TileTensor(m, row_major(n))
            for i in range(n):
                out.append(rebind[Scalar[DType.float32]](mt[i]))
        return out^

    def token_logprobs(
        mut self,
        ctx: DeviceContext,
        mut h: DevBuf,
        n: Int,
        targets: List[Int],
        mut dummy: DevBuf,
    ) raises -> List[Float32]:
        var logits = mm_norm(
            ctx,
            h,
            self.final_norm,
            self.embed,
            dummy,
            n,
            self.hidden,
            self.vocab,
            0,
        )
        softcap(ctx, logits, n * self.vocab, E_FINAL_SOFTCAP)
        return nll_gather(ctx, logits, targets, n, self.vocab)


def _is_full_layer(l: Int) -> Bool:
    # layer_types: full_attention at 4,9,14,19,24,29,34 (every 5th, 1-indexed).
    return ((l + 1) % 5) == 0


def _load_scalar(
    ctx: DeviceContext,
    paths: List[String],
    entries: List[TensorEntry],
    name2idx: Dict[String, Int],
    name: String,
) raises -> Float32:
    var b = load_named(ctx, paths, entries, name2idx, name)
    ctx.synchronize()
    var out = List[Float32]()
    with b.map_to_host() as m:
        out.append(m[0])
    return out[0]


def load_e2b_weights(
    ctx: DeviceContext, path: String, q4: Bool = True
) raises -> GemmaE2bWeights:
    """Load the gemma-4 e2b text decoder (int4 projections by default — the draft
    is ~5 GB bf16, ~1.4 GB int4, so it co-resides with the 12B target)."""
    var gathered = gather_tensors(path)
    var entries = gathered[0].copy()
    var paths = gathered[1].copy()
    var name2idx = Dict[String, Int]()
    for e in range(len(entries)):
        name2idx[entries[e].name] = e
    var pfx = String("language_model.model.")

    var embed = load_named_bf16(
        ctx, paths, entries, name2idx, pfx + "embed_tokens.weight"
    )
    var embed_pl = load_named_bf16(
        ctx, paths, entries, name2idx, pfx + "embed_tokens_per_layer.weight"
    )
    var final_norm = load_named(
        ctx, paths, entries, name2idx, pfx + "norm.weight"
    )
    var plm_proj = load_proj(
        ctx,
        paths,
        entries,
        name2idx,
        pfx + "per_layer_model_projection.weight",
        E_HIDDEN,
        q4,
    )
    var plp_norm = load_named(
        ctx, paths, entries, name2idx, pfx + "per_layer_projection_norm.weight"
    )

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
    var pli_gate = List[QMat]()
    var pli_proj = List[QMat]()
    var pli_post_norm = List[DevBuf]()
    var is_full = List[Bool]()

    # KV sharing: layers ≥ FIRST_KV_SHARED reuse the last same-type layer's KV from
    # the pre-shared range; layers below compute (and cache) their own.
    comptime FIRST_KV_SHARED = E_NLAYERS - 20  # 15
    var kv_src = List[Int]()
    var last_sliding = 0
    var last_full = 0

    for l in range(E_NLAYERS):
        var full = _is_full_layer(l)
        is_full.append(full)
        if l < FIRST_KV_SHARED:
            kv_src.append(l)
            if full:
                last_full = l
            else:
                last_sliding = l
        else:
            kv_src.append(last_full if full else last_sliding)
        var p = pfx + "layers." + String(l) + "."
        var head_dim = EFU_HEAD_DIM if full else ESL_HEAD_DIM
        var hkv = EFU_HKV if full else ESL_HKV
        var nkv = hkv * head_dim
        var q_dim = E_HQ * head_dim

        ln1.append(
            load_named(
                ctx, paths, entries, name2idx, p + "input_layernorm.weight"
            )
        )
        ln_post_attn.append(
            load_named(
                ctx,
                paths,
                entries,
                name2idx,
                p + "post_attention_layernorm.weight",
            )
        )
        ln_pre_ff.append(
            load_named(
                ctx,
                paths,
                entries,
                name2idx,
                p + "pre_feedforward_layernorm.weight",
            )
        )
        ln_post_ff.append(
            load_named(
                ctx,
                paths,
                entries,
                name2idx,
                p + "post_feedforward_layernorm.weight",
            )
        )

        var qpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.q_proj.weight",
            E_HIDDEN,
            q4,
        )
        var kpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.k_proj.weight",
            E_HIDDEN,
            q4,
        )
        var vpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.v_proj.weight",
            E_HIDDEN,
            q4,
        )
        var qk = fuse_pair(ctx, qpw^, kpw^, q_dim, nkv, E_HIDDEN, q4)
        qkv.append(
            fuse_pair(ctx, qk^, vpw^, q_dim + nkv, nkv, E_HIDDEN, q4)
        )  # [q|k|v]

        ow.append(
            load_proj(
                ctx,
                paths,
                entries,
                name2idx,
                p + "self_attn.o_proj.weight",
                q_dim,
                q4,
            )
        )
        qnorm.append(
            load_named(
                ctx, paths, entries, name2idx, p + "self_attn.q_norm.weight"
            )
        )
        knorm.append(
            load_named(
                ctx, paths, entries, name2idx, p + "self_attn.k_norm.weight"
            )
        )

        var inter_l = (
            2 * E_INTER
        ) if l >= E_FIRST_KV_SHARED else E_INTER  # double-wide for shared
        var gp = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "mlp.gate_proj.weight",
            E_HIDDEN,
            q4,
        )
        var upj = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "mlp.up_proj.weight",
            E_HIDDEN,
            q4,
        )
        gate_up.append(
            fuse_pair(ctx, gp^, upj^, inter_l, inter_l, E_HIDDEN, q4)
        )
        down.append(
            load_proj(
                ctx,
                paths,
                entries,
                name2idx,
                p + "mlp.down_proj.weight",
                inter_l,
                q4,
            )
        )
        layer_scalar.append(
            _load_scalar(ctx, paths, entries, name2idx, p + "layer_scalar")
        )

        pli_gate.append(
            load_proj(
                ctx,
                paths,
                entries,
                name2idx,
                p + "per_layer_input_gate.weight",
                E_HIDDEN,
                q4,
            )
        )
        pli_proj.append(
            load_proj(
                ctx,
                paths,
                entries,
                name2idx,
                p + "per_layer_projection.weight",
                E_HPLI,
                q4,
            )
        )
        pli_post_norm.append(
            load_named(
                ctx,
                paths,
                entries,
                name2idx,
                p + "post_per_layer_input_norm.weight",
            )
        )

    var cfg = ModelConfig(
        FAMILY_GEMMA,
        E_NLAYERS,
        EFU_NKV,
        False,
        True,
        ACT_GELU,
        0.0,
        E_FINAL_SOFTCAP,
        E_SLIDING_WINDOW,
        ESL_THETA,
        E_EMBED_SCALE,
        0.0,
        E_EOS1,
        E_EOS2,
        TOOL_GEMMA,
        50,
    )
    return GemmaE2bWeights(
        embed^,
        embed_pl^,
        final_norm^,
        ln1^,
        ln_post_attn^,
        ln_pre_ff^,
        ln_post_ff^,
        qkv^,
        ow^,
        qnorm^,
        knorm^,
        gate_up^,
        down^,
        layer_scalar^,
        plm_proj^,
        plp_norm^,
        pli_gate^,
        pli_proj^,
        pli_post_norm^,
        ctx.enqueue_create_buffer[DType.float32](1),
        is_full^,
        kv_src^,
        E_HIDDEN,
        E_INTER,
        E_NLAYERS,
        E_VOCAB,
        E_HQ,
        False,
        cfg^,
    )


# ── attention dispatch (mirrors gemma.mojo, e2b geometry) ───────────────────────


def e2b_attn(
    ctx: DeviceContext,
    mut qkv: DevBuf,
    mut kc: DevBuf,
    mut vc: DevBuf,
    write: Bool,
    mut qnw: DevBuf,
    mut knw: DevBuf,
    l_full: Bool,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
) raises -> DevBuf:
    # `kc`/`vc` are the caches to ATTEND against. write=True (own-KV layer): compute
    # K/V from `qkv` and store them into kc/vc first. write=False (KV-shared layer):
    # kc/vc are an EARLIER layer's caches — only Q is computed; K/V are left as-is.
    var head_dim = EFU_HEAD_DIM if l_full else ESL_HEAD_DIM
    var hkv = EFU_HKV if l_full else ESL_HKV
    var hq = E_HQ
    var nkv = hkv * head_dim
    var q_dim = hq * head_dim
    var theta = EFU_THETA if l_full else ESL_THETA
    var rot = EFU_ROT_PAIRS if l_full else ESL_ROT_PAIRS
    var W = q_dim + 2 * nkv  # fused [q|k|v]
    var k_off = q_dim
    var v_off = q_dim + nkv

    var qr = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var qslay = row_major(Tq * W)
    var qrlay = row_major(Tq * q_dim)
    var qnlay = row_major(head_dim)
    if l_full:
        comptime kq = rope_q_kernel[type_of(qrlay), E_HQ, EFU_HEAD_DIM, True]
        ctx.enqueue_function[kq](
            TileTensor(qkv, qslay),
            TileTensor(qr, qrlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            W,
            0,
            theta,
            rot,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime kq = rope_q_kernel[type_of(qrlay), E_HQ, ESL_HEAD_DIM, True]
        ctx.enqueue_function[kq](
            TileTensor(qkv, qslay),
            TileTensor(qr, qrlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            W,
            0,
            theta,
            rot,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )

    var clay = row_major(cache_len)
    var knlay = row_major(head_dim)
    if write:
        # K: per-head k_norm + RoPE → own cache rows.
        if l_full:
            comptime kk = rope_k_kernel[
                type_of(qslay), EFU_HKV, EFU_HEAD_DIM, True
            ]
            ctx.enqueue_function[kk](
                TileTensor(qkv, qslay),
                TileTensor(kc, clay),
                TileTensor(knw, knlay),
                Tq,
                q_offset,
                W,
                k_off,
                theta,
                rot,
                grid_dim=ceildiv(Tq * hkv, BLOCK),
                block_dim=BLOCK,
            )
        else:
            comptime kk = rope_k_kernel[
                type_of(qslay), ESL_HKV, ESL_HEAD_DIM, True
            ]
            ctx.enqueue_function[kk](
                TileTensor(qkv, qslay),
                TileTensor(kc, clay),
                TileTensor(knw, knlay),
                Tq,
                q_offset,
                W,
                k_off,
                theta,
                rot,
                grid_dim=ceildiv(Tq * hkv, BLOCK),
                block_dim=BLOCK,
            )
        # V: scale-free v_norm → own cache rows (mirror 12B: Gemma-4 norms V).
        if l_full:
            comptime kv = vnorm_kernel[type_of(qslay), EFU_HKV, EFU_HEAD_DIM]
            ctx.enqueue_function[kv](
                TileTensor(qkv, qslay),
                TileTensor(vc, clay),
                Tq,
                q_offset,
                W,
                v_off,
                grid_dim=ceildiv(Tq * hkv, BLOCK),
                block_dim=BLOCK,
            )
        else:
            comptime kv = vnorm_kernel[type_of(qslay), ESL_HKV, ESL_HEAD_DIM]
            ctx.enqueue_function[kv](
                TileTensor(qkv, qslay),
                TileTensor(vc, clay),
                Tq,
                q_offset,
                W,
                v_off,
                grid_dim=ceildiv(Tq * hkv, BLOCK),
                block_dim=BLOCK,
            )

    var o = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var olay = row_major(Tq * q_dim)
    var grid = ceildiv(Tq, 8) * hq
    if l_full:
        comptime ka = tc_attn_kernel[type_of(olay), E_HQ, EFU_HKV, EFU_HEAD_DIM]
        ctx.enqueue_function[ka](
            TileTensor(qr, qrlay),
            TileTensor(kc, clay),
            TileTensor(vc, clay),
            TileTensor(o, olay),
            Tq,
            q_offset,
            Float32(1.0),
            0,
            grid_dim=grid,
            block_dim=WARP_SIZE,
        )
    else:
        comptime ka = tc_attn_kernel[type_of(olay), E_HQ, ESL_HKV, ESL_HEAD_DIM]
        ctx.enqueue_function[ka](
            TileTensor(qr, qrlay),
            TileTensor(kc, clay),
            TileTensor(vc, clay),
            TileTensor(o, olay),
            Tq,
            q_offset,
            Float32(1.0),
            E_SLIDING_WINDOW,
            grid_dim=grid,
            block_dim=WARP_SIZE,
        )
    return o^


# ── decoder layer forward (standard gemma-4 block + PLE) ─────────────────────────


def e2b_layer(
    ctx: DeviceContext,
    mut w: GemmaE2bWeights,
    l: Int,
    mut h: DevBuf,
    mut kcs: List[DevBuf],
    mut vcs: List[DevBuf],
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    mut dummy: DevBuf,
) raises -> DevBuf:
    var full = w.is_full[l]
    var head_dim = EFU_HEAD_DIM if full else ESL_HEAD_DIM
    var hkv = EFU_HKV if full else ESL_HKV
    var nkv = hkv * head_dim
    var q_dim = w.hq * head_dim
    var hd = w.hidden
    # KV sharing: own layer (src==l) computes+caches its own K/V; a shared layer
    # (src<l) attends against the source layer's cache and writes nothing.
    var src = w.kv_src[l]
    var write = src == l
    var inter = (
        2 * w.inter
    ) if l >= E_FIRST_KV_SHARED else w.inter  # double-wide MLP for shared

    # ── standard Gemma-4 decoder block (mirrors gemma.mojo gemma_layer) ──
    var W = q_dim + 2 * nkv
    var qkv = mm_w_norm(
        ctx, h, w.ln1[l], w.qkv[l], dummy, Tq, hd, W, 0, w.simd_ok
    )
    var o = e2b_attn(
        ctx,
        qkv,
        kcs[src],
        vcs[src],
        write,
        w.qnorm[l],
        w.knorm[l],
        full,
        Tq,
        q_offset,
        cache_len,
    )
    var ao = mm_w(ctx, o, w.ow[l], dummy, Tq, q_dim, hd, 0, w.simd_ok)
    var h1 = rmsnorm_add(
        ctx, ao, w.ln_post_attn[l], h, Tq, hd
    )  # post_attn_norm + residual

    var gu = mm_w_norm(
        ctx,
        h1,
        w.ln_pre_ff[l],
        w.gate_up[l],
        dummy,
        Tq,
        hd,
        2 * inter,
        0,
        w.simd_ok,
    )
    var act = gelu_mul_cat(ctx, gu, Tq, inter)
    var ff = mm_w(ctx, act, w.down[l], dummy, Tq, inter, hd, 0, w.simd_ok)
    var h2 = rmsnorm_add(
        ctx, ff, w.ln_post_ff[l], h1, Tq, hd
    )  # post_ff_norm + residual

    # ── Per-Layer Embedding integration (BEFORE layer_scalar, per gemma4 ref) ──
    # first = gelu(per_layer_input_gate(h2)) ⊙ per_layer_input[l]
    # h_out = (h2 + post_per_layer_input_norm(per_layer_projection(first))) · layer_scalar
    var g = mm_w(ctx, h2, w.pli_gate[l], dummy, Tq, hd, E_HPLI, 0, w.simd_ok)
    var gp = gelu_mul_strided(
        ctx, g, w.ple, Tq, E_HPLI, E_PLE_WIDTH, l * E_HPLI
    )
    var pred = mm_w(ctx, gp, w.pli_proj[l], dummy, Tq, E_HPLI, hd, 0, w.simd_ok)
    # PLE residual + layer_scalar fused into the post-PLE norm.
    return rmsnorm_add(
        ctx, pred, w.pli_post_norm[l], h2, Tq, hd, w.layer_scalar[l]
    )
