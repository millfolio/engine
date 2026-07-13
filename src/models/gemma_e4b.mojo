"""Gemma-4 e4b-it TEXT model (dense, 42 layers) — the effective-4B PLE sibling of
e2b (gemma_e2b.mojo), sharing its tokenizer/vocab (262144) and block structure.

Deltas vs e2b (all confirmed from the checkpoint's tensor shapes + text_config):
  * bigger dims: hidden 2560, inter 10240 (uniform — use_double_wide_mlp=false),
    42 layers, 8 q heads, 2 kv heads (e2b had 1).
  * full_attention at every 6th layer ([5,11,…,41]); sliding elsewhere. Same
    per-type geometry as e2b: sliding = head_dim 256 / θ=1e4 / full rotary /
    window 512; full = head_dim 512 / θ=1e6 / partial 64-pair rotary.
  * num_kv_shared_layers=18 IS modelled here (unlike e2b, where the checkpoint
    kept every layer's k/v): layers ≥ 24 ship NO k_proj/v_proj/k_norm — their
    fused projection is [q] only and they attend against the last same-type
    owning layer's cache (kv_src) with write=False.
  * Per-Layer Embeddings (PLE): identical composition to e2b; table row is
    42 × 256 = 10752.

NOTE: like e2b, `gemma4` has no runnable reference in this environment, so the
geometry is validated by greedy COHERENCE (and the shared-code lineage from the
12B/e2b, which are), not per-layer numeric parity."""

from std.math import ceildiv, sqrt
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from runtime.kernel_cache import cached_enqueue

from kernels import rope_q_kernel, rope_k_kernel, tc_attn_kernel, vnorm_kernel
from runtime.tensor_ops import (
    BLOCK,
    DevBuf,
    KVBuf,
    KV_DTYPE,
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

# ── e4b text dims (from text_config) ─────────────────────────────────────────
comptime E_HIDDEN = 2560
"""Hidden (model) dimension of the e4b text decoder."""
comptime E_INTER = 10240
"""MLP intermediate dimension (uniform — e4b has no double-wide layers)."""
comptime E_NLAYERS = 42
"""Number of decoder layers."""
comptime E_FIRST_KV_SHARED = E_NLAYERS - 18  # 24 — layers ≥ this share KV (their
"""First layer index (24) whose checkpoint ships no k/v (KV-shared)."""
# checkpoint has NO k_proj/v_proj/k_norm).
comptime E_VOCAB = 262144
"""Vocabulary size (shared across the Gemma-4 family)."""
comptime E_HQ = 8
"""Number of query attention heads."""
comptime E_EOS1 = 1
"""Primary end-of-sequence token id."""
comptime E_EOS2 = 106
"""Secondary end-of-sequence token id (end-of-turn)."""
comptime E_EMBED_SCALE = Float32(50.59644256269407)  # sqrt(2560)
"""Input-embedding scale factor (sqrt(hidden) = sqrt(2560))."""
comptime E_FINAL_SOFTCAP = Float32(30.0)
"""Logit soft-cap applied to the final LM-head output."""

# sliding (default rope): head_dim 256, 2 kv heads, θ=1e4, full rotary (128 pairs).
comptime ESL_HEAD_DIM = 256
"""Per-head dimension for the sliding-attention layers."""
comptime ESL_HKV = 2
"""Number of key/value heads for the sliding-attention layers."""
comptime ESL_NKV = ESL_HKV * ESL_HEAD_DIM  # 512
"""Total KV width per sliding layer (hkv × head_dim = 512)."""
comptime ESL_THETA = Float32(10000.0)
"""RoPE base frequency θ for the sliding-attention layers."""
comptime ESL_ROT_PAIRS = ESL_HEAD_DIM // 2  # 128 (full)
"""Number of rotary pairs for sliding layers (128 = full rotary over head_dim)."""
comptime E_SLIDING_WINDOW = 512
"""Local-attention window size for the sliding layers."""
# full (proportional rope): head_dim 512, 2 kv heads, θ=1e6, partial 64 pairs.
comptime EFU_HEAD_DIM = 512
"""Per-head dimension for the full-attention layers."""
comptime EFU_HKV = 2
"""Number of key/value heads for the full-attention layers."""
comptime EFU_NKV = EFU_HKV * EFU_HEAD_DIM  # 1024
"""Total KV width per full layer (hkv × head_dim = 1024)."""
comptime EFU_THETA = Float32(1000000.0)
"""RoPE base frequency θ for the full-attention layers."""
comptime EFU_ROT_PAIRS = 64  # int(0.25 * 512 // 2)
"""Number of rotary pairs for full layers (partial rotary, 64 pairs)."""

# ── Per-Layer Embeddings (PLE) ───────────────────────────────────────────────
comptime E_HPLI = 256  # hidden_size_per_layer_input
"""Per-layer-input embedding width (hidden_size_per_layer_input)."""
comptime E_PLE_WIDTH = E_NLAYERS * E_HPLI  # 10752 = embed_tokens_per_layer row
"""Width of one embed_tokens_per_layer row (nlayers × 256 = 10752)."""
comptime E_PLE_EMBED_SCALE = Float32(
    16.0
)  # sqrt(256), embed_tokens_per_layer scale
"""Scale applied to the per-layer embedding lookup (sqrt(256))."""
comptime E_PLE_PROJ_SCALE = Float32(0.019764235376052372)  # 2560**-0.5
"""Scale applied to the per-layer model projection (hidden**-0.5)."""
comptime E_PLE_INPUT_SCALE = Float32(0.7071067811865476)  # 2**-0.5
"""Scale combining projected + looked-up per-layer inputs (2**-0.5)."""


@fieldwise_init
struct GemmaE4bWeights(ModelWeights, Movable):
    """Weights + config for the Gemma-4 e4b text decoder (int4).
    Conforms to `ModelWeights` so the generic engine can run it.
    """

    var embed: WBuf  # bf16, tied LM head (RAW — embed scale is input-only)
    """`bf16` token-embedding table, also tied as the LM head (raw, unscaled)."""
    var embed_pl: WBuf  # bf16 embed_tokens_per_layer [vocab, 10752]
    """`bf16` per-layer embedding table embed_tokens_per_layer [vocab, 10752]."""
    var final_norm: DevBuf
    """Final RMSNorm weight before the LM head."""
    var ln1: List[DevBuf]
    """Per-layer input (pre-attention) RMSNorm weights."""
    var ln_post_attn: List[DevBuf]
    """Per-layer post-attention RMSNorm weights."""
    var ln_pre_ff: List[DevBuf]
    """Per-layer pre-feedforward RMSNorm weights."""
    var ln_post_ff: List[DevBuf]
    """Per-layer post-feedforward RMSNorm weights."""
    var qkv: List[QMat]  # fused [q|k|v] for own-KV layers; [q] only for shared
    """Per-layer fused [q|k|v] projection ([q] only for KV-shared layers ≥ 24)."""
    var ow: List[QMat]
    """Per-layer attention output projection (o_proj)."""
    var qnorm: List[DevBuf]
    """Per-layer query RMSNorm weights (applied per head)."""
    var knorm: List[DevBuf]
    """Per-layer key RMSNorm weights (applied per head)."""
    var gate_up: List[QMat]
    """Per-layer fused gate+up MLP projection."""
    var down: List[QMat]
    """Per-layer MLP down projection."""
    var layer_scalar: List[Float32]
    """Per-layer output scalar applied after PLE integration."""
    # PLE
    var plm_proj: QMat  # per_layer_model_projection [10752, 2560]
    """Per-layer model projection per_layer_model_projection [10752, 2560]."""
    var plp_norm: DevBuf  # per_layer_projection_norm [256]
    """Per-layer projection RMSNorm weight per_layer_projection_norm [256]."""
    var pli_gate: List[QMat]  # per_layer_input_gate [256, 2560]
    """Per-layer input-gate projection per_layer_input_gate [256, 2560]."""
    var pli_proj: List[QMat]  # per_layer_projection [2560, 256]
    """Per-layer projection back to hidden per_layer_projection [2560, 256]."""
    var pli_post_norm: List[DevBuf]  # post_per_layer_input_norm [2560]
    """Per-layer post-PLE-input RMSNorm weights post_per_layer_input_norm [2560]."""
    var ple: DevBuf  # per-forward per-layer inputs [T, 10752]; set in embed_prompt
    """Per-forward per-layer input signal [T, 10752]; populated by embed_prompt."""
    var is_full: List[Bool]
    """Per-layer flag: True for full-attention layers, False for sliding."""
    var kv_src: List[Int]  # KV-share source layer per layer (own l for l<24)
    """Per-layer KV-share source layer (own index for layers < 24)."""
    var hidden: Int
    """Hidden dimension (E_HIDDEN)."""
    var inter: Int
    """MLP intermediate dimension (E_INTER)."""
    var nlayers: Int
    """Number of decoder layers (E_NLAYERS)."""
    var vocab: Int
    """Vocabulary size (E_VOCAB)."""
    var hq: Int
    """Number of query heads (E_HQ)."""
    var simd_ok: Bool
    """Whether the SIMD-group GEMM fast path is enabled for this device."""
    var cfg: ModelConfig
    """Generic model configuration consumed by the engine."""

    # ── ModelWeights conformance ─────────────────────────────────────────────
    def config(self) -> ModelConfig:
        """Return the generic model configuration.

        Returns:
            The `ModelConfig` describing this model to the engine.
        """
        return self.cfg

    def embed_prompt(
        mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int
    ) raises -> DevBuf:
        """Embed `T` token ids into hidden states and precompute the per-layer
        (PLE) input signal for this forward pass; returns the scaled hidden.

        Args:
            ctx: The device (GPU) context.
            ids: The `T` input token ids on device.
            T: Number of tokens in the prompt.

        Returns:
            The scaled hidden states for the prompt.

        Raises:
            If a GPU kernel launch or buffer operation fails.
        """
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
        mut kcs: List[KVBuf],
        mut vcs: List[KVBuf],
        Tq: Int,
        q_offset: Int,
        cache_len: Int,
        mut dummy: DevBuf,
    ) raises -> DevBuf:
        """Run decoder layer `l` over hidden `h`, updating the KV caches.

        Args:
            ctx: The device (GPU) context.
            l: Index of the decoder layer to run.
            h: The hidden states (mutated/consumed).
            kcs: Per-layer key caches.
            vcs: Per-layer value caches.
            Tq: Number of query positions in this pass.
            q_offset: Absolute position of the first query token.
            cache_len: Current length of the KV cache.
            dummy: Scratch buffer reused across GEMM calls.

        Returns:
            The updated hidden states after the layer.

        Raises:
            If a GPU kernel launch or buffer operation fails.
        """
        return e4b_layer(
            ctx, self, l, h, kcs, vcs, Tq, q_offset, cache_len, dummy
        )

    def lm_logits(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        """Compute soft-capped LM-head logits for the LAST position only.

        Args:
            ctx: The device (GPU) context.
            h: The final hidden states.
            T: Number of positions in `h`.
            dummy: Scratch buffer reused across GEMM calls.

        Returns:
            The soft-capped logits over the vocabulary for the last position.

        Raises:
            If a GPU kernel launch or buffer operation fails.
        """
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
        var out = List[Float32](capacity=self.vocab)
        with logits.map_to_host() as m:
            var mt = TileTensor(m, row_major(self.vocab))
            for i in range(self.vocab):
                out.append(rebind[Scalar[DType.float32]](mt[i]))
        return out^

    def lm_logits_all(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        """Compute soft-capped LM-head logits for ALL `T` positions (flattened).

        Args:
            ctx: The device (GPU) context.
            h: The final hidden states.
            T: Number of positions in `h`.
            dummy: Scratch buffer reused across GEMM calls.

        Returns:
            The soft-capped logits for all `T` positions, flattened.

        Raises:
            If a GPU kernel launch or buffer operation fails.
        """
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
        var out = List[Float32](capacity=n)
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
        """Return per-position log-probabilities of the given target tokens.

        Args:
            ctx: The device (GPU) context.
            h: The final hidden states.
            n: Number of positions to score.
            targets: The target token id at each position.
            dummy: Scratch buffer reused across GEMM calls.

        Returns:
            The log-probability of each target token, per position.

        Raises:
            If a GPU kernel launch or buffer operation fails.
        """
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
    # layer_types: full_attention at 5,11,17,23,29,35,41 (every 6th, 1-indexed).
    return ((l + 1) % 6) == 0


def _load_scalar(
    ctx: DeviceContext,
    paths: List[String],
    entries: List[TensorEntry],
    name2idx: Dict[String, Int],
    name: String,
) raises -> Float32:
    var b = load_named(ctx, paths, entries, name2idx, name)
    ctx.synchronize()
    var v: Float32
    with b.map_to_host() as m:
        v = m[0]
    return v


def load_e4b_weights(
    ctx: DeviceContext, path: String, q4: Bool = True
) raises -> GemmaE4bWeights:
    """Load the gemma-4 e4b text decoder (int4 projections by default; the
    checkpoint is ~15 GB bf16 and multimodal — only the text decoder is loaded).

    Args:
        ctx: The device (GPU) context.
        path: Directory holding the e4b checkpoint safetensors.
        q4: Quantize projections to int4 when True (the default).

    Returns:
        The loaded `GemmaE4bWeights`.

    Raises:
        If a tensor is missing or a load/GPU operation fails.
    """
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

    # One entry per decoder layer in every list below.
    var ln1 = List[DevBuf](capacity=E_NLAYERS)
    var ln_post_attn = List[DevBuf](capacity=E_NLAYERS)
    var ln_pre_ff = List[DevBuf](capacity=E_NLAYERS)
    var ln_post_ff = List[DevBuf](capacity=E_NLAYERS)
    var qkv = List[QMat](capacity=E_NLAYERS)
    var ow = List[QMat](capacity=E_NLAYERS)
    var qnorm = List[DevBuf](capacity=E_NLAYERS)
    var knorm = List[DevBuf](capacity=E_NLAYERS)
    var gate_up = List[QMat](capacity=E_NLAYERS)
    var down = List[QMat](capacity=E_NLAYERS)
    var layer_scalar = List[Float32](capacity=E_NLAYERS)
    var pli_gate = List[QMat](capacity=E_NLAYERS)
    var pli_proj = List[QMat](capacity=E_NLAYERS)
    var pli_post_norm = List[DevBuf](capacity=E_NLAYERS)
    var is_full = List[Bool](capacity=E_NLAYERS)

    # KV sharing: layers ≥ FIRST_KV_SHARED reuse the last same-type layer's KV from
    # the pre-shared range; layers below compute (and cache) their own. Unlike e2b,
    # the e4b checkpoint ships NO k_proj/v_proj/k_norm for the shared layers.
    comptime FIRST_KV_SHARED = E_NLAYERS - 18  # 24
    var kv_src = List[Int](capacity=E_NLAYERS)
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
        if l < FIRST_KV_SHARED:
            # Own-KV layer: fuse [q|k|v] exactly like e2b.
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
            knorm.append(
                load_named(
                    ctx,
                    paths,
                    entries,
                    name2idx,
                    p + "self_attn.k_norm.weight",
                )
            )
        else:
            # KV-shared layer: the checkpoint has no k/v — the "fused" projection
            # is [q] only, and k_norm gets a placeholder (never read: e4b_attn
            # touches knw only under write=True, which shared layers never are).
            qkv.append(qpw^)
            knorm.append(ctx.enqueue_create_buffer[DType.float32](1))

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

        var inter_l = E_INTER  # uniform — e4b has no double-wide MLP
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
    return GemmaE4bWeights(
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


# ── attention dispatch (mirrors gemma.mojo, e4b geometry) ───────────────────────


def e4b_attn(
    ctx: DeviceContext,
    mut qkv: DevBuf,
    mut kc: KVBuf,
    mut vc: KVBuf,
    write: Bool,
    mut qnw: DevBuf,
    mut knw: DevBuf,
    l_full: Bool,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
) raises -> DevBuf:
    """Compute one attention block (RoPE Q/K, V-norm, tensor-core attention) for
    e2b geometry; writes K/V into the caches only when `write` is True.

    Args:
        ctx: The device (GPU) context.
        qkv: The fused [q|k|v] projection output for this layer.
        kc: The key cache to attend against (and write into when `write`).
        vc: The value cache to attend against (and write into when `write`).
        write: True to compute and store K/V into `kc`/`vc` (own-KV layer).
        qnw: The query-norm weight.
        knw: The key-norm weight.
        l_full: True for a full-attention layer, False for sliding.
        Tq: Number of query positions in this pass.
        q_offset: Absolute position of the first query token.
        cache_len: Current length of the KV cache.

    Returns:
        The attention output for this block.

    Raises:
        If a GPU kernel launch or buffer operation fails.
    """
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
    # Own-KV layers carry a fused [q|k|v]; KV-shared layers (write=False) carry
    # [q] only — the checkpoint has no k/v for them, so the row stride is q_dim.
    var W = (q_dim + 2 * nkv) if write else q_dim
    var k_off = q_dim
    var v_off = q_dim + nkv

    var qr = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var qslay = row_major(Tq * W)
    var qrlay = row_major(Tq * q_dim)
    var qnlay = row_major(head_dim)
    if l_full:
        comptime kq = rope_q_kernel[type_of(qrlay), E_HQ, EFU_HEAD_DIM, True]
        cached_enqueue[kq](
            ctx,
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
        cached_enqueue[kq](
            ctx,
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
                type_of(qslay), EFU_HKV, EFU_HEAD_DIM, True, KV_DTYPE
            ]
            cached_enqueue[kk](
                ctx,
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
                type_of(qslay), ESL_HKV, ESL_HEAD_DIM, True, KV_DTYPE
            ]
            cached_enqueue[kk](
                ctx,
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
            comptime kv = vnorm_kernel[type_of(qslay), EFU_HKV, EFU_HEAD_DIM, KV_DTYPE]
            cached_enqueue[kv](
                ctx,
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
            comptime kv = vnorm_kernel[type_of(qslay), ESL_HKV, ESL_HEAD_DIM, KV_DTYPE]
            cached_enqueue[kv](
                ctx,
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
        comptime ka = tc_attn_kernel[type_of(olay), E_HQ, EFU_HKV, EFU_HEAD_DIM, KV_DTYPE]
        cached_enqueue[ka](
            ctx,
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
        comptime ka = tc_attn_kernel[type_of(olay), E_HQ, ESL_HKV, ESL_HEAD_DIM, KV_DTYPE]
        cached_enqueue[ka](
            ctx,
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


def e4b_layer(
    ctx: DeviceContext,
    mut w: GemmaE4bWeights,
    l: Int,
    mut h: DevBuf,
    mut kcs: List[KVBuf],
    mut vcs: List[KVBuf],
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    mut dummy: DevBuf,
) raises -> DevBuf:
    """Forward one Gemma-4 e2b decoder layer: attention + MLP residual blocks
    followed by Per-Layer-Embedding integration and the layer_scalar.

    Args:
        ctx: The device (GPU) context.
        w: The e2b model weights.
        l: Index of the decoder layer to run.
        h: The input hidden states.
        kcs: Per-layer key caches.
        vcs: Per-layer value caches.
        Tq: Number of query positions in this pass.
        q_offset: Absolute position of the first query token.
        cache_len: Current length of the KV cache.
        dummy: Scratch buffer reused across GEMM calls.

    Returns:
        The hidden states after the layer.

    Raises:
        If a GPU kernel launch or buffer operation fails.
    """
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
    var inter = w.inter  # uniform — e4b has no double-wide MLP

    # ── standard Gemma-4 decoder block (mirrors gemma.mojo gemma_layer) ──
    # Shared layers project [q] only (their qkv matrix has no k/v columns).
    var W = (q_dim + 2 * nkv) if write else q_dim
    var qkv = mm_w_norm(
        ctx, h, w.ln1[l], w.qkv[l], dummy, Tq, hd, W, 0, w.simd_ok
    )
    var o = e4b_attn(
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
