"""Qwen model family (Qwen2.5-0.5B / 3B, Qwen3-Embedding-0.6B).

This is the per-family module: it owns the Qwen weight layout (`Weights`), the
checkpoint→`Weights` mapping (`load_weights`, with auto arch detection), the
Qwen RoPE/attention dispatch, and the decoder-layer forward (`qwen_layer`). It
composes the shared libraries — `tensor_ops` (mm/norm/activation launchers),
`safetensors` (checkpoint I/O), `kernels` (GPU kernels) — and exposes a
`ModelConfig` carrying the behavior flags that distinguish model families, so the
shared `engine` can stay model-agnostic and dispatch on `cfg.family`.

Adding a new family (e.g. Gemma) = a new module like this one (its `ModelConfig`,
`load_weights`, `*_layer`) + one dispatch arm in `engine`; the shared libs are
reused as-is. The `arch` field still selects the comptime-specialized head
kernels by dim-tuple — that's inherent to comptime specialization, not per-arch
branching of behavior."""

from std.math import ceildiv, sqrt
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from kernels import (
    rope_k_kernel,
    rope_kv_kernel,
    rope_q_kernel,
    attn_cached_kernel,
    attn_cached_rope_kernel,
    flash_attn_kernel,
    tc_attn_kernel,
    FLASH_PW,
    THETA,
)
from runtime.tensor_ops import (
    BLOCK,
    DevBuf,
    WBuf,
    PBuf,
    QMat,
    qmat_bf16,
    mm_w_norm,
    mm_w_add,
    mm_w_silu_add,
    embed_tokens,
    mm_norm,
    last_row,
    rmsnorm,
    nll_gather,
)
from runtime.safetensors import (
    TensorEntry,
    gather_tensors,
    load_named,
    load_named_bf16,
    load_proj,
    fuse_pair,
    concat_bias,
)
from runtime.model_iface import (
    ModelConfig,
    ModelWeights,
    FAMILY_QWEN,
    ACT_SILU,
    ACT_GELU,
    TOOL_QWEN,
)
from runtime.engine import new_session, upload_ids

# Qwen2.5-0.5B preset (the default when the checkpoint matches hidden==896).
comptime HQ = 14
comptime H = 896
comptime NKV = 128
comptime HKV = 2
comptime INTER = 4864
comptime VOCAB = 151936
comptime NLAYERS = 24
comptime EOS1 = 151645
comptime EOS2 = 151643

# Above this context length (keys = q_offset + Tq) the f32 KV working set spills
# the M4 system cache and attn_cached_kernel super-cliffs; flash_attn_kernel
# (shared-memory K/V staging, bit-identical output) wins past the ~20K crossover.
comptime FLASH_THRESHOLD = 20480

# ModelConfig, ModelWeights, FAMILY_*/ACT_* now live in model_iface (shared across
# families). Qwen's Weights conforms to ModelWeights below.


@fieldwise_init
struct Weights(ModelWeights, Movable):
    var embed: WBuf  # bf16 — the input embedding table
    var lm_head: WBuf  # bf16 — the output LM head. A 2nd copy of `embed` for
    # tied models (0.5B/3B/0.6B); the separate lm_head.weight
    # for untied Qwen3 chat models (8B/14B).
    var final_norm: DevBuf
    var ln1: List[DevBuf]
    var qkv: List[QMat]  # q_proj|k_proj|v_proj concatenated along N (one GEMV)
    var qkv_b: List[DevBuf]  # concatenated q|k|v biases
    var ow: List[QMat]
    var ln2: List[DevBuf]
    var gate_up: List[
        QMat
    ]  # gate_proj and up_proj concatenated along N (one GEMV)
    var down: List[QMat]
    # Qwen3 (arch==2) only: per-layer QK-RMSNorm weights [head_dim] applied to Q
    # and K per head BEFORE RoPE. Empty for arch 0/1 (Qwen2.5 has no qk-norm).
    var qnorm: List[DevBuf]
    var knorm: List[DevBuf]
    # Architecture dims, auto-detected from the checkpoint (see load_weights).
    # `arch` selects the comptime head-kernel instantiation: 0 = 0.5B, 1 = 3B,
    # 2 = Qwen3-Embedding-0.6B (decoupled q_dim, qk-norm, no qkv bias).
    var arch: Int
    var nlayers: Int
    var hidden: Int  # H
    var inter: Int  # INTER (MLP)
    var nkv: Int  # HKV * HEAD_DIM (K/V row width)
    var hq: Int  # query heads
    var hkv: Int  # kv heads
    var head_dim: Int
    # q_dim = hq * head_dim — the Q-projection / attention-output width. Equals
    # `hidden` for Qwen2.5 (decoupled only for Qwen3: 16*128=2048 != 1024).
    var q_dim: Int
    var vocab: Int
    # Set once at startup by probe_simd_gemm: use the simdgroup-matrix GEMM for
    # prefill if this toolchain accepts the AIR intrinsics, else the scalar path.
    var simd_ok: Bool
    # True if the projection weights (qw/kw/vw/ow/gate/up/down) are group-128 int4;
    # embed/lm-head stays bf16 either way. Reported in the startup banner.
    var quant: Bool
    # Behavior flags + engine-relevant dims/eos (the engine is generic over this).
    var cfg: ModelConfig

    # ── ModelWeights conformance (the engine drives the loop via these) ──────────
    def config(self) -> ModelConfig:
        return self.cfg

    def embed_prompt(
        mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int
    ) raises -> DevBuf:
        return embed_tokens(
            ctx, ids, self.embed, T, self.hidden, self.vocab
        )  # Qwen: no embed scale

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
        return qwen_layer(
            ctx, self, l, h, kcs[l], vcs[l], Tq, q_offset, cache_len, dummy
        )

    def lm_logits(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        # Final RMSNorm + tied LM head over the last row (Qwen: no final softcap).
        var hl = last_row(ctx, h, T, self.hidden)
        var logits = mm_norm(
            ctx,
            hl,
            self.final_norm,
            self.lm_head,
            dummy,
            1,
            self.hidden,
            self.vocab,
            0,
        )
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
        # All-row logits for spec-decode verification (Qwen: no final softcap).
        var n = T * self.vocab
        var logits = mm_norm(
            ctx,
            h,
            self.final_norm,
            self.lm_head,
            dummy,
            T,
            self.hidden,
            self.vocab,
            0,
        )
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
            self.lm_head,
            dummy,
            n,
            self.hidden,
            self.vocab,
            0,
        )
        return nll_gather(ctx, logits, targets, n, self.vocab)


def _hidden_size(
    entries: List[TensorEntry], name2idx: Dict[String, Int], pfx: String
) raises -> Int:
    """Hidden size = the width of an RMSNorm weight ([hidden], bf16 → /2 bytes).
    """
    var idx = name2idx[pfx + "layers.0.input_layernorm.weight"]
    return (entries[idx].end - entries[idx].begin) // 2


def load_weights(
    ctx: DeviceContext, path: String, q4: Bool = False
) raises -> Weights:
    var gathered = gather_tensors(path)
    var entries = gathered[0].copy()
    var paths = gathered[1].copy()
    var name2idx = Dict[String, Int]()
    for e in range(len(entries)):
        name2idx[entries[e].name] = e

    # Auto-detect which Qwen2.5 this checkpoint is from its hidden size, and pick
    # the matching dims. Both sizes share vocab/θ/ε/tokenizer/template and tie the
    # LM head, so only these dims differ. arch selects the comptime head kernels.
    # The Qwen2.5 checkpoints (ForCausalLM) prefix every tensor with `model.`;
    # the Qwen3-Embedding checkpoint (a bare AutoModel) has NO prefix (e.g.
    # `embed_tokens.weight`, `layers.0…`, `norm.weight`). Detect which and use it
    # uniformly so the rest of the loader is prefix-agnostic.
    var pfx = String("model.") if (
        "model.embed_tokens.weight" in name2idx
    ) else String("")
    var hidden = _hidden_size(entries, name2idx, pfx)
    # Qwen3 has per-head QK-RMSNorm (q_norm/k_norm) and no QKV bias; Qwen2.5 has
    # neither. The q_norm weight is the unambiguous Qwen3 signal (hidden==1024 is
    # otherwise ambiguous, and Qwen2 q_proj has a bias Qwen3 lacks).
    var is_qwen3 = (pfx + "layers.0.self_attn.q_norm.weight") in name2idx
    var arch = 0
    var hq = HQ
    var hkv = HKV
    var head_dim = 64
    var inter = INTER
    var nlayers = NLAYERS
    var vocab = VOCAB
    if is_qwen3:  # Qwen3 family (qk-norm, no qkv bias, head_dim 128, 8 kv)
        hkv = 8
        head_dim = 128
        if hidden == 1024:  # Qwen3-Embedding-0.6B (decoupled q_dim)
            arch = 2
            hq = 16
            inter = 3072
            nlayers = 28
            vocab = 151669
        elif hidden == 4096:  # Qwen3-8B (untied lm-head)
            arch = 3
            hq = 32
            inter = 12288
            nlayers = 36
        elif hidden == 5120:  # Qwen3-14B (untied lm-head)
            arch = 4
            hq = 40
            inter = 17408
            nlayers = 40
        else:
            raise Error(
                "unsupported Qwen3 hidden size "
                + String(hidden)
                + " (expected 1024=Embedding-0.6B, 4096=8B, 5120=14B)"
            )
    elif hidden == 2048:  # Qwen2.5-3B
        arch = 1
        hq = 16
        hkv = 2
        head_dim = 128
        inter = 11008
        nlayers = 36
    elif hidden != 896:  # Qwen2.5-0.5B is the default preset above
        raise Error(
            "unsupported hidden size "
            + String(hidden)
            + " (expected 896 = Qwen2.5-0.5B, 2048 = Qwen2.5-3B,"
            + " or a Qwen3 checkpoint with q_norm)"
        )
    var nkv = hkv * head_dim
    var q_dim = (
        hq * head_dim
    )  # Q-proj / attn-output width (decoupled for Qwen3)

    # int4 needs K (the reduction dim) to be a multiple of Q4_GROUP; hidden and
    # inter satisfy this for both supported archs. embed/lm-head stays bf16.
    var embed = load_named_bf16(
        ctx, paths, entries, name2idx, pfx + "embed_tokens.weight"
    )
    # LM head: untied checkpoints (Qwen3 8B/14B) ship a separate lm_head.weight;
    # tied ones (0.5B/3B/0.6B) reuse the embedding, so load a 2nd copy of it so the
    # logit path can always read self.lm_head without a tied/untied branch.
    # `lm_head.weight` sits at the TOP level (sibling of `model.`), not under the
    # `model.` prefix the body tensors use — so try the bare name too before tying.
    var lm_head_name = pfx + "lm_head.weight"
    if lm_head_name not in name2idx:
        lm_head_name = String("lm_head.weight")
    var lm_head: WBuf
    if lm_head_name in name2idx:
        lm_head = load_named_bf16(ctx, paths, entries, name2idx, lm_head_name)
    else:
        lm_head = load_named_bf16(
            ctx, paths, entries, name2idx, pfx + "embed_tokens.weight"
        )
    var final_norm = load_named(
        ctx, paths, entries, name2idx, pfx + "norm.weight"
    )
    var ln1 = List[DevBuf]()
    var qkv = List[QMat]()
    var qkv_b = List[DevBuf]()
    var ow = List[QMat]()
    var ln2 = List[DevBuf]()
    var gate_up = List[QMat]()
    var down = List[QMat]()
    var qnorm = List[DevBuf]()  # Qwen3 only (arch==2); empty otherwise
    var knorm = List[DevBuf]()
    for l in range(nlayers):
        var p = pfx + "layers." + String(l) + "."
        ln1.append(
            load_named(
                ctx, paths, entries, name2idx, p + "input_layernorm.weight"
            )
        )
        # q|k|v concatenated along N → one GEMV (q rows [q_dim], then k, then v).
        var qpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.q_proj.weight",
            hidden,
            q4,
        )
        var kpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.k_proj.weight",
            hidden,
            q4,
        )
        var vpw = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "self_attn.v_proj.weight",
            hidden,
            q4,
        )
        var qk = fuse_pair(ctx, qpw^, kpw^, q_dim, nkv, hidden, q4)
        qkv.append(fuse_pair(ctx, qk^, vpw^, q_dim + nkv, nkv, hidden, q4))
        # Qwen2.5 has q/k/v_proj biases (concatenated into the fused GEMV); Qwen3
        # (arch >= 2) has none, so we push a size-1 dummy and pass use_bias=0.
        if arch >= 2:
            qkv_b.append(ctx.enqueue_create_buffer[DType.float32](1))
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
        else:
            var qpb = load_named(
                ctx, paths, entries, name2idx, p + "self_attn.q_proj.bias"
            )
            var kpb = load_named(
                ctx, paths, entries, name2idx, p + "self_attn.k_proj.bias"
            )
            var vpb = load_named(
                ctx, paths, entries, name2idx, p + "self_attn.v_proj.bias"
            )
            var qkb = concat_bias(ctx, qpb^, kpb^, q_dim, nkv)
            qkv_b.append(concat_bias(ctx, qkb^, vpb^, q_dim + nkv, nkv))
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
        ln2.append(
            load_named(
                ctx,
                paths,
                entries,
                name2idx,
                p + "post_attention_layernorm.weight",
            )
        )
        var gp = load_proj(
            ctx,
            paths,
            entries,
            name2idx,
            p + "mlp.gate_proj.weight",
            hidden,
            q4,
        )
        var upj = load_proj(
            ctx, paths, entries, name2idx, p + "mlp.up_proj.weight", hidden, q4
        )
        gate_up.append(fuse_pair(ctx, gp^, upj^, inter, inter, hidden, q4))
        down.append(
            load_proj(
                ctx,
                paths,
                entries,
                name2idx,
                p + "mlp.down_proj.weight",
                inter,
                q4,
            )
        )
    # Behavior config: Qwen2.5 (arch 0/1) has QKV bias + no qk-norm; Qwen3 (arch 2)
    # is the reverse. All Gemma-only knobs stay off. θ/ε are shared across Qwen.
    var cfg = ModelConfig(
        FAMILY_QWEN,
        nlayers,
        nkv,
        arch < 2,
        arch >= 2,
        ACT_SILU,
        0.0,
        0.0,
        0,
        1000000.0,
        1.0,
        0.0,
        EOS1,
        EOS2,
        TOOL_QWEN,
        -1,
    )
    return Weights(
        embed^,
        lm_head^,
        final_norm^,
        ln1^,
        qkv^,
        qkv_b^,
        ow^,
        ln2^,
        gate_up^,
        down^,
        qnorm^,
        knorm^,
        arch,
        nlayers,
        hidden,
        inter,
        nkv,
        hq,
        hkv,
        head_dim,
        q_dim,
        vocab,
        False,
        q4,
        cfg^,
    )


# ── Qwen RoPE / attention dispatch (comptime-specialized by arch dim-tuple) ─────


def rope_k(
    ctx: DeviceContext,
    mut kin: DevBuf,
    mut kc: DevBuf,
    mut knw: DevBuf,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    hkv: Int = HKV,
    head_dim: Int = 64,
    arch: Int = 0,
    in_stride: Int = -1,
    in_off: Int = 0,
) raises:
    """Apply RoPE to projected K and store it (rotated) at its cache rows —
    replaces the plain K copy so attention reads pre-rotated K (§11 #12).
    arch selects the comptime head-dim instantiation (0 = 0.5B, 1 = 3B,
    2 = Qwen3 with per-head QK-RMSNorm using `knw` [head_dim] before rotation).
    `kin` may be a strided K-slice of a fused [q|k|v] buffer (in_stride/in_off);
    in_stride<0 defaults to the contiguous nkv stride."""
    var nkv = hkv * head_dim
    var strd = in_stride if in_stride >= 0 else nkv
    var lay = row_major(Tq * strd)
    var nlay = row_major(head_dim if arch >= 2 else 1)
    if arch >= 2:  # all Qwen3 (0.6B/8B/14B): 8 kv heads, head_dim 128, qk-norm
        comptime k = rope_k_kernel[type_of(lay), 8, 128, True]
        ctx.enqueue_function[k](
            TileTensor(kin, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            strd,
            in_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 1:
        comptime k = rope_k_kernel[type_of(lay), 2, 128, False]
        ctx.enqueue_function[k](
            TileTensor(kin, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            strd,
            in_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime k = rope_k_kernel[type_of(lay), 2, 64, False]
        ctx.enqueue_function[k](
            TileTensor(kin, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            strd,
            in_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )


def rope_kv(
    ctx: DeviceContext,
    mut qkv: DevBuf,
    mut kc: DevBuf,
    mut vc: DevBuf,
    mut knw: DevBuf,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    hkv: Int,
    head_dim: Int,
    arch: Int,
    in_stride: Int,
    k_off: Int,
    v_off: Int,
) raises:
    """rope_k + the V cache-copy in one launch (rope_kv_kernel): rotates K into the
    cache and copies V into the cache from the fused [q|k|v] buffer — one dispatch
    instead of two per layer."""
    var lay = row_major(Tq * in_stride)
    var nlay = row_major(head_dim if arch >= 2 else 1)
    if arch >= 2:  # all Qwen3 (0.6B/8B/14B): 8 kv heads, head_dim 128, qk-norm
        comptime k = rope_kv_kernel[type_of(lay), 8, 128, True]
        ctx.enqueue_function[k](
            TileTensor(qkv, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            in_stride,
            k_off,
            v_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 1:
        comptime k = rope_kv_kernel[type_of(lay), 2, 128, False]
        ctx.enqueue_function[k](
            TileTensor(qkv, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            in_stride,
            k_off,
            v_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime k = rope_kv_kernel[type_of(lay), 2, 64, False]
        ctx.enqueue_function[k](
            TileTensor(qkv, lay),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(knw, nlay),
            Tq,
            q_offset,
            in_stride,
            k_off,
            v_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hkv, BLOCK),
            block_dim=BLOCK,
        )


def attn_cached(
    ctx: DeviceContext,
    mut q: DevBuf,
    mut kc: DevBuf,
    mut vc: DevBuf,
    mut qnw: DevBuf,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    hidden: Int = H,
    hq: Int = HQ,
    hkv: Int = HKV,
    head_dim: Int = 64,
    arch: Int = 0,
    q_stride: Int = -1,
    q_off: Int = 0,
) raises -> DevBuf:
    # Rotate Q first (rope_q), then attend; K is already rotated in the cache.
    # `q` may be a strided Q-slice of a fused [q|k|v] buffer (q_stride/q_off);
    # q_stride<0 means the contiguous q_dim stride. The rotated Q and the
    # attention output O are q_dim = hq*head_dim wide (== hidden for Qwen2.5; the
    # decoupled Qwen3 q_dim=2048 differs from hidden=1024). o_proj (the caller)
    # maps this q_dim-wide O back to hidden.
    var q_dim = hq * head_dim
    var qstr = q_stride if q_stride >= 0 else q_dim
    var qslay = row_major(Tq * qstr)
    var qnlay = row_major(head_dim if arch >= 2 else 1)
    var o = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var lay = row_major(Tq * q_dim)
    var flash_decode = arch == 0 and q_offset + Tq > FLASH_THRESHOLD

    # DECODE (Tq=1, non-flash): RoPE-Q fused into attention — no rope_q launch, no
    # qr buffer (attn_cached_rope_kernel rotates Q on load).
    if Tq == 1 and not flash_decode:
        var gd = ceildiv(Tq * hq * WARP_SIZE, BLOCK)
        if arch == 2:
            comptime k = attn_cached_rope_kernel[type_of(lay), 16, 8, 128, True]
            ctx.enqueue_function[k](
                TileTensor(q, qslay),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(qnw, qnlay),
                TileTensor(o, lay),
                Tq,
                q_offset,
                qstr,
                q_off,
                grid_dim=gd,
                block_dim=BLOCK,
            )
        elif arch == 3:
            comptime k = attn_cached_rope_kernel[type_of(lay), 32, 8, 128, True]
            ctx.enqueue_function[k](
                TileTensor(q, qslay),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(qnw, qnlay),
                TileTensor(o, lay),
                Tq,
                q_offset,
                qstr,
                q_off,
                grid_dim=gd,
                block_dim=BLOCK,
            )
        elif arch == 4:
            comptime k = attn_cached_rope_kernel[type_of(lay), 40, 8, 128, True]
            ctx.enqueue_function[k](
                TileTensor(q, qslay),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(qnw, qnlay),
                TileTensor(o, lay),
                Tq,
                q_offset,
                qstr,
                q_off,
                grid_dim=gd,
                block_dim=BLOCK,
            )
        elif arch == 1:
            comptime k = attn_cached_rope_kernel[
                type_of(lay), 16, 2, 128, False
            ]
            ctx.enqueue_function[k](
                TileTensor(q, qslay),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(qnw, qnlay),
                TileTensor(o, lay),
                Tq,
                q_offset,
                qstr,
                q_off,
                grid_dim=gd,
                block_dim=BLOCK,
            )
        else:
            comptime k = attn_cached_rope_kernel[type_of(lay), 14, 2, 64, False]
            ctx.enqueue_function[k](
                TileTensor(q, qslay),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(qnw, qnlay),
                TileTensor(o, lay),
                Tq,
                q_offset,
                qstr,
                q_off,
                grid_dim=gd,
                block_dim=BLOCK,
            )
        return o^

    # PREFILL (or flash decode): rotate Q into a contiguous qr buffer, then attend.
    var qr = ctx.enqueue_create_buffer[DType.float32](Tq * q_dim)
    var qlay = row_major(Tq * q_dim)
    if arch == 2:
        comptime kq = rope_q_kernel[type_of(qlay), 16, 128, True]
        ctx.enqueue_function[kq](
            TileTensor(q, qslay),
            TileTensor(qr, qlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            qstr,
            q_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 3:
        comptime kq = rope_q_kernel[type_of(qlay), 32, 128, True]
        ctx.enqueue_function[kq](
            TileTensor(q, qslay),
            TileTensor(qr, qlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            qstr,
            q_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 4:
        comptime kq = rope_q_kernel[type_of(qlay), 40, 128, True]
        ctx.enqueue_function[kq](
            TileTensor(q, qslay),
            TileTensor(qr, qlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            qstr,
            q_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 1:
        comptime kq = rope_q_kernel[type_of(qlay), 16, 128, False]
        ctx.enqueue_function[kq](
            TileTensor(q, qslay),
            TileTensor(qr, qlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            qstr,
            q_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime kq = rope_q_kernel[type_of(qlay), 14, 64, False]
        ctx.enqueue_function[kq](
            TileTensor(q, qslay),
            TileTensor(qr, qlay),
            TileTensor(qnw, qnlay),
            Tq,
            q_offset,
            qstr,
            q_off,
            THETA,
            -1,
            grid_dim=ceildiv(Tq * hq, BLOCK),
            block_dim=BLOCK,
        )

    if arch == 0 and q_offset + Tq > FLASH_THRESHOLD:
        # 0.5B long context: stream K/V through shared memory (bit-identical, no cliff).
        # Flash is 0.5B-only: HEAD_DIM=128 would double the staged tile to ~32 KB
        # (Metal threadgroup limit), so 3B always uses attn_cached_kernel.
        comptime kf = flash_attn_kernel[type_of(lay), 14, 2, 64, FLASH_PW]
        comptime nwarp = FLASH_PW * 7  # GROUP = 14/2
        ctx.enqueue_function[kf](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq, FLASH_PW) * hkv,
            block_dim=nwarp * WARP_SIZE,
        )
    elif Tq > 1:
        # PREFILL: tensor-core flash attention on the 8×8 simdgroup-matrix units —
        # one warp per (8-query tile, head). ~27–32× over the scalar attn_cached
        # (bit-exact). Grid = ceildiv(Tq,8)·HQ blocks, one warp (32 lanes) each.
        var grid = ceildiv(Tq, 8) * hq
        if arch == 2:
            comptime k = tc_attn_kernel[type_of(lay), 16, 8, 128]
            ctx.enqueue_function[k](
                TileTensor(qr, row_major(Tq * q_dim)),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(o, lay),
                Tq,
                q_offset,
                Float32(-1.0),
                0,
                grid_dim=grid,
                block_dim=WARP_SIZE,
            )
        elif arch == 3:
            comptime k = tc_attn_kernel[type_of(lay), 32, 8, 128]
            ctx.enqueue_function[k](
                TileTensor(qr, row_major(Tq * q_dim)),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(o, lay),
                Tq,
                q_offset,
                Float32(-1.0),
                0,
                grid_dim=grid,
                block_dim=WARP_SIZE,
            )
        elif arch == 4:
            comptime k = tc_attn_kernel[type_of(lay), 40, 8, 128]
            ctx.enqueue_function[k](
                TileTensor(qr, row_major(Tq * q_dim)),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(o, lay),
                Tq,
                q_offset,
                Float32(-1.0),
                0,
                grid_dim=grid,
                block_dim=WARP_SIZE,
            )
        elif arch == 1:
            comptime k = tc_attn_kernel[type_of(lay), 16, 2, 128]
            ctx.enqueue_function[k](
                TileTensor(qr, row_major(Tq * q_dim)),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(o, lay),
                Tq,
                q_offset,
                Float32(-1.0),
                0,
                grid_dim=grid,
                block_dim=WARP_SIZE,
            )
        else:
            comptime k = tc_attn_kernel[type_of(lay), 14, 2, 64]
            ctx.enqueue_function[k](
                TileTensor(qr, row_major(Tq * q_dim)),
                TileTensor(kc, row_major(cache_len)),
                TileTensor(vc, row_major(cache_len)),
                TileTensor(o, lay),
                Tq,
                q_offset,
                Float32(-1.0),
                0,
                grid_dim=grid,
                block_dim=WARP_SIZE,
            )
    elif arch == 2:
        # DECODE (Tq=1): warp-per-(query,head), keys split across the 32 lanes.
        comptime k = attn_cached_kernel[type_of(lay), 16, 8, 128]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 3:
        comptime k = attn_cached_kernel[type_of(lay), 32, 8, 128]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 4:
        comptime k = attn_cached_kernel[type_of(lay), 40, 8, 128]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    elif arch == 1:
        comptime k = attn_cached_kernel[type_of(lay), 16, 2, 128]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    else:
        comptime k = attn_cached_kernel[type_of(lay), 14, 2, 64]
        ctx.enqueue_function[k](
            TileTensor(qr, row_major(Tq * q_dim)),
            TileTensor(kc, row_major(cache_len)),
            TileTensor(vc, row_major(cache_len)),
            TileTensor(o, lay),
            Tq,
            q_offset,
            grid_dim=ceildiv(Tq * hq * WARP_SIZE, BLOCK),
            block_dim=BLOCK,
        )
    return o^


# ── decoder layer forward (the Qwen family forward; engine dispatches here) ─────


def qwen_layer(
    ctx: DeviceContext,
    mut w: Weights,
    l: Int,
    mut h: DevBuf,
    mut kc: DevBuf,
    mut vc: DevBuf,
    Tq: Int,
    q_offset: Int,
    cache_len: Int,
    mut dummy: DevBuf,
) raises -> DevBuf:
    """One Qwen decoder layer. Prefill = (Tq=P, q_offset=0); decode = (Tq=1,
    q_offset=pos). Dims come from `w`; behavior flags from `w.cfg`. Serves all
    three Qwen archs (0.5B/3B/Qwen3) via cfg flags + the arch dim-dispatch above.
    """
    var hd = w.hidden
    var nkv = w.nkv
    var qd = w.q_dim  # Q-proj width = hq*head_dim (== hd for Qwen2.5)
    var has_bias = 1 if w.cfg.qkv_bias else 0  # Qwen3 has no QKV bias
    # one GEMV for q|k|v; rope/copy/attn read the q,k,v slices in place (stride W).
    # ln1 RMSNorm is fused into the qkv GEMV at decode (mm_w_norm), saving a launch.
    var W = qd + 2 * nkv
    var qkv = mm_w_norm(
        ctx, h, w.ln1[l], w.qkv[l], w.qkv_b[l], Tq, hd, W, has_bias, w.simd_ok
    )
    # Qwen3 applies per-head QK-RMSNorm (qnorm/knorm) inside rope; Qwen2.5 passes
    # the same `dummy` (the kernels ignore it when qk_norm is false).
    var knw = w.knorm[l] if w.cfg.qk_norm else dummy
    var qnw = w.qnorm[l] if w.cfg.qk_norm else dummy
    # RoPE-K into the cache + V copy into the cache, fused into one launch (was 2).
    rope_kv(
        ctx,
        qkv,
        kc,
        vc,
        knw,
        Tq,
        q_offset,
        cache_len,
        w.hkv,
        w.head_dim,
        w.arch,
        W,
        qd,
        qd + nkv,
    )
    var o = attn_cached(
        ctx,
        qkv,
        kc,
        vc,
        qnw,
        Tq,
        q_offset,
        cache_len,
        w.hidden,
        w.hq,
        w.hkv,
        w.head_dim,
        w.arch,
        W,
        0,
    )  # Q slice [0:qd]
    var h2 = mm_w_add(
        ctx, o, w.ow[l], dummy, h, Tq, qd, hd, 0, w.simd_ok
    )  # o_proj(o)[q_dim→hd] + h
    # ln2 RMSNorm fused into the gate_up GEMV at decode.
    var gu = mm_w_norm(
        ctx,
        h2,
        w.ln2[l],
        w.gate_up[l],
        dummy,
        Tq,
        hd,
        2 * w.inter,
        0,
        w.simd_ok,
    )  # [gate|up]
    # down-proj with SwiGLU fused on input + residual on output (one launch at decode).
    return mm_w_silu_add(
        ctx, gu, w.down[l], h2, Tq, w.inter, hd, w.simd_ok
    )  # down(silu(gate)·up) + h2


def sess_embed(
    ctx: DeviceContext, mut w: Weights, prompt: List[Int]
) raises -> List[Float32]:
    """Qwen3-Embedding sentence vector for `prompt`: run the full decoder, take the
    LAST token's hidden state (official Qwen3-Embedding last-token pooling — the ids
    already carry the appended EOS), apply the final RMSNorm, then L2-normalize.
    Embedding-specific, so it lives in the Qwen module (not the generic engine).
    Runs its own one-shot Session (no KV reuse). Returns the D-element unit vector.
    """
    var P = len(prompt)
    var s = new_session(ctx, P + 2, w.nlayers, w.nkv)
    var ids_dev = upload_ids(ctx, prompt)
    var h = embed_tokens(ctx, ids_dev, w.embed, P, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = qwen_layer(
            ctx, w, l, h, s.kcs[l], s.vcs[l], P, 0, s.cache_len, s.dummy
        )
    var hl = last_row(ctx, h, P, w.hidden)
    var hn = rmsnorm(ctx, hl, w.final_norm, 1, w.hidden)
    ctx.synchronize()
    var out = List[Float32]()
    var ss = Float32(0.0)
    with hn.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.hidden))
        for i in range(w.hidden):
            var v = rebind[Scalar[DType.float32]](mt[i])
            out.append(v)
            ss += v * v
    var inv = Float32(1.0) / sqrt(ss)
    for i in range(len(out)):
        out[i] = out[i] * inv
    return out^
