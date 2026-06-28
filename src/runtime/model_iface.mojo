"""The model-family interface: `ModelConfig` (behavior flags + engine-relevant
dims) and the `ModelWeights` trait that each family's weight struct conforms to.
Lives in its own module so `qwen`/`gemma`/`engine` all import it without a cycle
(tensor_ops ← model_iface ← {qwen, gemma, engine}).

The engine is parametric over `ModelWeights`: it drives the KV-cache session,
the generate loop, and sampling, while each family supplies how to embed a
prompt, run one decoder layer, and produce final logits (incl. any softcap).
Adding a family = a new weights struct conforming to this trait + its loader."""

from std.gpu.host import DeviceContext, DeviceBuffer
from runtime.tensor_ops import DevBuf

# Model-family tags (ModelConfig.family) — the engine is generic, but a few
# spots still branch on family for diagnostics / banners.
comptime FAMILY_QWEN = 0
comptime FAMILY_GEMMA = 1

# Activation tags.
comptime ACT_SILU = 0
comptime ACT_GELU = 1

# Tool-call / chat post-processing style (server reads ModelConfig.tool_style so
# it never branches on family for behavior).
comptime TOOL_QWEN = 0  # <tool_call>…</tool_call> JSON (parse_tool_calls)
comptime TOOL_GEMMA = 1  # ```tool_code``` + thinking channel (parse_gemma_tool_calls)


@fieldwise_init
struct ModelConfig(ImplicitlyCopyable, Movable):
    """Per-model behavior flags + the dims the model-agnostic engine needs
    (nlayers/nkv for cache sizing, eos for the stop check). Family-specific dims
    (hidden, head_dim, …) live in the concrete weights struct. Qwen leaves the
    Gemma-only knobs off (act=SiLU, softcaps=0, sliding_window=0, norm_offset=0,
    embed_scale=1)."""

    var family: Int  # FAMILY_QWEN / FAMILY_GEMMA
    var nlayers: Int
    var nkv: Int  # K/V row width (hkv*head_dim) — for KV-cache sizing
    var qkv_bias: Bool
    var qk_norm: Bool
    var act: Int  # ACT_SILU / ACT_GELU
    var attn_softcap: Float32  # 0 = off
    var final_softcap: Float32  # 0 = off (Gemma final-logit softcap)
    var sliding_window: Int  # 0 = global attention
    var rope_theta: Float32
    var embed_scale: Float32  # 1.0 = none (Gemma scales embeddings by √hidden)
    var norm_offset: Float32  # 0.0 Qwen, 1.0 Gemma ((1+w) RMSNorm)
    var eos1: Int
    var eos2: Int
    var tool_style: Int  # TOOL_QWEN / TOOL_GEMMA (chat post-processing)
    var extra_stop: Int  # extra stop token id (-1 = none; Gemma's <|tool_response>)


trait ModelWeights(Movable):
    """What the engine needs from any model family. The family struct holds the
    buffers + dims and implements: expose its config, embed a prompt (+ any
    scaling), run one decoder layer into the KV cache, and produce the last
    position's logits (+ any final softcap)."""

    def config(self) -> ModelConfig:
        ...

    def embed_prompt(
        mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int
    ) raises -> DevBuf:
        ...

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
        # Receives ALL per-layer K/V caches (not just layer l's) so a family can
        # implement cross-layer KV sharing (Gemma-4 e2b). Dense families index [l].
        ...

    def lm_logits(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        ...

    def lm_logits_all(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        # Logits for ALL T positions (row-major T×vocab), for speculative-decode
        # batch verification. Same head as `lm_logits` but over every row, not just
        # the last. Returns a flat host list of length T*vocab.
        ...

    def token_logprobs(
        mut self,
        ctx: DeviceContext,
        mut h: DevBuf,
        n: Int,
        targets: List[Int],
        mut dummy: DevBuf,
    ) raises -> List[Float32]:
        # log P(targets[i] | h[i]) for the first n rows — the LM head (+ any softcap)
        # followed by an on-GPU per-row log-softmax-of-target (nll_gather), so the
        # n×vocab logits never reach the host. For perplexity / echo logprobs.
        ...
