"""The model-family interface: `ModelConfig` (behavior flags + engine-relevant
dims) and the `ModelWeights` trait that each family's weight struct conforms to.
Lives in its own module so `qwen`/`gemma`/`engine` all import it without a cycle
(tensor_ops ← model_iface ← {qwen, gemma, engine}).

The engine is parametric over `ModelWeights`: it drives the KV-cache session,
the generate loop, and sampling, while each family supplies how to embed a
prompt, run one decoder layer, and produce final logits (incl. any softcap).
Adding a family = a new weights struct conforming to this trait + its loader."""

from max.gpu.host import DeviceContext, DeviceBuffer
from runtime.tensor_ops import DevBuf, KVBuf

# Model-family tags (ModelConfig.family) — the engine is generic, but a few
# spots still branch on family for diagnostics / banners.
comptime FAMILY_QWEN = 0
"""Model-family tag for Qwen."""
comptime FAMILY_GEMMA = 1
"""Model-family tag for Gemma."""

# Activation tags.
comptime ACT_SILU = 0
"""MLP activation tag: SiLU (Qwen)."""
comptime ACT_GELU = 1
"""MLP activation tag: GELU (Gemma)."""

# Tool-call / chat post-processing style (server reads ModelConfig.tool_style so
# it never branches on family for behavior).
comptime TOOL_QWEN = 0  # <tool_call>…</tool_call> JSON (parse_tool_calls)
"""Tool-call style tag: Qwen `<tool_call>…</tool_call>` JSON."""
comptime TOOL_GEMMA = 1  # ```tool_code``` + thinking channel (parse_gemma_tool_calls)
"""Tool-call style tag: Gemma `tool_code` fenced blocks + thinking channel."""


@fieldwise_init
struct ModelConfig(ImplicitlyCopyable, Movable):
    """Per-model behavior flags + the dims the model-agnostic engine needs
    (nlayers/nkv for cache sizing, eos for the stop check). Family-specific dims
    (hidden, head_dim, …) live in the concrete weights struct. Qwen leaves the
    Gemma-only knobs off (act=SiLU, softcaps=0, sliding_window=0, norm_offset=0,
    embed_scale=1)."""

    var family: Int  # FAMILY_QWEN / FAMILY_GEMMA
    """Model-family tag (FAMILY_QWEN / FAMILY_GEMMA)."""
    var nlayers: Int
    """Number of decoder layers."""
    var nkv: Int  # K/V row width (hkv*head_dim) — for KV-cache sizing
    """K/V row width (hkv*head_dim), used for KV-cache sizing."""
    var qkv_bias: Bool
    """Whether the Q/K/V projections include bias terms."""
    var qk_norm: Bool
    """Whether per-head Q/K RMSNorm is applied."""
    var act: Int  # ACT_SILU / ACT_GELU
    """MLP activation tag (ACT_SILU / ACT_GELU)."""
    var attn_softcap: Float32  # 0 = off
    """Attention-logit softcap (0 = off)."""
    var final_softcap: Float32  # 0 = off (Gemma final-logit softcap)
    """Final-logit softcap (0 = off; Gemma only)."""
    var sliding_window: Int  # 0 = global attention
    """Sliding-window attention size (0 = global attention)."""
    var rope_theta: Float32
    """RoPE base frequency (theta)."""
    var embed_scale: Float32  # 1.0 = none (Gemma scales embeddings by √hidden)
    """Embedding scale factor (1.0 = none; Gemma scales by √hidden)."""
    var norm_offset: Float32  # 0.0 Qwen, 1.0 Gemma ((1+w) RMSNorm)
    """RMSNorm weight offset (0.0 Qwen, 1.0 Gemma's (1+w) form)."""
    var eos1: Int
    """Primary end-of-sequence token id."""
    var eos2: Int
    """Secondary end-of-sequence token id."""
    var tool_style: Int  # TOOL_QWEN / TOOL_GEMMA (chat post-processing)
    """Chat/tool-call post-processing style (TOOL_QWEN / TOOL_GEMMA)."""
    var extra_stop: Int  # extra stop token id (-1 = none; Gemma's <|tool_response>)
    """Extra stop token id (-1 = none; Gemma's <|tool_response>)."""


trait ModelWeights(Movable):
    """What the engine needs from any model family. The family struct holds the
    buffers + dims and implements: expose its config, embed a prompt (+ any
    scaling), run one decoder layer into the KV cache, and produce the last
    position's logits (+ any final softcap)."""

    def config(self) -> ModelConfig:
        """Return this model's `ModelConfig` (behavior flags + engine-relevant dims).

        Returns:
            This model's `ModelConfig` (behavior flags + engine-relevant dims).
        """
        ...

    def embed_prompt(
        mut self, ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], T: Int
    ) raises -> DevBuf:
        """Embed `T` prompt token ids into the hidden-state buffer (+ any scaling).

        Args:
            ctx: The GPU device context.
            ids: The prompt token ids (length `T`).
            T: The number of prompt tokens to embed.

        Returns:
            The hidden-state buffer for the `T` positions (with any embedding
            scaling applied).

        Raises:
            On device/compute errors.
        """
        ...

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
        """Run decoder layer `l` over hidden state `h`, updating the KV caches; return the new state.

        Args:
            ctx: The GPU device context.
            l: The decoder-layer index to run.
            h: The input hidden state for this layer.
            kcs: All layers' key caches (updated in place for layer `l`).
            vcs: All layers' value caches (updated in place for layer `l`).
            Tq: The number of query positions in this call.
            q_offset: The position offset of the query block within the sequence.
            cache_len: The current KV-cache length (keys/values already stored).
            dummy: A scratch buffer reused across calls.

        Returns:
            The new hidden state after layer `l`.

        Raises:
            On device/compute errors.
        """
        # Receives ALL per-layer K/V caches (not just layer l's) so a family can
        # implement cross-layer KV sharing (Gemma-4 e2b). Dense families index [l].
        ...

    def lm_logits(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        """Produce the last position's vocab logits (+ any final softcap).

        Args:
            ctx: The GPU device context.
            h: The final hidden state.
            T: The number of positions in `h`.
            dummy: A scratch buffer reused across calls.

        Returns:
            The last position's vocab logits (with any final softcap applied).

        Raises:
            On device/compute errors.
        """
        ...

    def lm_logits_dev(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> DevBuf:
        """Produce the last position's vocab logits ON DEVICE (+ any final
        softcap). No synchronize, no host copy — feed GPU token selection.

        Args:
            ctx: The GPU device context.
            h: The final hidden state.
            T: The number of positions in `h`.
            dummy: A scratch buffer reused across calls.

        Returns:
            The last position's vocab logits as a device buffer.

        Raises:
            On device/compute errors.
        """
        ...

    def lm_logits_all_dev(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> DevBuf:
        """Produce logits for all `T` positions ON DEVICE (row-major T×vocab).

        Args:
            ctx: The GPU device context.
            h: The final hidden state for all `T` positions.
            T: The number of positions in `h`.
            dummy: A scratch buffer reused across calls.

        Returns:
            The `T`×vocab logits as one device buffer (row-major).

        Raises:
            On device/compute errors.
        """
        ...

    def lm_logits_all(
        mut self, ctx: DeviceContext, mut h: DevBuf, T: Int, mut dummy: DevBuf
    ) raises -> List[Float32]:
        """Produce logits for all `T` positions (row-major T×vocab) for spec-decode verification.

        Args:
            ctx: The GPU device context.
            h: The final hidden state for all `T` positions.
            T: The number of positions in `h`.
            dummy: A scratch buffer reused across calls.

        Returns:
            A flat host list of logits for all `T` positions (row-major
            T×vocab), length `T*vocab`.

        Raises:
            On device/compute errors.
        """
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
        """Return log P(targets[i] | h[i]) for the first `n` rows (LM head + on-GPU per-row log-softmax).

        Args:
            ctx: The GPU device context.
            h: The hidden states (one row per scored position).
            n: The number of rows to score.
            targets: The target token id for each row.
            dummy: A scratch buffer reused across calls.

        Returns:
            Log P(targets[i] | h[i]) for the first `n` rows.

        Raises:
            On device/compute errors.
        """
        # log P(targets[i] | h[i]) for the first n rows — the LM head (+ any softcap)
        # followed by an on-GPU per-row log-softmax-of-target (nll_gather), so the
        # n×vocab logits never reach the host. For perplexity / echo logprobs.
        ...
