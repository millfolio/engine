"""Facade: the single stable public API for the engine. `model` is the one import
surface external consumers use (`from model import …` in the entry points and the
`-I src` test gates); it re-exports the engine's public symbols so the library can
be reorganized internally without touching call sites.

The implementation lives in package subdirs of `src/`, each imported here by its
package-qualified path. This facade is the ONLY top-level re-export module — there
is no per-module shim. Internal cross-imports (inside `models/`, `runtime/`,
`chat/`, `io/`) use the package-qualified form directly (e.g.
`from runtime.tensor_ops import …`, `from models.gemma import …`); `kernels` and
`tokenizer` are single-module packages imported bare (`from kernels import …`).

Package layout:
  - `kernels/`   — pure GPU kernels (GEMM, attention, RoPE, norms, …); code in __init__
  - `runtime/`   — tensor_ops (op launchers + weight types QMat/DevBuf), safetensors
                   (checkpoint I/O), sampling (logit processing), engine (model-agnostic
                   Session / prefill / decode / generate), model_iface (ModelConfig,
                   ModelWeights trait, family/act/tool constants)
  - `models/`    — the model families: qwen, gemma, gemma_e2b (Weights + load_weights +
                   per-layer forward); add a sibling module per new family
  - `chat/`      — chat-template render (chat), gemma render (gemma_chat/gemma_tools),
                   tool-call parsing (toolcall)
  - `tokenizer/` — byte-level BPE tokenizer; code in __init__
  - `io/`        — disk block KV-cache (blockcache)

Prefer importing from `model`; this facade is the surface to keep stable as the
packages evolve."""

from runtime.sampling import Dist, process_logits, next_rand, sample, argmax_f
from runtime.tensor_ops import (
    BLOCK,
    DevBuf,
    WBuf,
    PBuf,
    QMat,
    qmat_bf16,
    mm,
    mm_w,
    mm_w_add,
    mm_norm,
    mm_w_norm,
    mm_w_silu_add,
    probe_simd_gemm,
    rmsnorm,
    add,
    silu_mul,
    silu_mul_cat,
    embed_tokens,
    last_row,
    copy_into,
    copy_strided,
)
from runtime.safetensors import (
    TensorEntry,
    parse_header,
    read_header,
    load_one,
    load_named,
    load_one_bf16,
    load_named_bf16,
    load_one_q4,
    load_proj,
    fuse_pair,
    concat_bias,
    gather_tensors,
)
from runtime.model_iface import (
    ModelConfig,
    ModelWeights,
    FAMILY_QWEN,
    FAMILY_GEMMA,
    ACT_SILU,
    ACT_GELU,
    TOOL_QWEN,
    TOOL_GEMMA,
)
from models.qwen import (
    Weights,
    load_weights,
    rope_k,
    rope_kv,
    attn_cached,
    sess_embed,
    qwen_layer,
    qwen_layer as layer_cached,
    EOS1,
    EOS2,
    FLASH_THRESHOLD,
)
from models.gemma import GemmaWeights, load_gemma_weights, G_NLAYERS
from runtime.engine import (
    Session,
    new_session,
    sess_prefill,
    sess_prefill_suffix,
    sess_step,
    sess_token_logprobs,
    generate,
    generate_sample,
    generate_spec,
    generate_spec_draft,
    sess_verify,
    _ngram_draft,
    _argmax_row,
    upload_ids,
    argmax_last,
    logits_last,
)
from chat.toolcall import (
    ToolCall,
    ParsedReply,
    repair_json,
    parse_tool_calls,
    parse_gemma_tool_calls,
)
from io.blockcache import BlockCache
