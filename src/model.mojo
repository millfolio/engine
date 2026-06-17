"""Facade: the model/runtime API re-exported from the split modules so existing
`from model import …` call sites (server, main, tests) keep working unchanged.

The implementation now lives in focused, reusable modules:
  - `kernels.mojo`     — pure GPU kernels (GEMM, attention, RoPE, norms, …)
  - `tensor_ops.mojo`  — host-side op launchers + weight types (QMat, DevBuf, …)
  - `safetensors.mojo` — checkpoint I/O (header parse, shard gather, bf16/int4 load)
  - `sampling.mojo`    — logit processing + token sampling (CPU)
  - `engine.mojo`      — model-agnostic Session / prefill / decode / generate loop
  - `qwen.mojo`        — the Qwen model family (Weights, ModelConfig, load_weights,
                         qwen_layer); add a sibling module per new family

New code should import from the specific module; this facade only preserves the
old surface."""

from sampling import Dist, process_logits, next_rand, sample, argmax_f
from tensor_ops import (
    BLOCK, DevBuf, WBuf, PBuf, QMat, qmat_bf16,
    mm, mm_w, mm_w_add, mm_norm, mm_w_norm, mm_w_silu_add, probe_simd_gemm,
    rmsnorm, add, silu_mul, silu_mul_cat, embed_tokens, last_row, copy_into, copy_strided,
)
from safetensors import (
    TensorEntry, parse_header, read_header, load_one, load_named,
    load_one_bf16, load_named_bf16, load_one_q4, load_proj, fuse_pair,
    concat_bias, gather_tensors,
)
from model_iface import ModelConfig, ModelWeights, FAMILY_QWEN, FAMILY_GEMMA, ACT_SILU, ACT_GELU, TOOL_QWEN, TOOL_GEMMA
from qwen import (
    Weights, load_weights, rope_k, rope_kv, attn_cached, sess_embed,
    qwen_layer, qwen_layer as layer_cached, EOS1, EOS2, FLASH_THRESHOLD,
)
from engine import (
    Session, new_session, sess_prefill, sess_prefill_suffix,
    sess_step, generate, generate_sample, generate_spec, generate_spec_draft, sess_verify,
    _ngram_draft, _argmax_row, upload_ids, argmax_last, logits_last,
)
