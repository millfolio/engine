"""Speculative decode with the gemma-4 e2b DRAFT model verifying against the 12B
TARGET. Loads both (int4) co-resident, runs greedy `generate` (baseline) and
`generate_spec_draft` on the same prompts, asserts bit-identical output (greedy is
exact), and reports tok/s + draft acceptance + speedup. `pixi run e2b-spec`."""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from models.gemma import load_gemma_weights, GemmaWeights, G_NLAYERS
from models.gemma_e2b import load_e2b_weights, GemmaE2bWeights, E_NLAYERS
from runtime.engine import generate, generate_spec_draft
from chat import load_chat_template, render_value
from tokenizer import load_gemma_tokenizer_json, Tokenizer
from runtime.tensor_ops import probe_simd_gemm
from template import Template
from runtime.model_iface import FAMILY_GEMMA
from json import parse_json

comptime SNAP12B = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"
comptime SNAPE2B = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"
comptime TMPL = "assets/qwen2.5-chat-template.jinja"
comptime MAX_NEW = 96


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for byte in s.as_bytes():
        b.append(byte)
    return b^


def bench(ctx: DeviceContext, mut tgt: GemmaWeights, mut drf: GemmaE2bWeights,
          tok: Tokenizer, tmpl: Template, label: String, body: String) raises:
    var prompt = tok.encode(_bytes(render_value(tmpl, parse_json(body), FAMILY_GEMMA)))
    print("── ", label, " (prompt_toks=", len(prompt), ") ──", sep="")

    var t0 = perf_counter_ns()
    var base = generate(ctx, tgt, prompt, MAX_NEW)
    var t1 = perf_counter_ns()
    var spec = generate_spec_draft(ctx, tgt, drf, prompt, MAX_NEW, 4, True)
    var t2 = perf_counter_ns()

    var base_ms = Float64(t1 - t0) / 1.0e6
    var spec_ms = Float64(t2 - t1) / 1.0e6
    var n = len(base) if len(base) < len(spec) else len(spec)
    var ok = len(base) == len(spec)
    var diverge = -1
    for i in range(n):
        if base[i] != spec[i]:
            ok = False; diverge = i; break
    print("  baseline:   ", base_ms, " ms, ", len(base), " toks (",
          Float64(len(base)) / (base_ms / 1000.0), " tok/s)", sep="")
    print("  spec-draft: ", spec_ms, " ms, ", len(spec), " toks (",
          Float64(len(spec)) / (spec_ms / 1000.0), " tok/s, ", base_ms / spec_ms, "x)", sep="")
    if not ok:
        if diverge >= 0:
            print("  MISMATCH at ", diverge, ": base=", base[diverge], " spec=", spec[diverge], sep="")
        raise Error("draft-model spec decode does NOT match greedy — FAILED (" + label + ")")
    print("  OK — identical (", len(base), " toks)", sep="")


def main() raises:
    var ctx = DeviceContext()
    print("loading gemma tokenizer + both models (int4)…")
    var tok = load_gemma_tokenizer_json(SNAPE2B + "/tokenizer.json")
    var tmpl = load_chat_template(TMPL)
    var t12 = List[Int]()
    for i in range(G_NLAYERS): t12.append(i)
    var tgt = load_gemma_weights(ctx, SNAP12B, t12, True)
    tgt.simd_ok = probe_simd_gemm(ctx)
    var drf = load_e2b_weights(ctx, SNAPE2B, True)
    drf.simd_ok = tgt.simd_ok
    print("  target=12B (", G_NLAYERS, " layers)  draft=e2b (", E_NLAYERS, " layers)", sep="")

    bench(ctx, tgt, drf, tok, tmpl, "code-rewrite",
        '{"messages":[{"role":"user","content":"Fix the bug in this Python function and return the complete corrected function, nothing else:\\n\\ndef factorial(n):\\n    result = 1\\n    for i in range(n):\\n        result = result * i\\n    return result\\n"}]}')
    bench(ctx, tgt, drf, tok, tmpl, "prose",
        '{"messages":[{"role":"user","content":"Explain in a few sentences why the sky is blue."}]}')
