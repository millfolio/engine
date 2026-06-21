"""Speculative-decode validation + timing on Gemma 4 12B-it (int4).

Loads all 48 layers, then runs the SAME greedy prompts two ways:
  1. `generate`       — the baseline one-token-per-forward greedy loop.
  2. `generate_spec`  — prompt-lookup (n-gram) speculative decode.

For greedy, spec decode is exact: every accepted token equals the target's own
argmax, so the two token streams MUST be bit-identical. We assert that, then
report wall-clock + tok/s for each. Prompt-lookup drafting pays off when the
output echoes spans already in the context (code edits, reformatting, quoting),
so we use an echo-heavy code-rewrite prompt as the headline case plus a plain
prose prompt as a control.

    pixi run spec-decode
"""
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from models.gemma import load_gemma_weights, GemmaWeights, G_NLAYERS
from runtime.engine import generate, generate_spec, new_session, sess_verify, sess_prefill
from chat import load_chat_template, render_value
from tokenizer import load_gemma_tokenizer_json, Tokenizer
from runtime.tensor_ops import probe_simd_gemm
from template import Template
from runtime.model_iface import FAMILY_GEMMA
from json import parse_json

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"
comptime TMPL = "assets/qwen2.5-chat-template.jinja"   # placeholder; render_value(FAMILY_GEMMA) renders in Mojo
comptime MAX_NEW = 200


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for byte in s.as_bytes():
        b.append(byte)
    return b^


def _check(base: List[Int], spec: List[Int], label: String) raises:
    var n = len(base) if len(base) < len(spec) else len(spec)
    var ok = len(base) == len(spec)
    var diverge = -1
    for i in range(n):
        if base[i] != spec[i]:
            ok = False; diverge = i; break
    if not ok:
        if diverge >= 0:
            print("  MISMATCH at ", diverge, ": base=", base[diverge], " spec=", spec[diverge], sep="")
        raise Error("speculative decode does NOT match greedy — FAILED (" + label + ")")


def bench_prompt(ctx: DeviceContext, mut gw: GemmaWeights, tok: Tokenizer,
                 tmpl: Template, label: String, body: String) raises:
    var rendered = render_value(tmpl, parse_json(body), FAMILY_GEMMA)
    var prompt = tok.encode(_bytes(rendered))
    print("── ", label, " (prompt_toks=", len(prompt), ") ──", sep="")

    var t0 = perf_counter_ns()
    var base = generate(ctx, gw, prompt, MAX_NEW)
    var t1 = perf_counter_ns()
    var base_ms = Float64(t1 - t0) / 1.0e6
    print("  baseline:    ", base_ms, " ms, ", len(base), " toks (",
          Float64(len(base)) / (base_ms / 1000.0), " tok/s)", sep="")

    for K in [4, 7, 11, 15]:
        var ta = perf_counter_ns()
        var spec = generate_spec(ctx, gw, prompt, MAX_NEW, K, 3, True)
        var tb = perf_counter_ns()
        var spec_ms = Float64(tb - ta) / 1.0e6
        _check(base, spec, label)
        print("  K=", K, ": ", spec_ms, " ms (", Float64(len(spec)) / (spec_ms / 1000.0),
              " tok/s, ", base_ms / spec_ms, "x)", sep="")


def main() raises:
    var ctx = DeviceContext()
    print("loading gemma tokenizer + weights (int4, 48 layers)…")
    var tok = load_gemma_tokenizer_json(SNAP + "/tokenizer.json")
    var tmpl = load_chat_template(TMPL)
    var alllayers = List[Int]()
    for i in range(G_NLAYERS): alllayers.append(i)
    var gw = load_gemma_weights(ctx, SNAP, alllayers, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    print("  simd_ok=", gw.simd_ok, sep="")

    # M-scaling micro-bench: how does ONE target forward scale with batch size Q?
    # If decode were purely bandwidth-bound on weight reads (reused across rows),
    # a Q=8 forward would cost ~Q=1. If it scales ~linearly, batched verification
    # cannot beat single-token decode no matter the draft acceptance.
    var warm: List[Int] = [2,105,2364,107,1567,506,1171,2390,46501,699,506,3768,236764,55348,15914,236761,106,107,105,4368,107]
    var s = new_session(ctx, 512, gw.config().nlayers, gw.config().nkv)
    _ = sess_prefill(ctx, gw, s, warm)
    var base_pos = s.pos
    print("forward latency vs batch size Q (at pos ", base_pos, "):", sep="")
    for q in [1, 2, 4, 8, 12, 16, 21]:
        var batch = List[Int]()
        for i in range(q): batch.append(warm[i])
        s.pos = base_pos
        _ = sess_verify(ctx, gw, s, batch)   # warmup
        var rep = 5
        var tt0 = perf_counter_ns()
        for _ in range(rep):
            s.pos = base_pos
            _ = sess_verify(ctx, gw, s, batch)
        var tt1 = perf_counter_ns()
        var ms = Float64(tt1 - tt0) / 1.0e6 / Float64(rep)
        print("  Q=", q, ": ", ms, " ms/forward (", ms / Float64(q), " ms/token)", sep="")
    s.pos = base_pos

    # Echo-heavy: the corrected code repeats most of the given snippet verbatim,
    # so prompt-lookup drafts long accepted runs.
    var code = String('{"messages":[{"role":"user","content":"Fix the bug in this Python function and return the complete corrected function, nothing else:\\n\\ndef factorial(n):\\n    result = 1\\n    for i in range(n):\\n        result = result * i\\n    return result\\n"}]}')
    bench_prompt(ctx, gw, tok, tmpl, "code-rewrite (echo-heavy)", code)

    # Control: free-form prose, little verbatim echo — drafts rarely hit.
    var prose = String('{"messages":[{"role":"user","content":"Explain in a few sentences why the sky is blue."}]}')
    bench_prompt(ctx, gw, tok, tmpl, "prose (control)", prose)
