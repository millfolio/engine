"""Perplexity smoke test for a Qwen checkpoint — a regression guard for the whole
load + tokenize + serve stack. Mirrors what the HTTP server's /v1/completions does:
tokenize a fixed sentence, build a KV session sized like the server (MAX_SEQ), run
the teacher-forced forward (sess_token_logprobs), and assert the resulting
perplexity is in a healthy range.

Catches every bug hit during Qwen3 bring-up, each of which left an unmistakable PPL
signature:
  * broken BPE merges (per-character tokenization) -> token count ~= char count
  * wrong/missing lm_head, zero hidden, or KV over-commit -> uniform logits, so
    PPL ~= vocab size (~151936)
  * a healthy model lands around 8-13 on this sentence.

  pixi run ppl-smoke                          # Qwen2.5-3B-Instruct int4 (default)
  pixi run -- mojo ... ppl_smoke.mojo Qwen/Qwen3-8B 50
"""

from std.sys import argv
from std.os import getenv
from std.math import exp, isfinite
from std.gpu.host import DeviceContext
from models.qwen import load_weights
from runtime.engine import new_session, sess_token_logprobs
from tokenizer import load_tokenizer_json, Tokenizer
from runtime.tensor_ops import probe_simd_gemm

comptime SESSION_LEN = 4096       # fits every model's KV; correctness, not deploy-fit
                                  # (the server's per-model max_seq cap handles fit)
comptime SENTENCE = "The capital of France is Paris, and the capital of Japan is Tokyo."
comptime DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
comptime DEFAULT_THRESHOLD = Float64(50.0)   # healthy ~8-13; broken is 100s..1e9


def _slug(model_id: String) -> String:
    var b = model_id.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        if b[i] == 47:                     # '/'
            out.append(45); out.append(45)
        else:
            out.append(b[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _snapshot(model: String) raises -> String:
    """Resolve an HF id to its cached snapshot dir; pass a literal path through."""
    if "/snapshots/" in model:
        return model
    var home = String(getenv("HF_HOME"))
    var hub = (home + "/hub") if home.byte_length() > 0 else (String(getenv("HOME")) + "/.cache/huggingface/hub")
    var repo = hub + "/models--" + _slug(model)
    var commit: String
    with open(repo + "/refs/main", "r") as f:
        commit = String(f.read().strip())
    return repo + "/snapshots/" + commit


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for byte in s.as_bytes():
        b.append(byte)
    return b^


def main() raises:
    var a = argv()
    var model = String(a[1]) if len(a) > 1 else String(DEFAULT_MODEL)
    var threshold = Float64(atof(String(a[2]))) if len(a) > 2 else DEFAULT_THRESHOLD
    var snap = _snapshot(model)
    print("ppl-smoke:", model, " (int4, threshold PPL <", threshold, ")")

    var ctx = DeviceContext()
    var tok = load_tokenizer_json(snap + "/tokenizer.json")
    var ids = tok.encode(_bytes(SENTENCE))
    var nchars = len(SENTENCE)
    print("  tokens:", len(ids), " for", nchars, "chars")

    var ok = True
    # Sanity 1: real BPE merges to ~1 token / 4-5 chars; per-char tokenization
    # (the Qwen3 merges-format regression) would give ~1 token / char.
    if len(ids) > nchars // 2:
        print("  FAIL: tokenization looks per-character (", len(ids), "tokens for",
              nchars, "chars) — BPE merges not applied")
        ok = False

    var w = load_weights(ctx, snap, True)
    w.simd_ok = probe_simd_gemm(ctx)
    var s = new_session(ctx, SESSION_LEN, w.config().nlayers, w.config().nkv)
    var lp = sess_token_logprobs(ctx, w, s, ids)
    var nll = Float64(0.0)
    for i in range(len(lp)):
        nll += -Float64(lp[i])
    var ppl = exp(nll / Float64(len(lp)))
    print("  PPL:", ppl, " over", len(lp), "scored tokens")

    # Sanity 2: a healthy model lands in ~[2, threshold). Too HIGH => uniform logits
    # (broken forward / lm_head / KV over-commit, PPL ~= vocab ~151936). Too LOW
    # (PPL ~ 1) => degenerate/overconfident logits (NaN/Inf, e.g. an int4 overflow),
    # which can't happen for natural text.
    if not isfinite(ppl) or ppl >= threshold or ppl < 1.5:
        print("  FAIL: PPL", ppl, "outside healthy [1.5,", threshold,
              ") — broken/degenerate forward, lm_head, or KV")
        ok = False

    if ok:
        print("PASS")
    else:
        print("FAILED")
        raise Error("ppl-smoke failed for " + model)
