"""Per-phase decode-step profile: where do the milliseconds of one token go?

Replicates sess_step with ctx.synchronize() at phase boundaries — embed |
layers (all 36, pipelined) | lm_head/logits | host argmax — over PROFILE_STEPS
decode steps after a real prefill, reporting per-phase medians. Sync points
serialize the GPU pipeline, so the phase sum can exceed a production step;
treat the numbers as ATTRIBUTION (upper bounds per phase), and the no-sync
whole-step timing (also printed) as ground truth.

Run against a specific checkpoint (defaults to meta.txt like the other manual
gates); QWEN_Q4=1-style int4 is forced on since that's the shipping config.

    QWEN_SAFETENSORS=<snapshot-dir> pixi run decode-profile
"""

from std.time import perf_counter_ns
from std.os import getenv
from max.gpu.host import DeviceContext

from std.math import log, exp

from model import (
    load_weights,
    probe_gemv_w1,
    new_session,
    sess_prefill,
    sess_step,
    argmax_f,
)
from tokenizer import load_tokenizer
from chat import load_chat_template, render_chat
from runtime.engine import upload_ids


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"


def to_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        out.append(b[i])
    return out^


comptime PROFILE_STEPS = 32


def med(mut v: List[Float64]) -> Float64:
    """Median (sorts in place)."""
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


def main() raises:
    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        # FALLBACK IS THE 0.5B FIXTURE MODEL — set QWEN_SAFETENSORS for real
        # measurements (this bit us: a 0.5B number was read as the 3B's).
        ckpt = String(
            String(
                read_text("tests/fixtures/forward/meta.txt").split("\n")[1]
            ).strip()
        )
        print("WARNING: QWEN_SAFETENSORS unset — using the 0.5B fixture model")
    print("checkpoint: ", ckpt, sep="")
    var user = String(
        "Explain how a hash map works and why lookups are fast. Then write a"
        " short Python example."
    )
    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ids = tok.encode(to_bytes(render_chat(tmpl, user)))
    var ctx = DeviceContext()

    print("loading int4 weights…")
    var w = load_weights(ctx, ckpt, True)
    probe_gemv_w1(ctx)
    var nlayers = w.config().nlayers
    var s = new_session(ctx, len(ids) + PROFILE_STEPS + 8, nlayers, w.nkv)
    var lg = sess_prefill(ctx, w, s, ids)
    var nxt = argmax_f(lg)
    print(
        "prefilled ",
        len(ids),
        " tokens; profiling ",
        PROFILE_STEPS,
        " decode steps…",
        sep="",
    )

    # Ground truth first: whole steps, no internal syncs (production shape).
    var whole = List[Float64]()
    for _ in range(PROFILE_STEPS):
        var t0 = perf_counter_ns()
        lg = sess_step(ctx, w, s, nxt)
        nxt = argmax_f(lg)
        whole.append(Float64(perf_counter_ns() - t0) / 1.0e6)

    # Attribution: the same step with syncs at phase boundaries.
    var t_embed = List[Float64]()
    var t_layers = List[Float64]()
    var t_logits = List[Float64]()
    var t_host = List[Float64]()
    for _ in range(PROFILE_STEPS):
        var t0 = perf_counter_ns()
        var one = upload_ids(ctx, [nxt])
        var h = w.embed_prompt(ctx, one, 1)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for l in range(nlayers):
            h = w.run_layer(
                ctx, l, h, s.kcs, s.vcs, 1, s.pos, s.cache_len, s.dummy
            )
        ctx.synchronize()
        var t2 = perf_counter_ns()
        s.pos += 1
        lg = w.lm_logits(ctx, h, 1, s.dummy)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        nxt = argmax_f(lg)
        var t4 = perf_counter_ns()
        t_embed.append(Float64(t1 - t0) / 1.0e6)
        t_layers.append(Float64(t2 - t1) / 1.0e6)
        t_logits.append(Float64(t3 - t2) / 1.0e6)
        t_host.append(Float64(t4 - t3) / 1.0e6)

    var wm = med(whole)
    print(
        "whole step (no syncs, production shape): ",
        wm,
        " ms  = ",
        1000.0 / wm,
        " tok/s",
        sep="",
    )
    print("attributed phases (sync at boundaries — upper bounds):")
    print("  embed+upload : ", med(t_embed), " ms", sep="")
    print("  ", nlayers, " layers    : ", med(t_layers), " ms", sep="")
    print("  lm_head+logits DOWNLOAD: ", med(t_logits), " ms", sep="")
    print("  host argmax  : ", med(t_host), " ms", sep="")
