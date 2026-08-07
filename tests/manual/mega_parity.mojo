"""Parity gate for the decode MEGAKERNEL A/B (qwen.qwen_layer_mega).

Runs the SAME greedy decode twice from an identical prefill — once on the
production per-op path (MILLFOLIO_DECODE_MEGA off) and once on the fused
megakernel path (on) — and asserts they agree: same next token every step, and
per-step logits within tolerance. Prefill (Tq>1) is ALWAYS the per-op path (the
mega branch is decode-only, Tq==1), so any divergence is the megakernel's.

While `qwen_layer_mega` merely delegates (the skeleton) this is bit-identical
(max|Δlogit| ~ 0). It becomes the real gate the moment a fused kernel replaces
the delegate: a wrong fusion trips it here instead of silently degrading output.

    QWEN_SAFETENSORS=<snapshot-dir> pixi run mega-parity
"""

from std.os import getenv
from max.gpu.host import DeviceContext

from model import (
    Weights,
    load_weights,
    new_session,
    sess_prefill,
    sess_step,
    argmax_f,
)
from tokenizer import load_tokenizer
from chat import load_chat_template, render_chat

comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"
comptime STEPS = 16  # decode steps to compare
comptime TOL = Float32(1.0e-3)  # max |Δlogit| allowed (0 while delegating)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def to_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        out.append(b[i])
    return out^


@fieldwise_init
struct DecodeOut(Movable):
    """One decode run's result: the tokens produced + the final step's logits."""

    var toks: List[Int]
    var last: List[Float32]


def _decode_tokens(
    ctx: DeviceContext,
    mut w: Weights,
    ids: List[Int],
    ncap: Int,
) raises -> DecodeOut:
    """Greedy-decode STEPS tokens; return the tokens produced and the FINAL
    step's logits (for a numeric compare). `w.decode_mega` is set by the caller."""
    var s = new_session(ctx, ncap, w.config().nlayers, w.nkv)
    var lg = sess_prefill(ctx, w, s, ids)
    var toks = List[Int]()
    var last = lg.copy()
    var t = argmax_f(lg)
    for _ in range(STEPS):
        toks.append(t)
        last = sess_step(ctx, w, s, t)
        t = argmax_f(last)
    return DecodeOut(toks^, last^)


def main() raises:
    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        ckpt = String(
            String(
                read_text("tests/fixtures/forward/meta.txt").split("\n")[1]
            ).strip()
        )
    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ids = tok.encode(
        to_bytes(
            render_chat(tmpl, String("Give me three uses for a paperclip."))
        )
    )
    var ncap = len(ids) + STEPS + 8

    var ctx = DeviceContext()
    print("loading int4 weights…")
    var w = load_weights(ctx, ckpt, True)

    print("decode A — per-op path (decode_mega=False)…")
    w.decode_mega = False
    var ra = _decode_tokens(ctx, w, ids, ncap)

    print("decode B — megakernel path (decode_mega=True)…")
    w.decode_mega = True
    var rb = _decode_tokens(ctx, w, ids, ncap)

    # 1) token-for-token agreement across the whole decode (index in place — the
    # Lists aren't ImplicitlyCopyable, so don't bind them to locals).
    var mismatch = -1
    for i in range(STEPS):
        if ra.toks[i] != rb.toks[i]:
            mismatch = i
            break

    # 2) final-step logit numeric distance
    var maxd = Float32(0.0)
    for i in range(len(ra.last)):
        var d = abs(ra.last[i] - rb.last[i])
        if d > maxd:
            maxd = d

    print("  steps compared:", STEPS)
    print("  max |Δlogit| (final step):", maxd)
    if mismatch >= 0:
        print("  ✗ TOKEN MISMATCH at decode step", mismatch)
        print("     per-op:", ra.toks[mismatch], " mega:", rb.toks[mismatch])
        raise Error("mega parity FAILED: token divergence")
    if maxd > TOL:
        print("  ✗ logits drift beyond tolerance", TOL)
        raise Error("mega parity FAILED: logit tolerance")
    print("  ✓ PASS — megakernel path matches the per-op path")
