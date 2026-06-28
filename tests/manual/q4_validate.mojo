"""Authoritative correctness+quality check for the wired int4 path: compare it
against bf16 with the SAME top-1/KL methodology the quality gate used, and verify
it lands at the gate's g128 numbers (0.5B ≈ 73% top-1, KL ≈ 0.33; 3B ≈ 85%, KL
≈ 0.16). Matching the gate means the wired quantizer+kernels reproduce a correct
group-128 quantization (a wiring bug would show as far worse top-1 / garbage).

    pixi run q4-validate
"""

from std.os import getenv
from std.math import log, exp
from std.gpu.host import DeviceContext

from model import (
    load_weights,
    new_session,
    sess_prefill,
    sess_step,
    argmax_f,
    EOS1,
    EOS2,
)
from tokenizer import load_tokenizer
from chat import load_chat_template, render_chat
from json import bytes_to_string

comptime MAX_NEW = 96
comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var sb = s.as_bytes()
    for i in range(len(sb)):
        out.append(sb[i])
    return out^


def kl_top1(
    p: List[Float32], q: List[Float32]
) raises -> Tuple[Float64, Int, Int]:
    var n = len(p)
    var pm = Float32(-1.0e30)
    var qm = Float32(-1.0e30)
    var pi = 0
    var qi = 0
    for i in range(n):
        if p[i] > pm:
            pm = p[i]
            pi = i
        if q[i] > qm:
            qm = q[i]
            qi = i
    var pz = Float64(0.0)
    var qz = Float64(0.0)
    for i in range(n):
        pz += exp(Float64(p[i] - pm))
        qz += exp(Float64(q[i] - qm))
    var lpz = log(pz)
    var lqz = log(qz)
    var kl = Float64(0.0)
    for i in range(n):
        var lp = Float64(p[i] - pm) - lpz
        var pp = exp(lp)
        if pp > 1.0e-12:
            kl += pp * (lp - (Float64(q[i] - qm) - lqz))
    return (kl, pi, qi)


def main() raises:
    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        ckpt = String(
            String(
                read_text("tests/fixtures/forward/meta.txt").split("\n")[1]
            ).strip()
        )

    var user = String(
        "Explain how a hash map works and why lookups are fast. Then write a"
        " short Python example."
    )
    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ids = tok.encode(to_bytes(render_chat(tmpl, user)))
    var ctx = DeviceContext()

    # bf16 reference: greedy tokens + per-step logits
    print("bf16 reference…")
    var wb = load_weights(ctx, ckpt, False)
    var s = new_session(ctx, len(ids) + MAX_NEW + 2, wb.nlayers, wb.nkv)
    var ref_tok = List[Int]()
    var ref_lg = List[List[Float32]]()
    var lg = sess_prefill(ctx, wb, s, ids)
    var nxt = argmax_f(lg)
    ref_lg.append(lg.copy())
    ref_tok.append(nxt)
    while len(ref_tok) < MAX_NEW and nxt != EOS1 and nxt != EOS2:
        lg = sess_step(ctx, wb, s, nxt)
        nxt = argmax_f(lg)
        ref_lg.append(lg.copy())
        ref_tok.append(nxt)
    var nsteps = len(ref_tok)

    # wired int4: teacher-force the bf16 tokens, compare distributions
    print("wired int4 (q4=True)…")
    var wq = load_weights(ctx, ckpt, True)
    var s2 = new_session(ctx, len(ids) + MAX_NEW + 2, wq.nlayers, wq.nkv)
    var lg2 = sess_prefill(ctx, wq, s2, ids)
    var top1 = 0
    var kl_sum = 0.0
    for t in range(nsteps):
        var r = kl_top1(ref_lg[t], lg2)
        kl_sum += r[0]
        if r[1] == r[2]:
            top1 += 1
        if t + 1 < nsteps:
            lg2 = sess_step(ctx, wq, s2, ref_tok[t])

    print("  arch=", wq.arch, " quant=", wq.quant, " steps=", nsteps)
    print(
        "  top-1 agreement vs bf16: ",
        top1,
        "/",
        nsteps,
        " (",
        Float64(top1) / Float64(nsteps) * 100.0,
        "%)",
    )
    print("  mean KL(bf16||int4): ", kl_sum / Float64(nsteps), " nats")
    print("  expected (gate g128): 0.5B ~73% / KL~0.33,  3B ~85% / KL~0.16")
