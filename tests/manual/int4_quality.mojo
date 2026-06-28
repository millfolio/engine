"""Phase-0 quality gate for LOW-BIT RTN weight quantization on 0.5B.

Generalizes int8_quality to arbitrary bit width + group size, and sweeps several
schemes against one bf16 reference so we can pick what holds quality before
building any int4 kernels. Fake-quantization (quantize/dequantize per group,
store back as bf16, reuse the bf16 forward) reproduces dequant numerics with no
kernel work — the same trick validated for int8.

Schemes swept: int8 per-channel (sanity), int4 per-channel, int4 g128, int4 g64.
For each: per-matrix recon error, teacher-forced top-1 + KL drift over the bf16
reference continuation, and free-running greedy divergence + decoded text.

    pixi run int4-quality
"""

from std.os import getenv
from std.math import sqrt, log, exp
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from model import (
    Weights,
    WBuf,
    load_weights,
    new_session,
    sess_prefill,
    sess_step,
    argmax_f,
    EOS1,
    EOS2,
)
from kernels import bf16_widen
from tokenizer import load_tokenizer, Tokenizer
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


def f32_to_bf16(f: Float32) -> Scalar[DType.uint16]:
    var x = f
    var bits = UnsafePointer(to=x).bitcast[UInt32]()[0]
    var lsb = (bits >> 16) & 1
    var r = bits + 0x7FFF + lsb
    return Scalar[DType.uint16]((r >> 16) & 0xFFFF)


def fakequant_rows(
    ctx: DeviceContext, mut w: WBuf, N: Int, K: Int, bits: Int, group: Int
) raises -> Float64:
    """In-place symmetric RTN round-trip with `bits` precision and `group`-sized
    blocks within each row (group<=0 or >=K means per-channel). Returns mean
    relative L2 reconstruction error across rows."""
    var qmax = Float32((1 << (bits - 1)) - 1)  # int8->127, int4->7
    var g = K if (group <= 0 or group > K) else group
    var rel_sum = 0.0
    var rows = 0
    with w.map_to_host() as m:
        var t = TileTensor(m, row_major(N * K))
        for n in range(N):
            var base = n * K
            var num = Float64(0.0)
            var den = Float64(0.0)
            var gstart = 0
            while gstart < K:
                var gend = gstart + g
                if gend > K:
                    gend = K
                var amax = Float32(0.0)
                for k in range(gstart, gend):
                    var v = bf16_widen(
                        rebind[Scalar[DType.uint16]](t[base + k])
                    )
                    var a = v if v >= 0.0 else -v
                    if a > amax:
                        amax = a
                if amax > 0.0:
                    var scale = amax / qmax
                    var inv = 1.0 / scale
                    for k in range(gstart, gend):
                        var v = bf16_widen(
                            rebind[Scalar[DType.uint16]](t[base + k])
                        )
                        var q = v * inv
                        var half = Float32(0.5) if q >= 0.0 else Float32(-0.5)
                        var qr = Float32(Int(q + half))
                        if qr > qmax:
                            qr = qmax
                        elif qr < -qmax:
                            qr = -qmax
                        var dq = qr * scale
                        var d = Float64(dq - v)
                        num += d * d
                        den += Float64(v) * Float64(v)
                        t[base + k] = rebind[t.ElementType](f32_to_bf16(dq))
                gstart = gend
            if den > 0.0:
                rel_sum += sqrt(num / den)
                rows += 1
    return rel_sum / Float64(rows) if rows > 0 else 0.0


def kl_and_top1(
    p_logits: List[Float32], q_logits: List[Float32]
) raises -> Tuple[Float64, Int, Int]:
    var n = len(p_logits)
    var pmax = Float32(-1.0e30)
    var qmax = Float32(-1.0e30)
    var pi = 0
    var qi = 0
    for i in range(n):
        if p_logits[i] > pmax:
            pmax = p_logits[i]
            pi = i
        if q_logits[i] > qmax:
            qmax = q_logits[i]
            qi = i
    var pz = Float64(0.0)
    var qz = Float64(0.0)
    for i in range(n):
        pz += exp(Float64(p_logits[i] - pmax))
        qz += exp(Float64(q_logits[i] - qmax))
    var logpz = log(pz)
    var logqz = log(qz)
    var kl = Float64(0.0)
    for i in range(n):
        var lp = Float64(p_logits[i] - pmax) - logpz
        var p = exp(lp)
        if p <= 1.0e-12:
            continue
        var lq = Float64(q_logits[i] - qmax) - logqz
        kl += p * (lp - lq)
    return (kl, pi, qi)


def eval_scheme(
    ctx: DeviceContext,
    ckpt: String,
    label: String,
    bits: Int,
    group: Int,
    ids: List[Int],
    ref_tokens: List[Int],
    ref_logits: List[List[Float32]],
    mut tok: Tokenizer,
) raises:
    # load bf16 (q4=False) — projection weights are QMat-wrapped bf16; .bf16 gets
    # the underlying buffer this fake-quant gate mutates in place.
    var w = load_weights(ctx, ckpt)
    var qkvN = w.hidden + 2 * w.nkv
    var e_q = fakequant_rows(ctx, w.qkv[0].bf16, qkvN, w.hidden, bits, group)
    var e_down = fakequant_rows(
        ctx, w.down[0].bf16, w.hidden, w.inter, bits, group
    )
    _ = fakequant_rows(ctx, w.embed, w.vocab, w.hidden, bits, group)
    for l in range(w.nlayers):
        if l != 0:
            _ = fakequant_rows(ctx, w.qkv[l].bf16, qkvN, w.hidden, bits, group)
            _ = fakequant_rows(
                ctx, w.down[l].bf16, w.hidden, w.inter, bits, group
            )
        _ = fakequant_rows(ctx, w.ow[l].bf16, w.hidden, w.hidden, bits, group)
        _ = fakequant_rows(
            ctx, w.gate_up[l].bf16, 2 * w.inter, w.hidden, bits, group
        )

    var nsteps = len(ref_tokens)
    var s2 = new_session(ctx, len(ids) + MAX_NEW + 2, w.nlayers, w.nkv)
    var lg2 = sess_prefill(ctx, w, s2, ids)
    var top1 = 0
    var kl_sum = 0.0
    for t in range(nsteps):
        var r = kl_and_top1(ref_logits[t], lg2)
        kl_sum += r[0]
        if r[1] == r[2]:
            top1 += 1
        if t + 1 < nsteps:
            lg2 = sess_step(ctx, w, s2, ref_tokens[t])

    var s3 = new_session(ctx, len(ids) + MAX_NEW + 2, w.nlayers, w.nkv)
    var qtok = List[Int]()
    var lg3 = sess_prefill(ctx, w, s3, ids)
    var qn = argmax_f(lg3)
    qtok.append(qn)
    while len(qtok) < nsteps and qn != EOS1 and qn != EOS2:
        lg3 = sess_step(ctx, w, s3, qn)
        qn = argmax_f(lg3)
        qtok.append(qn)
    var diverge = -1
    var lim = nsteps if nsteps < len(qtok) else len(qtok)
    for t in range(lim):
        if ref_tokens[t] != qtok[t]:
            diverge = t
            break

    var body = List[Int]()
    for i in range(len(qtok)):
        if qtok[i] == EOS1 or qtok[i] == EOS2:
            break
        body.append(qtok[i])

    var gtxt = group if group > 0 else 0
    print("── ", label, " (bits=", bits, " group=", gtxt, ") ──", sep="")
    print("   recon rel-L2: q_proj=", e_q, "  down_proj=", e_down)
    print(
        "   top-1: ",
        top1,
        "/",
        nsteps,
        " (",
        Float64(top1) / Float64(nsteps) * 100.0,
        "%)   mean KL=",
        kl_sum / Float64(nsteps),
        " nats",
    )
    if diverge < 0:
        print("   greedy IDENTICAL through ", lim, " tokens")
    else:
        print("   greedy diverges at index ", diverge)
    print("   text: ", bytes_to_string(tok.decode(body)), sep="")


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

    print("loading weights (bf16 reference)…")
    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)

    print("bf16: generating reference continuation…")
    var s = new_session(ctx, len(ids) + MAX_NEW + 2, w.nlayers, w.nkv)
    var ref_tokens = List[Int]()
    var ref_logits = List[List[Float32]]()
    var lg = sess_prefill(ctx, w, s, ids)
    var nxt = argmax_f(lg)
    ref_logits.append(lg.copy())
    ref_tokens.append(nxt)
    while len(ref_tokens) < MAX_NEW and nxt != EOS1 and nxt != EOS2:
        lg = sess_step(ctx, w, s, nxt)
        nxt = argmax_f(lg)
        ref_logits.append(lg.copy())
        ref_tokens.append(nxt)
    print("  bf16 produced", len(ref_tokens), "tokens\n")

    var ref_body = List[Int]()
    for i in range(len(ref_tokens)):
        if ref_tokens[i] == EOS1 or ref_tokens[i] == EOS2:
            break
        ref_body.append(ref_tokens[i])
    print(">>> ", user, sep="")
    print("bf16: ", bytes_to_string(tok.decode(ref_body)), "\n", sep="")

    eval_scheme(
        ctx, ckpt, "int8 per-channel", 8, 0, ids, ref_tokens, ref_logits, tok
    )
    eval_scheme(
        ctx, ckpt, "int4 per-channel", 4, 0, ids, ref_tokens, ref_logits, tok
    )
    eval_scheme(
        ctx, ckpt, "int4 group-128", 4, 128, ids, ref_tokens, ref_logits, tok
    )
    eval_scheme(
        ctx, ckpt, "int4 group-64", 4, 64, ids, ref_tokens, ref_logits, tok
    )
