"""Quality eval for per-output-channel int8 RTN weight quantization on 0.5B.

Strategy: *fake-quantization*. We reuse the entire existing bf16 forward pass
unchanged. For each weight matrix W[N,K] we quantize each output channel (row n)
to symmetric int8 with scale = max_k|W[n,k]|/127, then dequantize back and store
the result as bf16. The bf16 GEMV then sees exactly the values an int8 dequant
kernel would produce (scale*q, which a 7-bit q renders ~losslessly in bf16's
8-bit mantissa), so this reproduces int8-RTN numerics with zero kernel work.

Reports:
  - per-matrix-class reconstruction error (mean relative L2 over rows)
  - teacher-forced distribution drift: top-1 agreement + mean KL(bf16||int8)
    over a greedy reference continuation
  - free-running greedy divergence index + decoded text, bf16 vs int8

Needs weights + Metal GPU.

    pixi run int8-quality
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


def f32_to_bf16(f: Float32) -> Scalar[DType.uint16]:
    """Round-to-nearest-even narrow f32 -> bf16 (top 16 bits)."""
    var x = f
    var bits = UnsafePointer(to=x).bitcast[UInt32]()[0]
    var lsb = (bits >> 16) & 1
    var r = bits + 0x7FFF + lsb
    return Scalar[DType.uint16]((r >> 16) & 0xFFFF)


def fakequant_rows(
    ctx: DeviceContext, mut w: WBuf, N: Int, K: Int
) raises -> Float64:
    """In-place per-row (output-channel) symmetric int8 RTN round-trip. Returns
    mean relative L2 reconstruction error across rows."""
    var rel_sum = 0.0
    var rows = 0
    with w.map_to_host() as m:
        var t = TileTensor(m, row_major(N * K))
        for n in range(N):
            var base = n * K
            var amax = Float32(0.0)
            for k in range(K):
                var v = bf16_widen(rebind[Scalar[DType.uint16]](t[base + k]))
                var a = v if v >= 0.0 else -v
                if a > amax:
                    amax = a
            if amax == 0.0:
                continue
            var scale = amax / 127.0
            var inv = 1.0 / scale
            var num = Float64(0.0)
            var den = Float64(0.0)
            for k in range(K):
                var v = bf16_widen(rebind[Scalar[DType.uint16]](t[base + k]))
                var q = v * inv
                # round-to-nearest, clamp to int8 symmetric range
                var half = Float32(0.5) if q >= 0.0 else Float32(-0.5)
                var qr = Float32(Int(q + half))
                if qr > 127.0:
                    qr = 127.0
                elif qr < -127.0:
                    qr = -127.0
                var dq = qr * scale
                var d = Float64(dq - v)
                num += d * d
                den += Float64(v) * Float64(v)
                t[base + k] = rebind[t.ElementType](f32_to_bf16(dq))
            if den > 0.0:
                rel_sum += sqrt(num / den)
                rows += 1
    return rel_sum / Float64(rows) if rows > 0 else 0.0


def kl_and_top1(
    p_logits: List[Float32], q_logits: List[Float32]
) raises -> Tuple[Float64, Int, Int]:
    """KL(softmax(p) || softmax(q)) in nats, plus argmax of each."""
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
        var lp = Float64(p_logits[i] - pmax) - logpz  # log p_i
        var p = exp(lp)
        if p <= 1.0e-12:
            continue
        var lq = Float64(q_logits[i] - qmax) - logqz  # log q_i
        kl += p * (lp - lq)
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

    print("loading weights (bf16 reference)…")
    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)

    # ---- bf16 reference: greedy generate, record tokens + per-step logits ----
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
    var nsteps = len(ref_tokens)
    print("  bf16 produced", nsteps, "tokens")

    # ---- fake-quantize all bf16 weight matrices in place ----
    print("quantizing weights (per-channel int8 RTN)…")
    var e_embed = fakequant_rows(ctx, w.embed, w.vocab, w.hidden)
    var e_qw = 0.0
    var e_kw = 0.0
    var e_vw = 0.0
    var e_ow = 0.0
    var e_gate = 0.0
    var e_up = 0.0
    var e_down = 0.0
    for l in range(w.nlayers):
        e_qw += fakequant_rows(ctx, w.qw[l], w.hidden, w.hidden)
        e_kw += fakequant_rows(ctx, w.kw[l], w.nkv, w.hidden)
        e_vw += fakequant_rows(ctx, w.vw[l], w.nkv, w.hidden)
        e_ow += fakequant_rows(ctx, w.ow[l], w.hidden, w.hidden)
        e_gate += fakequant_rows(ctx, w.gate[l], w.inter, w.hidden)
        e_up += fakequant_rows(ctx, w.up[l], w.inter, w.hidden)
        e_down += fakequant_rows(ctx, w.down[l], w.hidden, w.inter)
    var nl = Float64(w.nlayers)
    print("  rel-L2 recon error (mean over rows):")
    print("    embed/lmhead:", e_embed)
    print(
        "    q_proj:",
        e_qw / nl,
        "  k_proj:",
        e_kw / nl,
        "  v_proj:",
        e_vw / nl,
        "  o_proj:",
        e_ow / nl,
    )
    print("    gate:", e_gate / nl, "  up:", e_up / nl, "  down:", e_down / nl)

    # ---- teacher-forced drift: feed bf16 tokens, compare distributions ----
    print("int8: teacher-forced distribution drift…")
    var s2 = new_session(ctx, len(ids) + MAX_NEW + 2, w.nlayers, w.nkv)
    var lg2 = sess_prefill(ctx, w, s2, ids)
    var top1_match = 0
    var kl_sum = 0.0
    for t in range(nsteps):
        var r = kl_and_top1(ref_logits[t], lg2)
        kl_sum += r[0]
        if r[1] == r[2]:
            top1_match += 1
        if t + 1 < nsteps:
            lg2 = sess_step(ctx, w, s2, ref_tokens[t])
    print(
        "  top-1 agreement:",
        top1_match,
        "/",
        nsteps,
        "(",
        Float64(top1_match) / Float64(nsteps) * 100.0,
        "%)",
    )
    print("  mean KL(bf16||int8):", kl_sum / Float64(nsteps), "nats")

    # ---- free-running int8 greedy: divergence index + decoded text ----
    print("int8: free-running greedy…")
    var s3 = new_session(ctx, len(ids) + MAX_NEW + 2, w.nlayers, w.nkv)
    var q_tokens = List[Int]()
    var lg3 = sess_prefill(ctx, w, s3, ids)
    var qn = argmax_f(lg3)
    q_tokens.append(qn)
    while len(q_tokens) < nsteps and qn != EOS1 and qn != EOS2:
        lg3 = sess_step(ctx, w, s3, qn)
        qn = argmax_f(lg3)
        q_tokens.append(qn)
    var diverge = -1
    var lim = nsteps if nsteps < len(q_tokens) else len(q_tokens)
    for t in range(lim):
        if ref_tokens[t] != q_tokens[t]:
            diverge = t
            break
    if diverge < 0:
        print("  greedy sequences IDENTICAL through", lim, "tokens")
    else:
        print("  greedy diverges at token index", diverge)

    var ref_body = List[Int]()
    for i in range(len(ref_tokens)):
        if ref_tokens[i] == EOS1 or ref_tokens[i] == EOS2:
            break
        ref_body.append(ref_tokens[i])
    var q_body = List[Int]()
    for i in range(len(q_tokens)):
        if q_tokens[i] == EOS1 or q_tokens[i] == EOS2:
            break
        q_body.append(q_tokens[i])
    print("\n  >>> ", user, sep="")
    print("  bf16: ", bytes_to_string(tok.decode(ref_body)), sep="")
    print("  int8: ", bytes_to_string(tok.decode(q_body)), sep="")
