"""Manual GPU gate: greedy-parity of the simdgroup-matrix prefill GEMM.

Generates the same prompt twice with the real 0.5B weights — once forcing the
scalar tiled GEMM (simd_ok=False), once the simdgroup-matrix GEMM (simd_ok=True)
— and asserts the greedy token sequences are IDENTICAL. The two GEMMs differ
numerically by ~2e-6 (hardware FMA/order), so this confirms that drift never
flips an argmax across a full prefill + decode. Needs weights + Metal GPU.

    pixi run simd-parity            (defaults to the meta.txt 0.5B checkpoint)
"""

from std.sys import argv
from std.os import getenv
from std.gpu.host import DeviceContext

from model import load_weights, generate, probe_simd_gemm, EOS1, EOS2
from tokenizer import load_tokenizer
from chat import load_chat_template, render_chat
from json import bytes_to_string

comptime MAX_NEW = 48
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


def trim_eos(g: List[Int]) -> List[Int]:
    var body = List[Int]()
    for i in range(len(g)):
        if g[i] == EOS1 or g[i] == EOS2:
            break
        body.append(g[i])
    return body^


def main() raises:
    var user = String("What is the capital of France? Explain in one sentence.")
    if len(argv()) > 1:
        var joined = String("")
        for i in range(1, len(argv())):
            if i > 1:
                joined += " "
            joined += String(argv()[i])
        user = joined

    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        ckpt = String(String(read_text("tests/fixtures/forward/meta.txt").split("\n")[1]).strip())

    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ids = tok.encode(to_bytes(render_chat(tmpl, user)))

    print("loading weights…")
    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)

    if not probe_simd_gemm(ctx):
        raise Error("probe_simd_gemm failed — cannot test parity on this toolchain")
    print("probe_simd_gemm: OK   prompt tokens=", len(ids))

    w.simd_ok = False
    var g_scalar = generate(ctx, w, ids, MAX_NEW)
    w.simd_ok = True
    var g_simd = generate(ctx, w, ids, MAX_NEW)

    var bs = trim_eos(g_scalar)
    var bi = trim_eos(g_simd)
    print("\nscalar: ", bytes_to_string(tok.decode(bs)))
    print("simd  : ", bytes_to_string(tok.decode(bi)))

    var n = len(g_scalar)
    if len(g_simd) < n:
        n = len(g_simd)
    var diverge = -1
    for i in range(n):
        if g_scalar[i] != g_simd[i]:
            diverge = i
            break

    if len(g_scalar) == len(g_simd) and diverge == -1:
        print("\nsimd-parity gate: PASS — ", len(g_scalar),
              " greedy tokens identical (scalar == simdgroup-matrix)", sep="")
    else:
        print("\nsimd-parity gate: DIVERGED at token ", diverge,
              " (scalar len ", len(g_scalar), ", simd len ", len(g_simd), ")", sep="")
        raise Error("greedy parity broken: simd GEMM changed the token sequence")
