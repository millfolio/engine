"""Gate: greedy decode token-for-token vs HF (ARCHITECTURE.md §6 Phase 4, §11 #8).

Calls the library `generate` and compares the produced ids to HF greedy generation
(`pixi run generate-capture`). `pixi run test-generate`.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from model import load_weights, generate
from testio import read_text, read_i32, ints_from


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var dir = "tests/fixtures/generate/"
    var lines = read_text(dir + "expected.txt").split("\n")
    var ckpt = String(String(lines[1]).strip())
    var expected = ints_from(String(lines[2]))
    var max_new = len(expected)

    var prompt = List[Int]()
    var ids32 = read_i32(dir + "prompt_ids.bin")
    for i in range(len(ids32)):
        prompt.append(Int(ids32[i]))
    print(
        "generate gate — P=",
        len(prompt),
        " max_new=",
        max_new,
        "; loading weights…",
        sep="",
    )

    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    var gen = generate(ctx, w, prompt, max_new)

    var ok = len(gen) == len(expected)
    var n = len(gen) if len(gen) < len(expected) else len(expected)
    for i in range(n):
        if gen[i] != expected[i]:
            ok = False
    var gs = String("")
    var es = String("")
    for i in range(len(gen)):
        gs += String(gen[i]) + " "
    for i in range(len(expected)):
        es += String(expected[i]) + " "
    print("  gpu gen: ", gs, sep="")
    print("  hf  ref: ", es, sep="")

    if not ok:
        raise Error(
            "greedy decode does NOT match HF token-for-token — gate FAILED"
        )
    print(
        "OK — Mojo greedy decode matches HF token-for-token (",
        len(gen),
        " tokens)",
        sep="",
    )
