"""Gate: safetensors loader vs torch (ARCHITECTURE.md §6 Phase 2, §11 #5).

Uses the library header parser to locate tensors, reads the first 8 bf16 elements
of each, and checks they decode bit-exactly to torch's values (`pixi run
loader-capture`). Shapes are covered transitively by the forward gate.
`pixi run test-loader`.
"""

from model import read_header, TensorEntry
from testio import read_text, read_f32


def bf16_to_f32(lo: Int, hi: Int) -> Float32:
    var bits: UInt32 = (UInt32(hi) << 24) | (UInt32(lo) << 16)
    return UnsafePointer(to=bits).bitcast[Float32]()[0]


def find_idx(entries: List[TensorEntry], name: String) -> Int:
    for e in range(len(entries)):
        if entries[e].name == name:
            return e
    return -1


def main() raises:
    var dir = "tests/fixtures/loader/"
    var lines = read_text(dir + "meta.txt").split("\n")
    var path = String(String(lines[0]).strip())
    var count = Int(atol(String(lines[1]).strip()))

    var names = List[String]()
    for i in range(count):
        var parts = String(lines[2 + i]).split(" ")
        names.append(String(parts[0]))

    var expected = read_f32(dir + "expected.bin")
    var entries = read_header(path)

    print("loader gate — ", path, ":", sep="")
    var all_ok = True
    with open(path, "r") as f:
        for i in range(count):
            var name = names[i]
            var idx = find_idx(entries, name)
            if idx < 0:
                print("  ", name, " [FAIL — not in header]", sep="")
                all_ok = False
                continue
            var begin = entries[idx].begin
            _ = f.seek(UInt64(begin))
            var raw = f.read_bytes(16)
            var worst = Float32(0.0)
            for e in range(8):
                var got = bf16_to_f32(Int(raw[2 * e]), Int(raw[2 * e + 1]))
                var d = abs(got - expected[i * 8 + e])
                if d > worst:
                    worst = d
            var ok = worst == 0.0
            var tag = "OK" if ok else "FAIL"
            print("  ", name, " max_abs=", worst, " [", tag, "]", sep="")
            all_ok = all_ok and ok

    if not all_ok:
        raise Error("safetensors loader does NOT match torch — gate FAILED")
    print(
        "OK — Mojo safetensors parse + bf16→f32 match torch (first-8 of each"
        " tensor)"
    )
