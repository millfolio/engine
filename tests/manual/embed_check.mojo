"""Manual GPU gate: Qwen3-Embedding-0.6B numerical conformance.

Runs the from-scratch Mojo embedder (`sess_embed`) for the exact token ids in
tests/fixtures/embed/ids.txt and compares each unit vector against the torch
reference in tests/fixtures/embed/expected.bin (captured by
`pixi run -e oracle embed-capture`, the official last-token-pooled + L2-normalized
Qwen3-Embedding recipe). Passes iff cosine similarity >= 0.999 for EVERY sample
(reports max-abs-diff too). Needs the Qwen3-Embedding-0.6B checkpoint + Metal GPU.

Checkpoint resolution (first that is set):
    $QWEN_SAFETENSORS                      (a .safetensors file or snapshot dir)
    the Qwen3-Embedding-0.6B HF cache snapshot (fallback, this machine)

    pixi run embed-check
"""

from std.math import sqrt
from std.os import getenv
from std.os.path import exists
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from model import load_weights, sess_embed
from testio import read_text, read_bytes_file, ints_from

comptime COS_MIN = Float32(0.999)
# Fallback checkpoint on the serving Mac Mini (the HF cache snapshot).
comptime CACHE_CKPT = (
    "/Users/mseritan/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors"
)


def f32_at(raw: List[UInt8], byte_off: Int) -> Float32:
    """Read a little-endian f32 from a raw byte buffer at `byte_off`."""
    var p = raw.unsafe_ptr().bitcast[Float32]()
    return p[byte_off // 4]


def i32_at(raw: List[UInt8], byte_off: Int) -> Int:
    var p = raw.unsafe_ptr().bitcast[Int32]()
    return Int(p[byte_off // 4])


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only gate")

    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        ckpt = String(CACHE_CKPT)
    if not exists(ckpt):
        raise Error(
            "checkpoint not found: "
            + ckpt
            + "\nset $QWEN_SAFETENSORS to the Qwen3-Embedding-0.6B"
            + " .safetensors file or snapshot dir"
        )

    var dir = String("tests/fixtures/embed/")
    var id_lines = read_text(dir + "ids.txt").split("\n")
    var raw = read_bytes_file(dir + "expected.bin")

    print("embed-check — loading Qwen3-Embedding-0.6B weights…")
    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    if w.arch != 2:
        raise Error(
            "loaded checkpoint is not Qwen3-Embedding (arch="
            + String(w.arch)
            + "); expected arch=2"
        )
    print(
        "  arch=",
        w.arch,
        " hidden=",
        w.hidden,
        " q_dim=",
        w.q_dim,
        " heads=",
        w.hq,
        "/",
        w.hkv,
        " head_dim=",
        w.head_dim,
        " layers=",
        w.nlayers,
        sep="",
    )

    var worst_cos = Float32(1.0)
    var worst_diff = Float32(0.0)
    var n_samples = 0
    var all_ok = True
    var off = 0  # byte cursor into expected.bin

    for li in range(len(id_lines)):
        var line = String(String(id_lines[li]).strip())
        if line.byte_length() == 0:
            continue
        var ids = ints_from(line)

        # expected.bin record: [D:i32][D float32]
        var D = i32_at(raw, off)
        off += 4
        if D != w.hidden:
            raise Error(
                "fixture dim "
                + String(D)
                + " != model hidden "
                + String(w.hidden)
            )

        var got = sess_embed(ctx, w, ids)
        if len(got) != D:
            raise Error(
                "sess_embed returned "
                + String(len(got))
                + " floats, expected "
                + String(D)
            )

        # cosine similarity + max-abs-diff vs the reference unit vector. Both are
        # already L2-normalized, so cos = dot; we still divide by norms for safety.
        var dot = Float32(0.0)
        var ng = Float32(0.0)
        var ne = Float32(0.0)
        var maxd = Float32(0.0)
        for i in range(D):
            var ev = f32_at(raw, off + i * 4)
            var gv = got[i]
            dot += gv * ev
            ng += gv * gv
            ne += ev * ev
            var d = abs(gv - ev)
            if d > maxd:
                maxd = d
        off += D * 4
        var cos = dot / (sqrt(ng) * sqrt(ne))

        var ok = cos >= COS_MIN
        if not ok:
            all_ok = False
        if cos < worst_cos:
            worst_cos = cos
        if maxd > worst_diff:
            worst_diff = maxd
        n_samples += 1
        print(
            "  sample ",
            li,
            ": cos=",
            cos,
            " max_abs_diff=",
            maxd,
            "  " + ("OK" if ok else "FAIL"),
            sep="",
        )

    if n_samples == 0:
        raise Error("no samples read from " + dir + "ids.txt")

    print(
        "\nworst cos=",
        worst_cos,
        "  worst max_abs_diff=",
        worst_diff,
        "  (",
        n_samples,
        " samples)",
        sep="",
    )
    if not all_ok:
        raise Error(
            "embed-check FAILED — cosine below "
            + String(COS_MIN)
            + " for at least one sample"
        )
    print(
        "embed-check: PASS — all ",
        n_samples,
        " samples cos >= ",
        COS_MIN,
        sep="",
    )
