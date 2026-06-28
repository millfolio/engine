"""Phase-0 GPU go/no-go check (ARCHITECTURE.md §6).

Runs a hand-written Mojo Metal kernel on this machine's Apple Silicon GPU and
verifies the result on the host. This re-confirms max-backend's isolation-ladder
rung 1 (its §8 #2): hand-written Mojo GPU kernels execute *and* compute correctly
on the M4 — the premise the whole GPU-only build rests on (ARCHITECTURE.md §1).

The kernel computes `c = a * b + a` elementwise (exercises both GPU multiply and
add). With a=1, b=2 every element must equal 3.0; any mismatch — or no detected
accelerator — exits non-zero so `pixi run gpu-hello` is a real gate, not a demo.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime layout = row_major[N]()


def fma_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        c[tid] = a[tid] * b[tid] + a[tid]


def main() raises:
    comptime if not has_accelerator():
        raise Error(
            "no GPU accelerator detected — this is a GPU-only build (needs"
            " Metal)"
        )

    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)
    c_buf.enqueue_fill(0.0)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    ctx.enqueue_function[fma_kernel](
        a,
        b,
        c,
        N,
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()

    var errors = 0
    with c_buf.map_to_host() as host:
        var result = TileTensor(host, layout)
        comptime assert result.flat_rank == 1, "expected 1D tensor"
        for i in range(N):
            var v = rebind[Scalar[dtype]](result[i])
            if v != 3.0:
                errors += 1

    if errors != 0:
        raise Error(
            "GPU result mismatch: "
            + String(errors)
            + "/"
            + String(N)
            + " elements wrong (expected 3.0)"
        )

    print(
        "OK — Mojo Metal kernel ran on the GPU; c = a*b+a == 3.0 for all",
        N,
        "elements",
    )
