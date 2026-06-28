"""Small host-side helpers shared by the verification harnesses (test_*.mojo).

File readers for the fixtures `pixi run *-capture` produces, plus device-buffer
comparison helpers. Not part of the inference library — only the gates use these.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

comptime DevBuf = DeviceBuffer[DType.float32]


def upload_f32(ctx: DeviceContext, host: List[Float32]) raises -> DevBuf:
    var n = len(host)
    var dev = ctx.enqueue_create_buffer[DType.float32](n)
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](host[i])
    return dev^


def upload_bf16(
    ctx: DeviceContext, host: List[Float32]
) raises -> DeviceBuffer[DType.uint16]:
    """Truncate f32 → bf16 (round-to-nearest-even on the dropped 16 bits) and
    upload as raw u16 — matches how the model's bf16 weights live on device, so
    the matmul gate exercises the real (bf16-weight) kernel path (§11 #12)."""
    var n = len(host)
    var dev = ctx.enqueue_create_buffer[DType.uint16](n)
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            var f = host[i]
            var bits = UnsafePointer(to=f).bitcast[UInt32]()[0]
            # round-to-nearest-even: add 0x7FFF + lsb of the kept mantissa
            var rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            mt[i] = rebind[mt.ElementType](UInt16(rounded >> 16))
    return dev^


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def read_f32(path: String) raises -> List[Float32]:
    var out = List[Float32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var p = raw.unsafe_ptr().bitcast[Float32]()
        for i in range(len(raw) // 4):
            out.append(p[i])
    return out^


def read_i32(path: String) raises -> List[Int32]:
    var out = List[Int32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var p = raw.unsafe_ptr().bitcast[Int32]()
        for i in range(len(raw) // 4):
            out.append(p[i])
    return out^


def read_bytes_file(path: String) raises -> List[UInt8]:
    var out = List[UInt8]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        for i in range(len(raw)):
            out.append(raw[i])
    return out^


def ints_from(s: String) raises -> List[Int]:
    """Parse whitespace-separated integers from a string."""
    var out = List[Int]()
    for t in s.split(" "):
        var ts = String(t).strip()
        if ts.byte_length() > 0:
            out.append(Int(atol(ts)))
    return out^


def max_abs(mut dev: DevBuf, expected: List[Float32]) raises -> Float32:
    var n = len(expected)
    var worst = Float32(0.0)
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            var d = abs(rebind[Scalar[DType.float32]](mt[i]) - expected[i])
            if d > worst:
                worst = d
    return worst


def argmax_row(mut dev: DevBuf, row: Int, width: Int) raises -> Int:
    var base = row * width
    var best = -1
    var best_v = Float32(-1.0e30)
    with dev.map_to_host() as m:
        var mt = TileTensor(m, row_major((row + 1) * width))
        for i in range(width):
            var v = rebind[Scalar[DType.float32]](mt[base + i])
            if v > best_v:
                best_v = v
                best = i
    return best


def argmax_list(a: List[Float32]) -> Int:
    var best = -1
    var best_v = Float32(-1.0e30)
    for i in range(len(a)):
        if a[i] > best_v:
            best_v = a[i]
            best = i
    return best
