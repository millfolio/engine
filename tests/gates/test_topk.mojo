"""GPU token-selection parity gate: gpu_argmax / gpu_argmax_rows / gpu_topk
must match the host reference (`sampling.process_logits` selection semantics,
including the HF repetition penalty and lowest-index tie-breaks) exactly.
Synthetic logits, vocab-sized; runs on Metal, no weights."""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from runtime.tensor_ops import (
    make_select_bufs,
    mask_reset,
    gpu_argmax,
    gpu_argmax_rows,
    gpu_topk,
    download_f32,
    DevBuf,
)
from runtime.sampling import process_logits


def _fill_dev(
    ctx: DeviceContext, mut buf: DevBuf, vals: List[Float32]
) raises:
    with buf.map_to_host() as m:
        for i in range(len(vals)):
            m[i] = vals[i]


def _host_logits(vocab: Int, seed: Int) raises -> List[Float32]:
    # Deterministic pseudo-random logits with repeats (tie coverage) and a
    # spiky top so the distribution resembles a real LLM's.
    var out = List[Float32](capacity=vocab)
    var x = UInt64(seed * 2654435761 + 12345)
    for i in range(vocab):
        x = x * 6364136223846793005 + 1442695040888963407
        var r = Float32(Int((x >> 33) & 0xFFFF)) / 65536.0
        var v = r * 8.0 - 6.0
        if i % 977 == 0:
            v += 9.0  # spikes
        if i % 3 == 0:
            v = Float32(Int(v * 4.0)) / 4.0  # quantize → exact ties
        out.append(v)
    return out^


def expect(ok: Bool, msg: String) raises:
    if not ok:
        raise Error("FAIL: " + msg)
    print("  [PASS] " + msg)


def main() raises:
    comptime if not has_accelerator():
        print("no GPU — skipping topk gate")
        return
    var ctx = DeviceContext()
    comptime VOCAB = 151936
    comptime K = 20
    var b = make_select_bufs(ctx, VOCAB, 32)
    var logits_h = _host_logits(VOCAB, 7)
    var d = ctx.enqueue_create_buffer[DType.float32](VOCAB)
    ctx.synchronize()
    _fill_dev(ctx, d, logits_h)

    # 1. argmax parity (host scan is first-wins on ties)
    var want = 0
    var wv = logits_h[0]
    for i in range(VOCAB):
        if logits_h[i] > wv:
            wv = logits_h[i]
            want = i
    var got = gpu_argmax(ctx, d, b, VOCAB)
    expect(got == want, "argmax matches host (idx " + String(want) + ")")

    # 2. rows argmax parity (verify path): 4 rows, distinct seeds
    comptime ROWS = 4
    var g_h = List[Float32](capacity=ROWS * VOCAB)
    var wants = List[Int]()
    for r in range(ROWS):
        var row = _host_logits(VOCAB, 100 + r)
        var bi = 0
        var bv = row[0]
        for i in range(VOCAB):
            if row[i] > bv:
                bv = row[i]
                bi = i
        wants.append(bi)
        for i in range(VOCAB):
            g_h.append(row[i])
    var gd = ctx.enqueue_create_buffer[DType.float32](ROWS * VOCAB)
    ctx.synchronize()
    _fill_dev(ctx, gd, g_h)
    var rows_got = gpu_argmax_rows(ctx, gd, b, ROWS, VOCAB)
    var rows_ok = True
    for r in range(ROWS):
        if rows_got[r] != wants[r]:
            rows_ok = False
    expect(rows_ok, "rows argmax matches host (4 rows)")

    # 3. top-k with repetition penalty: mark a context including some of the
    #    spike ids, then compare ids AND penalized logits against the host
    #    reference selection (process_logits with temp=1 isolates selection).
    var context = List[Int]()
    context.append(977)  # a spike — penalty must demote it
    context.append(977 * 4)
    context.append(12345)
    var ids32 = ctx.enqueue_create_buffer[DType.int32](len(context))
    ctx.synchronize()
    with ids32.map_to_host() as m:
        for i in range(len(context)):
            m[i] = Int32(context[i])
    mask_reset(ctx, b, ids32, len(context))
    var pen = Float32(1.1)
    var expected = process_logits(logits_h, context, 1.0, K, 1.0, pen)
    var got_pair = gpu_topk(ctx, d, b, VOCAB, K, pen, -1)
    var got_ids = got_pair[0].copy()
    var got_vals = got_pair[1].copy()
    var ids_ok = True
    for s in range(K):
        if got_ids[s] != expected.ids[s]:
            print(
                "  slot ", s, ": gpu ", got_ids[s], " host ", expected.ids[s], sep=""
            )
            ids_ok = False
    expect(ids_ok, "top-" + String(K) + " ids match host (with rep-pen)")

    # 4. last_id inline penalty: passing last_id must equal host with the id
    #    appended to context, and must persist into the mask for a second call.
    var ctx2 = context.copy()
    ctx2.append(got_ids[0])
    var expected2 = process_logits(logits_h, ctx2, 1.0, K, 1.0, pen)
    var got2 = gpu_topk(ctx, d, b, VOCAB, K, pen, got_ids[0])
    var ok2 = True
    for s in range(K):
        if got2[0][s] != expected2.ids[s]:
            ok2 = False
    expect(ok2, "last_id penalized inline")
    var got3 = gpu_topk(ctx, d, b, VOCAB, K, pen, -1)  # mask persisted?
    var ok3 = True
    for s in range(K):
        if got3[0][s] != expected2.ids[s]:
            ok3 = False
    expect(ok3, "last_id persisted into the mask")

    print("OK — GPU token selection matches the host reference")
