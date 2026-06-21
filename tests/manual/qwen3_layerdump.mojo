"""Per-layer hidden magnitude trace through the Qwen3-8B (arch 3) bf16 forward, to
localize where the representation collapses (uniform logits => zero hidden)."""

from std.sys import argv
from std.gpu.host import DeviceContext
from std.math import sqrt
from layout import TileTensor, row_major
from models.qwen import load_weights
from runtime.engine import new_session, upload_ids
from runtime.tensor_ops import probe_simd_gemm, DevBuf

comptime SNAP_8B = "/Users/mseritan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime SNAP_14B = "/Users/mseritan/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"


def dump(ctx: DeviceContext, mut h: DevBuf, T: Int, hd: Int, label: String) raises:
    ctx.synchronize()
    var ss = Float64(0.0)
    var amax = Float64(0.0)
    with h.map_to_host() as m:
        var t = TileTensor(m, row_major(T * hd))
        var base = (T - 1) * hd
        for i in range(hd):
            var v = Float64(rebind[Scalar[DType.float32]](t[base + i]))
            ss += v * v
            if v < 0:
                v = -v
            if v > amax:
                amax = v
    print(label, " rms=", sqrt(ss / Float64(hd)), " absmax=", amax)


def main() raises:
    var a = argv()
    var is14 = len(a) > 1 and String(a[1]) == "14b"
    var q4 = len(a) > 2 and String(a[2]) == "int4"
    var snap = String(SNAP_14B) if is14 else String(SNAP_8B)
    var ctx = DeviceContext()
    print("loading", "14B" if is14 else "8B", "int4" if q4 else "bf16", "…")
    var w = load_weights(ctx, snap, q4)
    w.simd_ok = probe_simd_gemm(ctx)
    var hd = w.hidden
    print("arch=", w.arch, " hidden=", hd, " hq=", w.hq, " hkv=", w.hkv,
          " head_dim=", w.head_dim, " q_dim=", w.q_dim, " nlayers=", w.nlayers)

    var ids: List[Int] = [785, 6722, 315, 9625, 374, 12095, 11, 323]
    var T = len(ids)
    var s = new_session(ctx, 64, w.nlayers, w.nkv)
    var ids_dev = upload_ids(ctx, ids)
    var h = w.embed_prompt(ctx, ids_dev, T)
    dump(ctx, h, T, hd, "embed   ")
    for l in range(w.nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)
    dump(ctx, h, T, hd, "after L35")

    # M=1 path (last-position lm_logits):
    var logits = w.lm_logits(ctx, h, T, s.dummy)
    var mn = logits[0]
    var mx = logits[0]
    for i in range(len(logits)):
        if logits[i] < mn:
            mn = logits[i]
        if logits[i] > mx:
            mx = logits[i]
    print("lm_logits (M=1): spread=", mx - mn)

    # M=n path (token_logprobs — what the server's /v1/completions uses):
    var targets = List[Int]()
    for i in range(1, T):
        targets.append(ids[i])
    var lp = w.token_logprobs(ctx, h, T - 1, targets, s.dummy)
    var sm = Float64(0.0)
    for i in range(len(lp)):
        sm += Float64(lp[i])
    print("token_logprobs (M=n): mean=", sm / Float64(len(lp)), " (-ln vocab = -11.93 => uniform)")
