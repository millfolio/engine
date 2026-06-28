"""Per-layer hidden-state magnitude trace through the 12B (gemma.mojo) int4 forward.
If the full-attention layers (5,11,17,…,47) blow up or collapse the hidden state
relative to sliding layers, that localizes the 12B calibration bug to the
full-attention path (head_dim 512, 1 KV, V=k reuse). No HF needed."""

from std.gpu.host import DeviceContext
from std.math import sqrt
from models.gemma import load_gemma_weights, _is_full_layer, G_NLAYERS
from runtime.engine import new_session, upload_ids
from runtime.tensor_ops import probe_simd_gemm, DevBuf
from layout import TileTensor, row_major

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/58540658b6c08edab2ddc1fbde7f28cc9987ced3"


def dump_mag(
    ctx: DeviceContext, mut h: DevBuf, T: Int, hd: Int, label: String
) raises:
    ctx.synchronize()
    var sumsq = Float64(0.0)
    var amax = Float64(0.0)
    with h.map_to_host() as m:
        var t = TileTensor(m, row_major(T * hd))
        var base = (T - 1) * hd  # last row
        for i in range(hd):
            var v = Float64(rebind[Scalar[DType.float32]](t[base + i]))
            sumsq += v * v
            if v < 0:
                v = -v
            if v > amax:
                amax = v
    print(label, " rms=", sqrt(sumsq / Float64(hd)), " absmax=", amax)


def main() raises:
    var ctx = DeviceContext()
    print("loading 12B int4…")
    var layers = List[Int]()
    for l in range(G_NLAYERS):
        layers.append(l)
    var gw = load_gemma_weights(ctx, SNAP, layers, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    var hd = gw.hidden

    var ids: List[Int] = [
        2,
        818,
        7578,
        200258,
        568,
        1708,
        834,
        625,
        795,
        577,
        13139,
        531,
        8988,
        529,
        1515,
        236768,
        691,
        1520,
        44260,
        496,
        544,
        1488,
        785,
        4217,
        531,
        775,
        236761,
    ]
    var T = len(ids)
    var s = new_session(ctx, 64, cfg.nlayers, cfg.nkv)
    var ids_dev = upload_ids(ctx, ids)
    var h = gw.embed_prompt(ctx, ids_dev, T)
    dump_mag(ctx, h, T, hd, "embed       ")
    for l in range(cfg.nlayers):
        h = gw.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)
        var tag = "FULL" if _is_full_layer(l) else "    "
        dump_mag(ctx, h, T, hd, "after L" + String(l) + " " + tag)

    var targets = List[Int]()
    for i in range(1, T):
        targets.append(ids[i])
    var lp = gw.token_logprobs(ctx, h, T - 1, targets, s.dummy)
    var nll = Float64(0.0)
    for i in range(len(lp)):
        nll += -Float64(lp[i])
    var mean = nll / Float64(len(lp))
    print("G12 mean_nll=", mean, " PPL=", 2.718281828459045**mean)
    print("G12_LP=[null", end="")
    for i in range(len(lp)):
        print(",", lp[i], end="")
    print("]")
