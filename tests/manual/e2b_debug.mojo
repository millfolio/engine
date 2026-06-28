"""Numeric bisect of the e2b Mojo forward vs the HF gemma4 reference (.scratch/
e2b_hf_ref.py). Feeds the SAME raw ids [2,4521,1902] (bf16, no int4) and dumps the
hidden state after layer 0 + the last-position argmax, to localize the bug."""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from models.gemma_e2b import load_e2b_weights, GemmaE2bWeights, E_NLAYERS
from runtime.engine import new_session, upload_ids, argmax_f
from runtime.tensor_ops import DevBuf
from runtime.tensor_ops import probe_simd_gemm

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"


def dump(
    ctx: DeviceContext, mut h: DevBuf, T: Int, hd: Int, label: String
) raises:
    ctx.synchronize()
    with h.map_to_host() as m:
        var t = TileTensor(m, row_major(T * hd))
        var s0 = String(label) + " row0[:6]:"
        for i in range(6):
            s0 += " " + String(rebind[Scalar[DType.float32]](t[i]))
        print(s0)
        var sl = String(label) + " rowLast[:6]:"
        var base = (T - 1) * hd
        for i in range(6):
            sl += " " + String(rebind[Scalar[DType.float32]](t[base + i]))
        print(sl)


def main() raises:
    var ctx = DeviceContext()
    print("loading e2b bf16…")
    var gw = load_e2b_weights(ctx, SNAP, False)  # bf16 for exact compare
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    var hd = gw.hidden

    var ids: List[Int] = [2, 4521, 1902]
    var T = len(ids)
    var s = new_session(ctx, 64, cfg.nlayers, cfg.nkv)
    var ids_dev = upload_ids(ctx, ids)
    var h = gw.embed_prompt(ctx, ids_dev, T)
    for l in range(cfg.nlayers):
        h = gw.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)
        dump(ctx, h, T, hd, "after L" + String(l))
    var logits = gw.lm_logits(ctx, h, T, s.dummy)
    print("last-pos argmax:", argmax_f(logits), " (HF ref: 236787)")
    # Dump layer-13 K/V cache (the sliding shared source) pos0 head0 dims0-5.
    dump(ctx, s.kcs[13], 1, 256, "L13 Kcache pos0")
    dump(ctx, s.vcs[13], 1, 256, "L13 Vcache pos0")
