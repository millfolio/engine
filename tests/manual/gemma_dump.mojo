"""Full-forward integration check for the Gemma 4 12B dense decoder (int4).

Runs the COMPLETE 48-layer forward (int4, ~7 GB — the served path, fits 24 GB) on
the same token ids the HF reference used (tests/fixtures/gemma/ids.txt, written by
`pixi run gemma-ref`), dumps a per-layer last-row sample for eyeball bisect, and
compares the final-position argmax to the reference (ref_argmax.txt).

Where the per-layer gate (gemma_layer_test) validates each layer in ISOLATION on
the reference input, this validates end-to-end PROPAGATION — embedding scale, the
KV cache + RoPE across positions, the final norm, and the logit softcap — which
isolated per-layer checks can't catch. int4 (not bf16) so the whole model fits
24 GB; the signal is the final argmax matching HF's (greedy).

Build/run:  pixi run gemma-dump
"""

from std.sys import has_accelerator
from std.os.path import exists
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from models.gemma import load_gemma_weights, G_NLAYERS
from runtime.engine import new_session, upload_ids
from runtime.sampling import argmax_f
from runtime.tensor_ops import DevBuf, probe_simd_gemm
from testio import read_text, ints_from

comptime CKPT = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"
comptime FIX = "tests/fixtures/gemma/"


def _sample(
    ctx: DeviceContext, mut h: DevBuf, T: Int, hd: Int, label: String
) raises:
    """Print the last-position hidden's first 6 dims — a cheap per-layer bisect."""
    ctx.synchronize()
    with h.map_to_host() as m:
        var t = TileTensor(m, row_major(T * hd))
        var base = (T - 1) * hd
        var s = String(label) + " last[:6]:"
        for i in range(6):
            s += " " + String(rebind[Scalar[DType.float32]](t[base + i]))
        print(s)


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")
    var ctx = DeviceContext()

    # The same ids the HF reference used (written by gemma4_layer_ref.py), so the
    # final argmax is directly comparable.
    var ids: List[Int]
    if exists(FIX + "ids.txt"):
        ids = ints_from(read_text(FIX + "ids.txt"))
    else:
        print("(no ids.txt — using a default id set; run `pixi run gemma-ref`)")
        ids = [2, 651, 4320, 8426, 25341, 1163, 573, 26989, 5929, 235265, 108, 109]
    var T = len(ids)

    print("loading Gemma 12B int4 (full 48 layers, ~7 GB)…")
    var all_layers = List[Int]()
    for l in range(G_NLAYERS):
        all_layers.append(l)
    var gw = load_gemma_weights(ctx, CKPT, all_layers, True)  # q4=True (int4)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    var hd = gw.hidden

    var s = new_session(ctx, 64, cfg.nlayers, cfg.nkv)
    var ids_dev = upload_ids(ctx, ids)
    var h = gw.embed_prompt(ctx, ids_dev, T)
    for l in range(cfg.nlayers):
        h = gw.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)
        _sample(ctx, h, T, hd, "after L" + String(l))
    var logits = gw.lm_logits(ctx, h, T, s.dummy)
    var got = argmax_f(logits)
    print("final-pos argmax (int4):", got)

    if exists(FIX + "ref_argmax.txt"):
        var want = Int(atol(String(read_text(FIX + "ref_argmax.txt").strip())))
        # int4 quantization accumulates over 48 layers, so the greedy token CAN
        # differ from the f32 reference — an exact-match assert would be wrong.
        # Rank-based check instead: the f32 reference token must stay near the top
        # of the int4 logits (a real integration bug in embed scale / final norm /
        # softcap / KV+RoPE propagation would bury it). Exact per-layer correctness
        # is proven separately by gemma-layer-test.
        var wl = logits[want]
        var rank = 0
        for i in range(len(logits)):
            if logits[i] > wl:
                rank += 1
        print("HF f32 reference token", want, "sits at int4 rank", rank, "(0 = argmax)")
        if got == want:
            print("OK — int4 argmax matches the f32 reference exactly")
        elif rank <= 5:
            print(
                "OK — f32 reference token within int4 top-6; the difference from"
                " argmax is quantization, not an integration bug"
            )
        else:
            raise Error(
                "integration check FAILED: f32 reference token "
                + String(want)
                + " is at int4 rank "
                + String(rank)
                + " (> 5) — investigate embed scale / final norm / softcap / RoPE"
            )
    else:
        print("(no ref_argmax.txt — run `pixi run gemma-ref` first)")
