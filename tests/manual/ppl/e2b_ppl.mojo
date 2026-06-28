"""Dump millfolio e2b per-position logprobs for a fixed token sequence, to compare
against the HF gemma4-e2b reference (.scratch/e2b_ppl_ref.py). Same ids fed to
both; if millfolio is flat (no near-0 logprobs on predictable tokens) while HF is
confident, the shared logprobs/final-norm path is miscalibrated."""

from std.gpu.host import DeviceContext
from std.math import log
from models.gemma_e2b import load_e2b_weights
from runtime.engine import new_session, upload_ids
from runtime.tensor_ops import probe_simd_gemm

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"


def main() raises:
    var ctx = DeviceContext()
    print("loading e2b bf16…")
    var gw = load_e2b_weights(
        ctx, SNAP, True
    )  # INT4 — does quantization break calibration?
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()

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
    for l in range(cfg.nlayers):
        h = gw.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)

    var targets = List[Int]()
    for i in range(1, T):
        targets.append(ids[i])
    var lp = gw.token_logprobs(ctx, h, T - 1, targets, s.dummy)

    var nll = Float64(0.0)
    print("MILLFOLIO_E2B_LP=[null", end="")
    for i in range(len(lp)):
        print(",", lp[i], end="")
        nll += -Float64(lp[i])
    print("]")
    var mean = nll / Float64(len(lp))
    print("MILLFOLIO_E2B mean_nll=", mean, " PPL=", 2.718281828459045**mean)
