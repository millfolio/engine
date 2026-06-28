"""Corpus perplexity for the millfolio e2b model (gemma_e2b.mojo), comparable to
the assay numbers for Qwen / 12B. Reads pre-tokenized windows (BOS-prepended) from
.scratch/e2b_corpus_ids.txt — one window per line, space-separated token ids — runs
a teacher-forced forward per window, and aggregates NLL corpus-wide:
    PPL = exp( Σ -logP(token) / Σ tokens ).
Arg: "int4" loads group-128 int4 weights; default is bf16.
"""

from std.sys import argv
from std.gpu.host import DeviceContext
from models.gemma_e2b import load_e2b_weights
from runtime.engine import new_session, upload_ids
from runtime.tensor_ops import probe_simd_gemm

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"
comptime IDS = (
    "/Users/mseritan/dev/millfolio/engine/.scratch/e2b_corpus_ids.txt"
)


def main() raises:
    var q4 = False
    var a = argv()
    for i in range(len(a)):
        if String(a[i]) == "int4":
            q4 = True
    var ctx = DeviceContext()
    print("loading e2b", "int4" if q4 else "bf16", "…")
    var gw = load_e2b_weights(ctx, SNAP, q4)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()

    var text: String
    with open(IDS, "r") as f:
        text = f.read()

    var total_nll = Float64(0.0)
    var total_tok = 0
    var nwin = 0
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li].strip())
        if line.byte_length() == 0:
            continue
        var ids = List[Int]()
        var parts = line.split(" ")
        for p in range(len(parts)):
            var tokstr = String(parts[p].strip())
            if tokstr.byte_length() > 0:
                ids.append(atol(tokstr))
        var T = len(ids)
        if T < 2:
            continue

        var s = new_session(ctx, 1024, cfg.nlayers, cfg.nkv)
        var ids_dev = upload_ids(ctx, ids)
        var h = gw.embed_prompt(ctx, ids_dev, T)
        for l in range(cfg.nlayers):
            h = gw.run_layer(
                ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy
            )
        var targets = List[Int]()
        for i in range(1, T):
            targets.append(ids[i])
        var lp = gw.token_logprobs(ctx, h, T - 1, targets, s.dummy)
        for i in range(len(lp)):
            total_nll += -Float64(lp[i])
        total_tok += len(lp)
        nwin += 1
        if nwin % 5 == 0:
            var cur = 2.718281828459045 ** (total_nll / Float64(total_tok))
            print("  window", nwin, " tokens=", total_tok, " running PPL=", cur)

    var mean = total_nll / Float64(total_tok)
    print("\n=== e2b", "int4" if q4 else "bf16", "===")
    print("  PPL     =", 2.718281828459045**mean)
    print("  mean_nll=", mean, " nats/token")
    print("  tokens  =", total_tok, " over", nwin, "windows")
