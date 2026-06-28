"""Per-stage timing of the gemma-4 e2b M=1 decode forward (the draft model's hot
path), to find what to fuse. Loads e2b only (no 12B → no memory contention),
prefills, then times N single-token forwards split into embed / 35-layer loop /
LM head. `pixi run e2b-timing`."""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from models.gemma_e2b import load_e2b_weights, GemmaE2bWeights, E_NLAYERS
from runtime.engine import (
    new_session,
    upload_ids,
    argmax_f,
    sess_prefill,
    Session,
)
from runtime.tensor_ops import DevBuf, probe_simd_gemm

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"


def main() raises:
    var ctx = DeviceContext()
    print("loading e2b int4…")
    var gw = load_e2b_weights(ctx, SNAP, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    var s = new_session(ctx, 512, cfg.nlayers, cfg.nkv)
    var prompt: List[Int] = [2, 4521, 1902, 235, 108, 1024, 555]
    _ = sess_prefill(ctx, gw, s, prompt)

    var reps = 30
    var tok = 5
    var t_embed = 0.0
    var t_layers = 0.0
    var t_head = 0.0
    var t_total = 0.0
    # warmup
    for _ in range(3):
        var one = upload_ids(ctx, [tok])
        var h = gw.embed_prompt(ctx, one, 1)
        for l in range(cfg.nlayers):
            h = gw.run_layer(
                ctx, l, h, s.kcs, s.vcs, 1, s.pos, s.cache_len, s.dummy
            )
        _ = gw.lm_logits(ctx, h, 1, s.dummy)
    s.pos = len(prompt)
    for _ in range(reps):
        var a = perf_counter_ns()
        var one = upload_ids(ctx, [tok])
        var h = gw.embed_prompt(ctx, one, 1)
        ctx.synchronize()
        var b = perf_counter_ns()
        for l in range(cfg.nlayers):
            h = gw.run_layer(
                ctx, l, h, s.kcs, s.vcs, 1, s.pos, s.cache_len, s.dummy
            )
        ctx.synchronize()
        var c = perf_counter_ns()
        _ = gw.lm_logits(ctx, h, 1, s.dummy)
        var d = perf_counter_ns()
        t_embed += Float64(b - a) / 1.0e6
        t_layers += Float64(c - b) / 1.0e6
        t_head += Float64(d - c) / 1.0e6
        t_total += Float64(d - a) / 1.0e6
        s.pos = len(prompt)  # keep cache position fixed for the bench
    print("e2b M=1 forward (avg of ", reps, "):", sep="")
    print("  embed+PLE-setup: ", t_embed / Float64(reps), " ms", sep="")
    print(
        "  35-layer loop:   ",
        t_layers / Float64(reps),
        " ms (",
        t_layers / Float64(reps) / 35.0,
        " ms/layer)",
        sep="",
    )
    print("  LM head:         ", t_head / Float64(reps), " ms", sep="")
    print("  TOTAL:           ", t_total / Float64(reps), " ms", sep="")
