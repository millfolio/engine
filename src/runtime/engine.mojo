"""Model-agnostic decode runtime, generic over the `ModelWeights` trait: the KV-
cache `Session`, prefill (`sess_prefill`, `sess_prefill_suffix`), per-token decode
(`sess_step`), and the greedy / sampled `generate` loops. The model family supplies
the three weight-touching steps (embed a prompt, run one decoder layer, produce
last-position logits) via the trait; everything here — caching, the loop, sampling
— is reused unchanged across families. Adding a family needs no engine change."""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from runtime.tensor_ops import DevBuf
from runtime.sampling import process_logits, sample, argmax_f
from runtime.model_iface import ModelWeights


def upload_ids(ctx: DeviceContext, vals: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var n = len(vals)
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    with d.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(vals[i]))
    return d^


def argmax_last[W: ModelWeights](ctx: DeviceContext, mut w: W, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> Int:
    """Greedy: last-position logits (via the family's LM head) → argmax."""
    return argmax_f(w.lm_logits(ctx, h, T, dummy))


def logits_last[W: ModelWeights](ctx: DeviceContext, mut w: W, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
    """Last-position logits on the host (the family's tied LM head + any softcap)."""
    return w.lm_logits(ctx, h, T, dummy)


# ── decode session: KV caches + position, the per-step primitive ──────────────

@fieldwise_init
struct Session(Movable):
    """Holds the per-layer KV caches and the current position. `prefill` runs the
    prompt and returns the last-position logits; `step` advances one token. Shared
    by greedy/sampled generate and the server's streaming loop."""
    var kcs: List[DevBuf]
    var vcs: List[DevBuf]
    var dummy: DevBuf
    var cache_len: Int
    var pos: Int


def new_session(ctx: DeviceContext, max_seq: Int, nlayers: Int, nkv: Int) raises -> Session:
    var cache_len = max_seq * nkv
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(nlayers):
        kcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
        vcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
    return Session(kcs^, vcs^, ctx.enqueue_create_buffer[DType.float32](1), cache_len, 0)


def sess_prefill[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session, prompt: List[Int]) raises -> List[Float32]:
    var P = len(prompt)
    var ids_dev = upload_ids(ctx, prompt)
    var h = w.embed_prompt(ctx, ids_dev, P)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, P, 0, s.cache_len, s.dummy)
    s.pos = P
    return w.lm_logits(ctx, h, P, s.dummy)


# Below this suffix length, prefill is sub-second and frequent (the prefix-cache
# common case), so progress reporting — and its per-layer synchronize — is skipped
# entirely, leaving that hot path byte-for-byte as before.
comptime PROGRESS_MIN_TOK = 2048
comptime PROGRESS_EVERY_NS = 5_000_000_000   # ~5 s between progress lines


def _ktok(n: Int) -> String:
    """Compact token count: 9562 -> '9.5k', 800 -> '800'."""
    if n < 1000:
        return String(n)
    return String(n // 1000) + "." + String((n % 1000) // 100) + "k"

def _dur(secs: Float64) -> String:
    var s = Int(secs + 0.5)
    if s < 90:
        return String(s) + "s"
    return String(s // 60) + "m" + String(s % 60) + "s"


def sess_prefill_suffix[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session,
                        suffix: List[Int], offset: Int, progress: Bool = False) raises -> List[Float32]:
    """Prefill `suffix` tokens at cache position `offset`, reusing the K/V already
    stored in rows [0, offset). Returns the last-row logits. This is the engine
    behind the server's cross-request prefix cache; `sess_prefill` is just the
    offset==0 / whole-prompt special case. RoPE positions come from `offset`, so
    the rotated K and the attention mask stay correct for the reused prefix.

    With `progress` (and a large enough suffix), prints a throttled stdout line
    with percent done + ETA; gated off below PROGRESS_MIN_TOK so frequent tiny
    prefills are untouched."""
    var Q = len(suffix)
    var nlayers = w.config().nlayers
    var ids_dev = upload_ids(ctx, suffix)
    var h = w.embed_prompt(ctx, ids_dev, Q)
    var report = progress and Q >= PROGRESS_MIN_TOK
    var t0 = perf_counter_ns()
    var last = t0
    for l in range(nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, Q, offset, s.cache_len, s.dummy)
        if report:
            ctx.synchronize()
            var now = perf_counter_ns()
            if Float64(now - last) >= Float64(PROGRESS_EVERY_NS):
                var done = l + 1
                var elapsed = Float64(now - t0) / 1.0e9
                var eta = elapsed * Float64(nlayers - done) / Float64(done)
                print("  prefill ", _ktok(Q), "tok: ", (done * 100) // nlayers,
                      "% (layer ", done, "/", nlayers, "), ~", _dur(eta), " left", sep="")
                last = now
    s.pos = offset + Q
    return w.lm_logits(ctx, h, Q, s.dummy)


def sess_token_logprobs[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session,
                        tokens: List[Int]) raises -> List[Float32]:
    """Teacher-forced log-probs for a token window: a fresh forward (offset 0) over
    `tokens`, then log P(tokens[i+1] | tokens[0..i]) for i in [0, T-2] — T-1 floats,
    each conditioned on ≥1 in-window token. Drives perplexity / echo logprobs; the
    n×vocab logits stay on-GPU (token_logprobs)."""
    var T = len(tokens)
    var ids_dev = upload_ids(ctx, tokens)
    var h = w.embed_prompt(ctx, ids_dev, T)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, T, 0, s.cache_len, s.dummy)
    s.pos = T
    var targets = List[Int]()
    for i in range(1, T):
        targets.append(tokens[i])
    return w.token_logprobs(ctx, h, T - 1, targets, s.dummy)


def sess_step[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session, token: Int) raises -> List[Float32]:
    var one = upload_ids(ctx, [token])
    var h = w.embed_prompt(ctx, one, 1)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, 1, s.pos, s.cache_len, s.dummy)
    s.pos += 1
    return w.lm_logits(ctx, h, 1, s.dummy)


def generate[W: ModelWeights](ctx: DeviceContext, mut w: W, prompt: List[Int], max_new: Int) raises -> List[Int]:
    """Greedy decode: prefill the prompt then emit tokens until EOS or max_new."""
    var cfg = w.config()
    var s = new_session(ctx, len(prompt) + max_new + 2, cfg.nlayers, cfg.nkv)
    var nxt = argmax_f(sess_prefill(ctx, w, s, prompt))
    var gen = List[Int]()
    gen.append(nxt)
    while len(gen) < max_new and nxt != cfg.eos1 and nxt != cfg.eos2:
        nxt = argmax_f(sess_step(ctx, w, s, nxt))
        gen.append(nxt)
    return gen^


def generate_sample[W: ModelWeights](ctx: DeviceContext, mut w: W, prompt: List[Int], max_new: Int,
                    temp: Float32, top_k: Int, top_p: Float32, rep_pen: Float32,
                    seed: UInt64) raises -> List[Int]:
    """Greedy-structure decode but draw each token from the processed distribution."""
    var cfg = w.config()
    var s = new_session(ctx, len(prompt) + max_new + 2, cfg.nlayers, cfg.nkv)
    var rng = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)
    var context = prompt.copy()
    var nxt = sample(process_logits(sess_prefill(ctx, w, s, prompt), context, temp, top_k, top_p, rep_pen), rng)
    var gen = List[Int]()
    gen.append(nxt)
    context.append(nxt)
    while len(gen) < max_new and nxt != cfg.eos1 and nxt != cfg.eos2:
        var dist = process_logits(sess_step(ctx, w, s, nxt), context, temp, top_k, top_p, rep_pen)
        nxt = sample(dist, rng)
        context.append(nxt)
        gen.append(nxt)
    return gen^


# ── speculative decoding (greedy = exact) ─────────────────────────────────────
# Draft K tokens cheaply, verify them all in ONE batched target forward, accept
# the longest prefix the target itself would have produced. For greedy this is
# bit-identical to `generate` — every accepted token equals the target's argmax —
# but it collapses up to K+1 per-token forwards (each ~48 dispatches on Gemma)
# into a single Q=K+1 forward, trading the dispatch-bound decode for a few extra
# matmul rows. The draft here is prompt-lookup (n-gram): no second model, just
# the context's own recent history; works well for repetitive / code / quoted text.


def _argmax_row(logits: List[Float32], row: Int, vocab: Int) -> Int:
    """Argmax over one position's logits in a flat row-major T×vocab buffer."""
    var base = row * vocab
    var best = 0
    var bestv = logits[base]
    for i in range(1, vocab):
        var v = logits[base + i]
        if v > bestv:
            bestv = v
            best = i
    return best


def _ngram_draft(context: List[Int], K: Int, ngram: Int) -> List[Int]:
    """Prompt-lookup draft: find the most recent earlier occurrence of the last
    `ngram` tokens of `context` and propose the up-to-K tokens that followed it.
    Empty when there's no match (the caller falls back to a single-token step)."""
    var draft = List[Int]()
    var n = len(context)
    if n < ngram + 1:
        return draft^
    var i = n - ngram - 1
    while i >= 0:
        var matched = True
        for j in range(ngram):
            if context[i + j] != context[n - ngram + j]:
                matched = False
                break
        if matched:
            var src = i + ngram
            for k in range(K):
                if src + k < n:
                    draft.append(context[src + k])
                else:
                    break
            return draft^
        i -= 1
    return draft^


def sess_verify[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session, batch: List[Int]) raises -> List[Float32]:
    """Forward `batch` at cache position s.pos (RoPE positions s.pos..s.pos+Q-1),
    returning logits for ALL Q positions (row-major Q×vocab). Does NOT advance
    s.pos — the caller commits the accepted prefix length. KV rows written here for
    accepted tokens are valid; any rejected tail is harmlessly overwritten next
    round (linear KV: the next forward writes from the committed s.pos)."""
    var Q = len(batch)
    var ids_dev = upload_ids(ctx, batch)
    var h = w.embed_prompt(ctx, ids_dev, Q)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs, s.vcs, Q, s.pos, s.cache_len, s.dummy)
    return w.lm_logits_all(ctx, h, Q, s.dummy)


def _spec_emit(mut gen: List[Int], mut context: List[Int], tok: Int,
               max_new: Int, eos1: Int, eos2: Int) -> Bool:
    """Append a generated token; return True if generation should now stop. Keeps
    the emitted sequence (and its stop point) bit-identical to greedy `generate`."""
    gen.append(tok)
    context.append(tok)
    return len(gen) >= max_new or tok == eos1 or tok == eos2


def generate_spec[W: ModelWeights](ctx: DeviceContext, mut w: W, prompt: List[Int],
                  max_new: Int, K: Int = 4, ngram: Int = 3, verbose: Bool = False) raises -> List[Int]:
    """Greedy speculative decode (bit-identical output to `generate`). Drafts K
    tokens via prompt-lookup, verifies the [c0]+drafts batch in one target
    forward, accepts the longest prefix matching the target's argmaxes, and takes
    the target's correction (or its next prediction if all K accepted) as the next
    committed token. Returns the same tokens `generate` would, just fewer
    forwards."""
    var cfg = w.config()
    var s = new_session(ctx, len(prompt) + max_new + K + 4, cfg.nlayers, cfg.nkv)
    var context = prompt.copy()
    var gen = List[Int]()
    var c0 = argmax_f(sess_prefill(ctx, w, s, prompt))     # s.pos = P; c0's KV not yet written
    var stop = _spec_emit(gen, context, c0, max_new, cfg.eos1, cfg.eos2)
    var n_verify = 0          # batched target forwards
    var n_step = 0            # single-token fallback forwards (no draft)
    var n_drafted = 0         # draft tokens proposed
    var n_accepted = 0        # draft tokens accepted
    while not stop:
        var drafts = _ngram_draft(context, K, ngram)
        if len(drafts) == 0:
            c0 = argmax_f(sess_step(ctx, w, s, c0))         # writes c0's KV, s.pos += 1
            n_step += 1
            stop = _spec_emit(gen, context, c0, max_new, cfg.eos1, cfg.eos2)
            continue
        var batch = List[Int]()
        batch.append(c0)
        for d in drafts:
            batch.append(d)
        var G = sess_verify(ctx, w, s, batch)               # Q×vocab; s.pos unchanged
        n_verify += 1
        n_drafted += len(drafts)
        var vocab = len(G) // len(batch)
        # Row i = target's prediction after batch[i]; batch[i+1] = drafts[i].
        var accepted = 0
        var carry = -1
        for i in range(len(drafts)):
            var pred = _argmax_row(G, i, vocab)
            if drafts[i] == pred:
                accepted += 1
            else:
                carry = pred
                break
        if carry == -1:                                     # all K drafts accepted
            carry = _argmax_row(G, len(drafts), vocab)
        # Commit c0 (row at old s.pos) + the accepted drafts.
        n_accepted += accepted
        s.pos = s.pos + 1 + accepted
        for i in range(accepted):
            stop = _spec_emit(gen, context, drafts[i], max_new, cfg.eos1, cfg.eos2)
            if stop:
                break
        if stop:
            break
        c0 = carry
        stop = _spec_emit(gen, context, c0, max_new, cfg.eos1, cfg.eos2)
    if verbose:
        var fwd = n_verify + n_step + 1                     # +1 prefill
        var rate = Float64(n_accepted) / Float64(n_drafted) if n_drafted > 0 else 0.0
        print("  spec: ", len(gen), " toks in ", fwd, " forwards (",
              n_verify, " verify Q=", K + 1, ", ", n_step, " single); drafts ",
              n_accepted, "/", n_drafted, " accepted (", rate * 100.0, "%)", sep="")
    return gen^


def generate_spec_draft[TW: ModelWeights, DW: ModelWeights](
    ctx: DeviceContext, mut target: TW, mut draft: DW, prompt: List[Int],
    max_new: Int, K: Int = 4, verbose: Bool = False) raises -> List[Int]:
    """Greedy speculative decode with a DRAFT MODEL (still bit-identical to
    `generate` on the target). The small draft proposes K tokens autoregressively;
    the target verifies the [c0]+drafts batch in ONE forward and accepts the
    longest prefix matching its own argmaxes. Both models keep their own KV session
    advanced in lockstep at the committed length: the draft writes c0+drafts as it
    proposes; after the accept the draft's `pos` is rolled back to drop the rejected
    tail (linear KV, overwritten next round). Unlike prompt-lookup, the draft can
    predict free text, so acceptance holds up on non-echoing prose."""
    var tcfg = target.config()
    var dcfg = draft.config()
    var cap = len(prompt) + max_new + K + 4
    var ts = new_session(ctx, cap, tcfg.nlayers, tcfg.nkv)
    var ds = new_session(ctx, cap, dcfg.nlayers, dcfg.nkv)
    var context = prompt.copy()
    var gen = List[Int]()
    var c0 = argmax_f(sess_prefill(ctx, target, ts, prompt))   # ts.pos = P; c0 not yet cached
    _ = sess_prefill(ctx, draft, ds, prompt)                   # ds.pos = P (draft prompt KV)
    var stop = _spec_emit(gen, context, c0, max_new, tcfg.eos1, tcfg.eos2)
    var n_round = 0
    var n_drafted = 0
    var n_accepted = 0
    while not stop:
        # ── draft K tokens from c0 (writes c0 + drafts into the draft's KV) ──
        var drafts = List[Int]()
        var dl = sess_step(ctx, draft, ds, c0)                 # c0 KV → draft, ds.pos += 1
        var cur = argmax_f(dl)
        for _ in range(K):
            drafts.append(cur)
            dl = sess_step(ctx, draft, ds, cur)               # draft token KV
            cur = argmax_f(dl)                                # last cur (after d_{K-1}) discarded
        # ── verify the batch with the target ──
        var batch = List[Int]()
        batch.append(c0)
        for d in drafts:
            batch.append(d)
        var G = sess_verify(ctx, target, ts, batch)           # Q×vocab; ts.pos unchanged
        var vocab = len(G) // len(batch)
        n_round += 1
        n_drafted += len(drafts)
        var accepted = 0
        var carry = -1
        for i in range(len(drafts)):
            var pred = _argmax_row(G, i, vocab)
            if drafts[i] == pred:
                accepted += 1
            else:
                carry = pred
                break
        if carry == -1:
            carry = _argmax_row(G, len(drafts), vocab)
        n_accepted += accepted
        # ── commit positions: target +1+accepted; draft rolls back to drop rejects.
        ts.pos = ts.pos + 1 + accepted
        ds.pos = ds.pos - (K + 1) + (1 + accepted)            # was committed+K+1
        var brk = False
        for i in range(accepted):
            stop = _spec_emit(gen, context, drafts[i], max_new, tcfg.eos1, tcfg.eos2)
            if stop:
                brk = True
                break
        if brk:
            break
        c0 = carry
        stop = _spec_emit(gen, context, c0, max_new, tcfg.eos1, tcfg.eos2)
    if verbose:
        var rate = Float64(n_accepted) / Float64(n_drafted) if n_drafted > 0 else 0.0
        print("  spec-draft: ", len(gen), " toks, ", n_round, " rounds (",
              (n_round * (K + 2)) + 1, " draft + ", n_round, " verify fwd); drafts ",
              n_accepted, "/", n_drafted, " accepted (", rate * 100.0, "%)", sep="")
    return gen^
