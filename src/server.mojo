"""OpenAI-compatible HTTP server, pure Mojo on the GPU, over flare (ARCHITECTURE.md §6).

Earlier this engine talked to libc sockets directly because flare pinned Mojo
1.0.0b1 while the GPU code needs the 1.0.0b2 nightly's std.gpu API (§11 #11).
That conflict is resolved: the ../flare fork now builds under 1.0.0b2 (one
systematic fix — `unsafe_from_address=0` → `=Int(0)`, since 1.0.0b2 makes
`UnsafePointer` non-nullable — plus its `libflare_tls.so` rebuilt from source).
So we reuse flare's kqueue reactor + Router/Handler/SSE just like ../max-backend,
but wired to *this* engine's real GPU generation instead of MAX.

Endpoints (each shares the one GPU-resident Session-based decode path):
    GET  /v1/models
    POST /v1/chat/completions   (stream + non-stream)
    POST /v1/responses          (stream + non-stream)  ← what opencode drives

The model (Weights + DeviceContext + tokenizer + chat template) is loaded once
into a heap `ServerState`; the `Api` flare Handler carries a pointer to it. The
pointer dodges flare's read-only `serve(self, …)` borrow so generation can take
`mut w` (the GPU kernels bind mutable buffers). Safe because flare's reactor is
single-threaded here — one request in flight at a time (max-backend §10 #4).

    pixi run serve            # listens on 127.0.0.1:8000
    curl -s localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.time import perf_counter_ns
from std.sys import argv
from std.os import getenv
from std.os.path import exists, isdir

from flare.prelude import *
from flare.http import Handler, SseChannel, SseEvent, sse_response

from std.utils import Variant
from model import (
    Weights, load_weights, probe_simd_gemm, EOS1, EOS2,
    Session, new_session, sess_prefill_suffix, sess_step, sess_verify, sess_token_logprobs,
    _ngram_draft, _argmax_row,
    argmax_f, process_logits, sample, sess_embed,
    FAMILY_QWEN, FAMILY_GEMMA, ModelConfig, TOOL_GEMMA,
    GemmaWeights, load_gemma_weights, G_NLAYERS,
    parse_tool_calls, parse_gemma_tool_calls, ParsedReply, ToolCall,
    BlockCache,
)
from tokenizer import Tokenizer, load_tokenizer, load_tokenizer_json, load_gemma_tokenizer_json
from chat import load_chat_template, render_value, json_escape_str
from template import Template
from value import Value
from json import parse_json, bytes_to_string

# Persistent KV-cache capacity (tokens). One Session of this size lives on
# ServerState for the whole process so successive requests in an agent loop reuse
# the prefix they share instead of re-prefilling it. 32768 = Qwen2.5's native
# context; the cache is ~MAX_SEQ * 24 KiB ≈ 805 MB resident on the GPU.
comptime MAX_SEQ = 32768

# Gemma's KV cache is uniform at nkv=2048 (the max of its sliding/full layer
# types) across all 48 layers → ~768 KiB/token, so a 32k context would be ~26 GB
# on top of the ~7 GB int4 weights. Cap Gemma's persistent cache so weights +
# cache + the secondary embed model fit a 24 GB unified GPU: 4096 * 768 KiB ≈
# 3.2 GB (raise on a larger machine).
comptime GEMMA_MAX_SEQ = 4096
# Gemma emits this after finishing its tool call(s); it marks the model's
# turn boundary (the tool result goes next). Stop decoding there so we don't
# greedily hallucinate a fake tool result past the call.
comptime GEMMA_TOOL_RESPONSE = 50

# Disk-backed prefix cache: K/V persisted in BLOCK_TOK-token blocks so prefills
# survive restarts and are shared across conversations (blockcache.mojo).
comptime BLOCK_TOK = 256
comptime KV_BUDGET_BYTES = 8 * 1024 * 1024 * 1024   # 8 GB LRU cap

comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"
# Default served model ids by detected arch (used when no explicit id is given on
# the CLI). The served id is otherwise whatever `serve <hf-id>` was launched with,
# and is what /v1/models and every response report.
comptime MODEL_05B = "Qwen/Qwen2.5-0.5B-Instruct"
comptime MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
comptime MODEL_GEMMA = "google/gemma-4-12b-it"
# Default SECONDARY embedding model (arch==2). Resolved from the HF cache when no
# $EMBED_SAFETENSORS / config `embed_model` is given; if it isn't cached either,
# the embedding endpoint stays unloaded (/v1/embeddings → 503).
comptime MODEL_EMBED = "Qwen/Qwen3-Embedding-0.6B"
comptime PORT = 8000
# Engine version, reported by GET /v1/version (used by the Millfolio menu app to
# detect a running engine and show its version). Bump on releases. Placeholder
# scheme for now; wire to a real build/version source later.
comptime MILLFOLIO_VERSION = "0.1.0"

# jinja2.mojo Value tags (value.mojo)
comptime VBOOL = 2
comptime VINT = 3
comptime VFLOAT = 4
comptime VSTR = 5
comptime VLIST = 6
# sampling defaults (generation_config.json) when temperature > 0
comptime DEF_TOPK = 20
comptime DEF_TOPP = Float32(0.8)
comptime DEF_REP = Float32(1.1)
comptime DEF_MAXNEW = 256
comptime SEED = UInt64(0x9E3779B97F4A7C15)

# Speculative decoding (greedy/temp==0 only, where it is bit-exact). Prompt-lookup
# (n-gram) drafts K tokens from the context's own history and verifies them in one
# batched forward (mm_w's batched int4 GEMV at small M). A win when output echoes
# the context (code edits, tool results, quoting) — the agentic-coding common case.
# K=7 → a Q=8 verify, which the dedicated 1-tile MMA int4 GEMM runs flat (~425 ms
# vs ~125 ms single-step), so ~6/8 accepted beats single-stepping; the echo-heavy
# agentic-coding case clears that easily (~80% accept → ~1.36×). An adaptive guard
# pauses drafting after SPEC_COLD_LIMIT consecutive zero-accept verifies and
# re-probes after SPEC_COOLDOWN tokens, bounding non-echo text to ~baseline.
comptime SPEC_K = 7
comptime SPEC_NGRAM = 3
comptime SPEC_COLD_LIMIT = 2
comptime SPEC_COOLDOWN = 32

# Responses-API ids (opencode / Vercel AI SDK)
comptime RESP_ID = "resp_millfolio"
comptime MSG_ID = "msg_millfolio"


# ── Shared model state ───────────────────────────────────────────────────────


struct ServerState(Movable):
    """The primary (chat) model, loaded once and reached by the (borrowed-self)
    handler through a pointer so generation can still take `mut w`.

    Optionally also carries a SECONDARY embedding model (`embed_w` / `embed_tok`)
    so one process/port serves both /v1/chat/completions and /v1/embeddings. The
    embedding path (sess_embed) needs no KV-cache Session, so none is stored for
    it. When the PRIMARY model is itself the embedding arch (arch==2), the embed
    fields stay unset and /v1/embeddings falls back to the primary w/tok."""

    var ctx: DeviceContext
    # The primary (chat) model is one of several weight structs, all conforming to
    # the ModelWeights trait, held in a Variant (Mojo has no trait objects). Every
    # weight-touching op dispatches once on `model.isa[…]()`; the rest of the server
    # is family-agnostic and reads per-model behavior (eos, tool style, extra stop)
    # from `cfg`. Adding a model = a Variant arm + a ModelConfig — no scattered ifs.
    var model: Variant[Weights, GemmaWeights]
    var cfg: ModelConfig            # the primary model's config (behavior flags + eos)
    var primary_arch: Int           # Qwen arch (0/1/2) for the embed gate; -1 for Gemma
    var max_seq: Int                # primary KV-cache context cap
    var tok: Tokenizer
    var tmpl: Template
    var sess: Session      # one long-lived KV cache, reused across requests
    var cached: List[Int]  # token ids currently held in sess rows [0, len)
    var model_id: String   # id reported by /v1/models + every response
    var bcache: BlockCache # disk-backed prefix cache (survives restarts)
    # Secondary embedding model (None when the primary is itself arch==2, or when
    # no embedding checkpoint could be resolved at startup). Always Qwen.
    var embed_w: Optional[Weights]
    var embed_tok: Optional[Tokenizer]
    var embed_id: String   # id reported for the embedding model ("" if unset)

    def __init__(out self, var ctx: DeviceContext,
                 var model: Variant[Weights, GemmaWeights], cfg: ModelConfig,
                 primary_arch: Int, max_seq: Int,
                 var tok: Tokenizer, var tmpl: Template, var sess: Session,
                 var model_id: String, var bcache: BlockCache,
                 var embed_w: Optional[Weights], var embed_tok: Optional[Tokenizer],
                 var embed_id: String):
        self.ctx = ctx^
        self.model = model^
        self.cfg = cfg
        self.primary_arch = primary_arch
        self.max_seq = max_seq
        self.tok = tok^
        self.tmpl = tmpl^
        self.sess = sess^
        self.cached = List[Int]()
        self.model_id = model_id^
        self.bcache = bcache^
        self.embed_w = embed_w^
        self.embed_tok = embed_tok^
        self.embed_id = embed_id^


# ── small helpers ────────────────────────────────────────────────────────────


def to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var sb = s.as_bytes()
    for i in range(len(sb)):
        out.append(sb[i])
    return out^

def get_int(req: Value, key: String, default: Int) -> Int:
    var o = req.map_get(key)
    if o:
        var v = o.value()
        if v.tag == VINT:
            return v.i
        if v.tag == VFLOAT:
            return Int(v.f)
    return default

def get_float(req: Value, key: String, default: Float64) -> Float64:
    var o = req.map_get(key)
    if o:
        var v = o.value()
        if v.tag == VFLOAT:
            return v.f
        if v.tag == VINT:
            return Float64(v.i)
    return default

def get_bool(req: Value, key: String, default: Bool) -> Bool:
    var o = req.map_get(key)
    if o and o.value().tag == VBOOL:
        return o.value().b
    return default

def get_str(req: Value, key: String) -> String:
    var o = req.map_get(key)
    if o and o.value().tag == VSTR:
        return o.value().s
    return String("")

def esc(s: String) -> String:
    """JSON-escape a String for embedding in a response body."""
    return json_escape_str(to_bytes(s))

def req_has_tools(req: Value) -> Bool:
    """True iff the request carries a non-empty `tools` array — only then do we
    lift the model's <tool_call> blocks into structured calls (a tools-less
    request that happens to emit the literal text is left as plain content)."""
    var t = req.map_get("tools")
    return Bool(t) and not t.value().is_none() and t.value().truthy()


def responses_to_chat(bv: Value) raises -> Optional[Value]:
    """Map a Responses-API body onto the chat-template's `messages` shape.

    opencode's `@ai-sdk/openai-compatible` provider actually drives
    /v1/chat/completions, so this endpoint is for direct Responses-API clients.
    We support the common `input`-as-string form (+ optional top-level
    `instructions` → system message); array `input` returns None (→ 400). Built
    by re-emitting JSON and reparsing so we reuse parse_json + render_value
    rather than constructing jinja2.mojo Values by hand."""
    if bv.map_get("messages"):
        return bv  # already chat-shaped (tools, if any, ride along)
    var inp = bv.map_get("input")
    if not (inp and inp.value().tag == VSTR):
        return None
    var msgs = String('{"messages":[')
    var instr = get_str(bv, "instructions")
    if instr.byte_length() > 0:
        msgs += '{"role":"system","content":"' + json_escape_str(to_bytes(instr)) + '"},'
    msgs += '{"role":"user","content":"' + json_escape_str(to_bytes(inp.value().s)) + '"}]}'
    var out = parse_json(msgs)
    # Forward any tool definitions so render_value advertises them in the prompt.
    var tools = bv.map_get("tools")
    if tools and not tools.value().is_none():
        out.map_set("tools", tools.value())
    return out^


def complete_utf8_len(b: List[UInt8]) -> Int:
    """Length of the longest prefix of `b` that ends on a UTF-8 char boundary —
    so a multibyte char split across tokens isn't emitted half-formed."""
    var n = len(b)
    if n == 0:
        return 0
    var i = n - 1
    while i >= 0 and (Int(b[i]) & 0xC0) == 0x80:  # skip continuation bytes
        i -= 1
    if i < 0:
        return n
    var lead = Int(b[i])
    var need = 1
    if (lead & 0x80) == 0:
        need = 1
    elif (lead & 0xE0) == 0xC0:
        need = 2
    elif (lead & 0xF0) == 0xE0:
        need = 3
    elif (lead & 0xF8) == 0xF0:
        need = 4
    return n if i + need <= n else i

def slice_bytes(b: List[UInt8], start: Int, stop: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(start, stop):
        out.append(b[i])
    return out^


# ── generation (buffered: produce the whole completion, then frame it) ───────


struct Reply(Movable):
    var ids: List[Int]      # generated token ids (EOS dropped)
    var stopped: Bool       # True if generation ended on EOS, False if length cap
    # Per-request stats (also printed to stdout) — surfaced to clients as a
    # non-standard `"millfolio"` field so a UI can show prefill cost + throughput.
    var n_prompt: Int       # prompt tokens
    var reused: Int         # prompt tokens served from the KV cache (not recomputed)
    var prefilled: Int      # prompt tokens actually prefilled this request
    var pf_ms: Float64      # prefill wall-clock (ms)
    var dec_ms: Float64     # decode wall-clock (ms)
    var tps: Float64        # decode throughput (tokens/sec)

    def __init__(out self, var ids: List[Int], stopped: Bool, n_prompt: Int = 0,
                 reused: Int = 0, prefilled: Int = 0, pf_ms: Float64 = 0.0,
                 dec_ms: Float64 = 0.0, tps: Float64 = 0.0):
        self.ids = ids^
        self.stopped = stopped
        self.n_prompt = n_prompt
        self.reused = reused
        self.prefilled = prefilled
        self.pf_ms = pf_ms
        self.dec_ms = dec_ms
        self.tps = tps


# The three weight-touching ops, each branching ONCE on the Variant's active type
# (the engine calls are parametric over ModelWeights). Adding a model = one arm here.
def _prefill_suffix(mut s: ServerState, suffix: List[Int], reuse: Int) raises -> List[Float32]:
    if s.model.isa[GemmaWeights]():
        return sess_prefill_suffix(s.ctx, s.model[GemmaWeights], s.sess, suffix, reuse, True)
    return sess_prefill_suffix(s.ctx, s.model[Weights], s.sess, suffix, reuse, True)


def _step(mut s: ServerState, token: Int) raises -> List[Float32]:
    if s.model.isa[GemmaWeights]():
        return sess_step(s.ctx, s.model[GemmaWeights], s.sess, token)
    return sess_step(s.ctx, s.model[Weights], s.sess, token)


def _verify(mut s: ServerState, batch: List[Int]) raises -> List[Float32]:
    """Speculative batch forward: logits for ALL positions in `batch` at the
    session's current pos (does NOT advance pos)."""
    if s.model.isa[GemmaWeights]():
        return sess_verify(s.ctx, s.model[GemmaWeights], s.sess, batch)
    return sess_verify(s.ctx, s.model[Weights], s.sess, batch)


def _is_stop(s: ServerState, tok: Int) -> Bool:
    """The server's stop set: the model's EOS pair + its optional extra stop token
    (e.g. Gemma's <|tool_response>), all carried by ModelConfig."""
    return tok == s.cfg.eos1 or tok == s.cfg.eos2 or tok == s.cfg.extra_stop


def _token_logprobs(mut s: ServerState, tokens: List[Int]) raises -> List[Float32]:
    """Teacher-forced per-token logprobs for one window (a fresh offset-0 forward).
    Overwrites the chat KV cache, so invalidate the prefix-cache record."""
    s.sess.pos = 0
    s.cached = List[Int]()
    if s.model.isa[GemmaWeights]():
        return sess_token_logprobs(s.ctx, s.model[GemmaWeights], s.sess, tokens)
    return sess_token_logprobs(s.ctx, s.model[Weights], s.sess, tokens)


def gen_full(mut s: ServerState, ids: List[Int], max_new: Int,
             temp: Float32, top_k: Int, top_p: Float32) raises -> Reply:
    """Run the GPU decode loop to completion for `ids`, honoring OpenAI knobs.

    Reuses the longest prefix already resident in the persistent KV cache (the
    common case in an agent loop, where each turn appends to the same growing
    conversation) and only prefills the diverging suffix. Times prefill vs decode
    separately — each `sess_*` call ends in a device→host logits copy, so the GPU
    is synced at the boundary — and logs a terse per-request line."""
    # Clamp generation so prefill + decode never overrun the cache.
    var room = s.max_seq - len(ids) - 1
    if room < 1:
        raise Error("prompt of " + String(len(ids)) + " tokens exceeds context "
                    + String(s.max_seq))
    var cap = max_new if max_new < room else room

    # Reuse = longest prefix already valid in GPU (in-memory, free), extended by
    # the longest leading run of blocks on disk (loaded into the session). Always
    # recompute the last prompt token so we have its logits.
    var lim = len(s.cached)
    if len(ids) - 1 < lim:
        lim = len(ids) - 1
    var mem_reuse = 0
    while mem_reuse < lim and s.cached[mem_reuse] == ids[mem_reuse]:
        mem_reuse += 1

    var hashes = s.bcache.chained_hashes(ids)
    var disk_run = s.bcache.longest_run(hashes, ids)          # # leading blocks on disk
    while disk_run > 0 and disk_run * BLOCK_TOK > len(ids) - 1:  # keep ≥1 token to prefill
        disk_run -= 1
    var reuse = mem_reuse
    var loaded = 0
    if disk_run * BLOCK_TOK > mem_reuse:
        # load disk blocks covering (mem_reuse … disk_run) into the GPU session
        var first = mem_reuse // BLOCK_TOK
        s.bcache.restore_blocks(s.sess.kcs, s.sess.vcs, hashes, first, disk_run)
        s.sess.pos = disk_run * BLOCK_TOK
        loaded = disk_run - first
        reuse = disk_run * BLOCK_TOK

    var suffix = List[Int]()
    for i in range(reuse, len(ids)):
        suffix.append(ids[i])

    var t0 = perf_counter_ns()
    var logits = _prefill_suffix(s, suffix, reuse)
    var t_pf = perf_counter_ns()
    s.cached = ids.copy()  # prompt is now resident; generated tokens are not cached

    # Persist newly-computed full blocks to disk + refresh LRU (warm prefix stays hot).
    var nblocks = len(ids) // BLOCK_TOK
    s.bcache.store_blocks(s.sess.kcs, s.sess.vcs, hashes, ids, disk_run, nblocks)
    s.bcache.touch_and_evict(hashes, nblocks)
    if loaded > 0:
        print("    kv-cache: restored ", loaded, " block(s) from disk (", loaded * BLOCK_TOK, " tok)", sep="")

    var context = ids.copy()
    var rng = SEED
    var gen = List[Int]()
    var stopped = False
    var last_beat = t_pf   # throttle for the ~5s decode heartbeat
    var spec_acc = 0       # accepted draft tokens (greedy spec path; for the log line)
    var spec_draft = 0     # drafted tokens proposed

    if temp > 0.0:
        # Sampling: per-token decode (speculative decode is only bit-exact for
        # greedy, so it's restricted to temp==0 below).
        while len(gen) < cap:
            var nxt = sample(process_logits(logits, context, temp, top_k, top_p, DEF_REP), rng)
            if _is_stop(s, nxt):
                stopped = True
                break
            gen.append(nxt)
            context.append(nxt)
            if len(gen) >= cap:
                break
            logits = _step(s, nxt)
            # sess_step already synced (host logits copy), so this is real wall-clock.
            var now = perf_counter_ns()
            if Float64(now - last_beat) >= 5.0e9:
                var rate = Float64(len(gen)) * 1.0e9 / Float64(now - t_pf)
                print("  decoding: ", len(gen), " tokens (", Int(rate + 0.5), " tok/s)", sep="")
                last_beat = now
    else:
        # Greedy: prompt-lookup speculative decode against the persistent session.
        # Bit-identical to single-step argmax (every committed token is the
        # target's own argmax). An adaptive guard pauses drafting when acceptance
        # collapses (non-echoing text) and re-probes later, bounding worst case.
        var c0 = argmax_f(logits)
        var draft_on = True
        var cold = 0          # consecutive zero-accept verifies
        var cooldown = 0      # tokens left before re-probing after a pause
        while True:
            if _is_stop(s, c0):
                stopped = True
                break
            gen.append(c0)
            context.append(c0)
            if len(gen) >= cap:
                break

            var drafts = _ngram_draft(context, SPEC_K, SPEC_NGRAM) if draft_on else List[Int]()
            if len(drafts) == 0:
                # No draft match (or paused) → single-token step, commits c0's KV.
                logits = _step(s, c0)
                c0 = argmax_f(logits)
                if cooldown > 0:
                    cooldown -= 1
                    if cooldown == 0:
                        draft_on = True
                        cold = 0
            else:
                var batch = List[Int]()
                batch.append(c0)
                for d in drafts:
                    batch.append(d)
                var G = _verify(s, batch)            # Q×vocab; session pos unchanged
                var vocab = len(G) // len(batch)
                spec_draft += len(drafts)
                var accepted = 0
                var carry = -1
                for i in range(len(drafts)):
                    var pred = _argmax_row(G, i, vocab)
                    if drafts[i] == pred:
                        accepted += 1
                    else:
                        carry = pred
                        break
                if carry == -1:                       # all drafts accepted
                    carry = _argmax_row(G, len(drafts), vocab)
                # Commit c0 (row at old pos) + accepted drafts; rejected tail is
                # overwritten by the next forward (linear KV).
                s.sess.pos = s.sess.pos + 1 + accepted
                spec_acc += accepted
                var brk = False
                for i in range(accepted):
                    if _is_stop(s, drafts[i]):
                        stopped = True
                        brk = True
                        break
                    gen.append(drafts[i])
                    context.append(drafts[i])
                    if len(gen) >= cap:
                        brk = True
                        break
                if brk:
                    break
                c0 = carry
                if accepted == 0:
                    cold += 1
                    if cold >= SPEC_COLD_LIMIT:
                        draft_on = False
                        cooldown = SPEC_COOLDOWN
                else:
                    cold = 0
            var now = perf_counter_ns()
            if Float64(now - last_beat) >= 5.0e9:
                var rate = Float64(len(gen)) * 1.0e9 / Float64(now - t_pf)
                print("  decoding: ", len(gen), " tokens (", Int(rate + 0.5), " tok/s)", sep="")
                last_beat = now
    var t_dec = perf_counter_ns()

    var pf_ms = Float64(t_pf - t0) / 1.0e6
    var dec_ms = Float64(t_dec - t_pf) / 1.0e6
    var tps = Float64(len(gen)) * 1000.0 / dec_ms if dec_ms > 0.0 else 0.0
    print("  gen: prompt=", len(ids), "tok (reused ", reuse, ", prefilled ",
          len(suffix), ")  prefill=", Int(pf_ms + 0.5), "ms  decode=", len(gen),
          "tok ", Int(dec_ms + 0.5), "ms (", Int(tps + 0.5), " tok/s)", sep="")
    if spec_draft > 0:
        print("    spec: ", spec_acc, "/", spec_draft, " drafts accepted (",
              Int(Float64(spec_acc) / Float64(spec_draft) * 100.0 + 0.5), "%)", sep="")
    return Reply(gen^, stopped, len(ids), reuse, len(suffix), pf_ms, dec_ms, tps)


# ── JSON envelopes ───────────────────────────────────────────────────────────



def service_unavailable(msg: String) -> Response:
    """503 with a JSON error body (flare has no built-in 503 helper)."""
    var resp = Response(status=Status.SERVICE_UNAVAILABLE, reason="Service Unavailable",
                        body=to_bytes(msg))
    try:
        resp.headers.set("Content-Type", "application/json")
    except:
        pass
    return resp^

def _model_obj(id: String) -> String:
    return ('{"id":"' + id + '","object":"model","created":0,"owned_by":"millfolio"}')

def models_json(model: String, embed_model: String) -> String:
    """List the chat model, plus the embedding model when one is loaded
    (embed_model == "" means none)."""
    var data = _model_obj(model)
    if embed_model.byte_length() > 0 and embed_model != model:
        data += "," + _model_obj(embed_model)
    return '{"object":"list","data":[' + data + "]}"

def version_json(model: String) -> String:
    return (
        '{"engine":"millfolio","version":"' + MILLFOLIO_VERSION
        + '","model":"' + model + '"}'
    )

def embedding_item_json(index: Int, vec: List[Float32]) -> String:
    """One OpenAI embedding object: the float vector as a JSON array. f32 stringifies
    at full precision (~8 sig figs), enough to reconstruct the unit vector."""
    var arr = String("[")
    for i in range(len(vec)):
        if i > 0:
            arr += ","
        arr += String(vec[i])
    arr += "]"
    return ('{"object":"embedding","index":' + String(index)
            + ',"embedding":' + arr + "}")

def embeddings_json(model: String, data: String, n_tok: Int) -> String:
    """OpenAI /v1/embeddings response envelope around the pre-built `data` array."""
    return (
        '{"object":"list","data":' + data + ',"model":"' + model
        + '","usage":{"prompt_tokens":' + String(n_tok)
        + ',"total_tokens":' + String(n_tok) + "}}"
    )

def millfolio_stats(r: Reply) -> String:
    """The non-standard `millfolio` stats object (prefill cost + decode throughput).
    Additive top-level field — OpenAI clients ignore unknown fields, a millfolio
    UI reads it. Mirrors the `gen:` line the server logs to stdout."""
    return (
        '{"prefill_ms":' + String(Int(r.pf_ms + 0.5))
        + ',"decode_ms":' + String(Int(r.dec_ms + 0.5))
        + ',"tok_per_s":' + String(Int(r.tps + 0.5))
        + ',"prompt_tokens":' + String(r.n_prompt)
        + ',"reused":' + String(r.reused)
        + ',"prefilled":' + String(r.prefilled)
        + ',"gen_tokens":' + String(len(r.ids)) + "}"
    )

def _reasoning_field(reasoning: String) -> String:
    """`,"reasoning_content":"…"` (the DeepSeek-R1 field clients read), or '' when
    there's no reasoning. `reasoning` must already be JSON-escaped."""
    if reasoning.byte_length() > 0:
        return ',"reasoning_content":"' + reasoning + '"'
    return String("")

def completion_json(model: String, content: String, n_prompt: Int, n_gen: Int,
                    finish: String, millfolio: String = String(""),
                    reasoning: String = String("")) -> String:
    var extra = (',"millfolio":' + millfolio) if millfolio.byte_length() > 0 else String("")
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion","created":0,"model":"'
        + model + '","choices":[{"index":0,"message":{"role":"assistant","content":"'
        + content + '"' + _reasoning_field(reasoning) + '},"finish_reason":"' + finish + '"}],'
        + '"usage":{"prompt_tokens":' + String(n_prompt)
        + ',"completion_tokens":' + String(n_gen)
        + ',"total_tokens":' + String(n_prompt + n_gen) + "}" + extra + "}"
    )

def chunk_reasoning_json(model: String, reasoning: String) -> String:
    """A streaming chunk carrying reasoning as a `reasoning_content` delta
    (`reasoning` must already be JSON-escaped)."""
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion.chunk","created":0,"model":"'
        + model + '","choices":[{"index":0,"delta":{"reasoning_content":"' + reasoning
        + '"},"finish_reason":null}]}'
    )

def chunk_json(model: String, delta: String, finish: Bool, fin: String,
               millfolio: String = String("")) -> String:
    var delta_obj = String("{}")
    var finish_reason = String("null")
    if finish:
        finish_reason = '"' + fin + '"'
    else:
        delta_obj = '{"content":"' + delta + '"}'
    var extra = (',"millfolio":' + millfolio) if millfolio.byte_length() > 0 else String("")
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion.chunk","created":0,"model":"'
        + model + '","choices":[{"index":0,"delta":' + delta_obj
        + ',"finish_reason":' + finish_reason + "}]" + extra + "}"
    )

# ── tool-calling envelopes (chat: `tool_calls`; responses: `function_call`) ──
# Call/item ids are deterministic per response (`call_<i>` / `fc_<i>`): the model
# never consumes them and clients only correlate within one turn, so we don't
# need entropy (which the GPU-only build can't cheaply get anyway).


def tool_calls_array_json(calls: List[ToolCall]) -> String:
    """OpenAI chat `message.tool_calls` array. `arguments` is itself a JSON
    *string*, so it's escaped a second time on the way in."""
    var s = String("[")
    for i in range(len(calls)):
        if i > 0:
            s += ","
        s += (
            '{"id":"call_' + String(i) + '","type":"function","function":{"name":"'
            + esc(calls[i].name) + '","arguments":"' + esc(calls[i].arguments) + '"}}'
        )
    return s + "]"

def completion_tools_json(model: String, content: String, calls: List[ToolCall],
                          n_prompt: Int, n_gen: Int,
                          reasoning: String = String("")) -> String:
    var content_field = String("null")
    if content.byte_length() > 0:
        content_field = '"' + esc(content) + '"'
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion","created":0,"model":"'
        + model + '","choices":[{"index":0,"message":{"role":"assistant","content":'
        + content_field + _reasoning_field(esc(reasoning)) + ',"tool_calls":' + tool_calls_array_json(calls)
        + '},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":' + String(n_prompt)
        + ',"completion_tokens":' + String(n_gen)
        + ',"total_tokens":' + String(n_prompt + n_gen) + "}}"
    )

def chunk_role_json(model: String) -> String:
    """Opening streaming chunk announcing the assistant role (content null)."""
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion.chunk","created":0,"model":"'
        + model + '","choices":[{"index":0,"delta":{"role":"assistant","content":null}'
        + ',"finish_reason":null}]}'
    )

def chunk_toolcall_json(model: String, i: Int, call: ToolCall) -> String:
    """One streaming chunk carrying a whole tool call at `index` i (name +
    full arguments). Clients accumulate per index; emitting it in one delta is
    valid since generation is already buffered."""
    var delta = (
        '{"tool_calls":[{"index":' + String(i) + ',"id":"call_' + String(i)
        + '","type":"function","function":{"name":"' + esc(call.name)
        + '","arguments":"' + esc(call.arguments) + '"}}]}'
    )
    return (
        '{"id":"chatcmpl-millfolio","object":"chat.completion.chunk","created":0,"model":"'
        + model + '","choices":[{"index":0,"delta":' + delta + ',"finish_reason":null}]}'
    )

def function_call_item_json(i: Int, name: String, args: String, status: String) -> String:
    """A Responses-API `function_call` output item."""
    return (
        '{"type":"function_call","id":"fc_' + String(i) + '","call_id":"call_' + String(i)
        + '","name":"' + esc(name) + '","arguments":"' + esc(args)
        + '","status":"' + status + '"}'
    )

def function_calls_output_json(calls: List[ToolCall]) -> String:
    var s = String("[")
    for i in range(len(calls)):
        if i > 0:
            s += ","
        s += function_call_item_json(i, calls[i].name, calls[i].arguments, "completed")
    return s + "]"


def output_message_json(content: String, status: String) -> String:
    return (
        '{"type":"message","id":"' + MSG_ID + '","status":"' + status
        + '","role":"assistant","content":[{"type":"output_text","text":"'
        + content + '","annotations":[]}]}'
    )

def output_reasoning_json(reasoning: String, status: String) -> String:
    """A Responses-API `reasoning` output item (`reasoning` pre-escaped)."""
    return (
        '{"type":"reasoning","id":"rs_millfolio","status":"' + status
        + '","summary":[{"type":"summary_text","text":"' + reasoning + '"}]}'
    )

def response_object_raw(model: String, output: String, status: String,
                        n_prompt: Int, n_gen: Int) -> String:
    """Responses-API `response` object with a pre-built `output` array (a list
    of message and/or function_call items)."""
    return (
        '{"id":"' + RESP_ID + '","object":"response","created_at":0,"status":"'
        + status + '","model":"' + model + '","output":' + output
        + ',"usage":{"input_tokens":' + String(n_prompt)
        + ',"output_tokens":' + String(n_gen)
        + ',"total_tokens":' + String(n_prompt + n_gen) + "}}"
    )

def resp_event(type: String, payload: String) -> SseEvent:
    # Named SSE frame: an `event:` line plus a matching `"type"` in the JSON
    # (the Vercel AI SDK switches on the latter). `payload` = fields after type.
    return SseEvent.named(type, '{"type":"' + type + '",' + payload + "}")


# ── UTF-8-safe streaming deltas ──────────────────────────────────────────────


def stream_deltas(mut s: ServerState, ids: List[Int]) raises -> List[String]:
    """Decode `ids` incrementally into JSON-escaped deltas, each ending on a
    UTF-8 char boundary.

    A multibyte char split across tokens is never emitted half-formed.
    (Buffered: all ids are already generated.)"""
    var out = List[String]()
    var prefix = List[Int]()
    var sent = 0
    for i in range(len(ids)):
        prefix.append(ids[i])
        var full = s.tok.decode(prefix)
        var clen = complete_utf8_len(full)
        if clen > sent:
            out.append(json_escape_str(slice_bytes(full, sent, clen)))
            sent = clen
    return out^


# ── the flare Handler: one struct, manual routing on method + path ───────────


@fieldwise_init
struct Api(Handler, Copyable, Movable):
    var st: UnsafePointer[ServerState, MutExternalOrigin]

    def serve(self, req: Request) raises -> Response:
        var path = req.url
        var is_post = req.method == Method.POST

        if path == "/health":
            return ok("millfolio ok")
        if path == "/":
            return ok("millfolio inference server — see /v1/models, POST /v1/chat/completions")
        if path == "/v1/models":
            return ok_json(models_json(self.st[].model_id, self.st[].embed_id))
        if path == "/v1/version":
            return ok_json(version_json(self.st[].model_id))
        if is_post and path == "/v1/completions":
            return self.handle_completions(req)
        if is_post and path == "/v1/chat/completions":
            return self.handle_chat(req)
        if is_post and path == "/v1/responses":
            return self.handle_responses(req)
        if is_post and path == "/v1/embeddings":
            return self.handle_embeddings(req)
        return not_found("no route for " + req.method + " " + path)

    def handle_completions(self, req: Request) raises -> Response:
        """OpenAI /v1/completions, echo+logprobs only — a scoring endpoint for
        perplexity. `prompt` is a token-id array (preferred — no tokenizer mismatch)
        or a raw string (tokenized with the model's tokenizer, NO chat template).
        Runs a teacher-forced forward and returns `logprobs.token_logprobs` =
        [null, log P(t1|t0), log P(t2|t0:1), …]; the client computes
        PPL = exp(-mean token_logprobs). No text is generated (echo-only)."""
        ref s = self.st[]
        var bv = parse_json(req.text())
        var pr = bv.map_get("prompt")
        if not pr:
            return bad_request('{"error":{"message":"completions: missing \\"prompt\\""}}')
        var tokens = List[Int]()
        var pv = pr.value()
        if pv.tag == VLIST:
            for j in range(len(pv.c[].vals)):
                var e = pv.c[].vals[j]
                if e.tag != VINT:
                    return bad_request('{"error":{"message":"completions: prompt array must be token ids"}}')
                tokens.append(e.i)
        elif pv.tag == VSTR:
            tokens = s.tok.encode(to_bytes(pv.s))
            # Gemma requires a leading <bos>. The chat path gets it from the
            # template's literal "<bos>"; raw-text encode() does NOT add one
            # (and must not, or the chat path would double-BOS). Without it
            # Gemma's logprobs are garbage (PPL in the thousands), so prepend
            # it here for raw-string scoring to match how the model is used.
            if s.cfg.tool_style == TOOL_GEMMA:
                var merged = s.tok.encode(to_bytes(String("<bos>")))
                for i in range(len(tokens)):
                    merged.append(tokens[i])
                tokens = merged^
        else:
            return bad_request('{"error":{"message":"completions: prompt must be a string or token-id array"}}')
        if len(tokens) < 2:
            return bad_request('{"error":{"message":"completions: need >= 2 tokens"}}')
        if len(tokens) > s.max_seq:
            return bad_request('{"error":{"message":"completions: window of ' + String(len(tokens))
                + ' exceeds context ' + String(s.max_seq) + '"}}')

        var lps = _token_logprobs(s, tokens)   # T-1 logprobs (for tokens[1..T-1])
        var tok_arr = String("[")
        for i in range(len(tokens)):
            if i > 0:
                tok_arr += ","
            tok_arr += String(tokens[i])
        tok_arr += "]"
        var lp_arr = String("[null")            # token_logprobs[0] = null (no context)
        for i in range(len(lps)):
            lp_arr += "," + String(lps[i])
        lp_arr += "]"
        print("  completions: ", len(tokens), " tok scored", sep="")
        return ok_json('{"id":"cmpl-millfolio","object":"text_completion","model":"'
            + s.model_id + '","choices":[{"text":"","index":0,"logprobs":{"tokens":'
            + tok_arr + ',"token_logprobs":' + lp_arr
            + '},"finish_reason":"length"}],"usage":{"prompt_tokens":'
            + String(len(tokens)) + ',"total_tokens":' + String(len(tokens)) + '}}')

    def handle_embeddings(self, req: Request) raises -> Response:
        """OpenAI /v1/embeddings. Routed to the SECONDARY embedding model
        (`embed_w`/`embed_tok`) when one is loaded; if the PRIMARY model is itself
        the embedding arch (arch==2) we use it directly; otherwise no embedding
        model is available → 503. `input` is a string or an array of strings. Each
        is tokenized with the embedding model's own tokenizer (NO EOS append —
        last-token pooling uses the raw final token) and run through sess_embed
        (last-token-pooled + L2-normalized vector)."""
        ref s = self.st[]
        # Which weights+tokenizer serve embeddings: the secondary embed model if
        # loaded, else the primary iff it is itself arch==2, else none → 503.
        var use_secondary = Bool(s.embed_w)
        if not use_secondary and s.primary_arch != 2:
            return service_unavailable('{"error":{"message":"no embedding model '
                + 'loaded (set EMBED_SAFETENSORS or cache Qwen/Qwen3-Embedding-0.6B, '
                + 'or serve an arch==2 model)"}}')
        var bv = parse_json(req.text())
        var inp = bv.map_get("input")
        if not inp:
            return bad_request('{"error":{"message":"embeddings: missing \\"input\\""}}')
        # Collect the input strings (single string or array of strings).
        var texts = List[String]()
        var iv = inp.value()
        if iv.tag == VSTR:
            texts.append(iv.s.copy())
        elif iv.tag == VLIST:
            for j in range(len(iv.c[].vals)):
                var e = iv.c[].vals[j]
                if e.tag == VSTR:
                    texts.append(e.s.copy())
                else:
                    return bad_request('{"error":{"message":"embeddings: '
                        + '\\"input\\" array must contain strings"}}')
        else:
            return bad_request('{"error":{"message":"embeddings: \\"input\\" '
                + 'must be a string or an array of strings"}}')
        if len(texts) == 0:
            return bad_request('{"error":{"message":"embeddings: empty input"}}')

        # Tokenize every input up-front with the EMBEDDING model's own tokenizer
        # (NO EOS append). Done before the embed loop so we don't bind a `ref` to
        # the model weights across the two cases (secondary vs primary-arch==2),
        # whose origins differ. The embed forward then runs per branch below.
        var id_lists = List[List[Int]]()
        var n_tok = 0
        for i in range(len(texts)):
            var ids: List[Int]
            if use_secondary:
                ids = s.embed_tok.value().encode(to_bytes(texts[i]))
            else:
                ids = s.tok.encode(to_bytes(texts[i]))
            if len(ids) == 0:
                return bad_request('{"error":{"message":"embeddings: input '
                    + String(i) + ' tokenized to zero tokens"}}')
            n_tok += len(ids)
            id_lists.append(ids^)

        var data = String("[")
        for i in range(len(id_lists)):
            var vec: List[Float32]
            if use_secondary:
                vec = sess_embed(s.ctx, s.embed_w.value(), id_lists[i])
            else:
                # Reached only when the PRIMARY is itself an embedding model
                # (arch==2, always Qwen); Gemma never takes this branch.
                vec = sess_embed(s.ctx, s.model[Weights], id_lists[i])
            if i > 0:
                data += ","
            data += embedding_item_json(i, vec^)
        data += "]"
        var emb_id = s.embed_id if use_secondary else s.model_id
        print("  embeddings: ", len(texts), " input(s), ", n_tok, " tokens", sep="")
        return ok_json(embeddings_json(emb_id, data, n_tok))

    def handle_chat(self, req: Request) raises -> Response:
        ref s = self.st[]
        var body = req.text()
        var bv = parse_json(body)
        var ids = s.tok.encode(to_bytes(render_value(s.tmpl, bv, s.cfg.family)))
        var max_new = get_int(bv, "max_tokens", DEF_MAXNEW)
        var temp = Float32(get_float(bv, "temperature", 0.0))
        var top_p = Float32(get_float(bv, "top_p", Float64(DEF_TOPP)))
        var top_k = get_int(bv, "top_k", DEF_TOPK)
        var want_stream = get_bool(bv, "stream", False)

        var r = gen_full(s, ids, max_new, temp, top_k, top_p)
        var fin = String("stop") if r.stopped else String("length")
        print("  chat: ", len(r.ids), " tokens [", fin, "]", sep="")
        var stats = millfolio_stats(r)
        var has_tools = req_has_tools(bv)

        # Gemma: always post-process — split off the thinking channel (surfaced as
        # `reasoning_content`) and lift any <|tool_call> blocks into `tool_calls`.
        if s.cfg.tool_style == TOOL_GEMMA:
            var pr = parse_gemma_tool_calls(bytes_to_string(s.tok.decode(r.ids)))
            var content_e = esc(pr.content)
            var reasoning_e = esc(pr.reasoning)
            var emit_tools = has_tools and pr.has_calls()
            if emit_tools:
                print("    -> ", len(pr.calls), " tool call(s)", sep="")
            if want_stream:
                var ch = SseChannel()
                ch.push(SseEvent.message(chunk_role_json(s.model_id)))
                if pr.reasoning.byte_length() > 0:
                    ch.push(SseEvent.message(chunk_reasoning_json(s.model_id, reasoning_e)))
                if pr.content.byte_length() > 0:
                    ch.push(SseEvent.message(chunk_json(s.model_id, content_e, False, fin)))
                if emit_tools:
                    for i in range(len(pr.calls)):
                        ch.push(SseEvent.message(chunk_toolcall_json(s.model_id, i, pr.calls[i])))
                    ch.push(SseEvent.message(chunk_json(s.model_id, "", True, "tool_calls")))
                else:
                    ch.push(SseEvent.message(chunk_json(s.model_id, "", True, fin, stats)))
                ch.push(SseEvent.message("[DONE]"))
                ch.close()
                return sse_response(ch)
            if emit_tools:
                return ok_json(completion_tools_json(s.model_id, pr.content, pr.calls,
                                                     len(ids), len(r.ids), pr.reasoning))
            return ok_json(completion_json(s.model_id, content_e, len(ids), len(r.ids),
                                           fin, stats, reasoning_e))

        # Qwen: lift <tool_call> blocks only when the request advertised tools (a
        # tools-less request that happens to emit the literal text stays content).
        if has_tools:
            var _completion = bytes_to_string(s.tok.decode(r.ids))
            var tc = parse_tool_calls(_completion)
            if tc.has_calls():
                print("    -> ", len(tc.calls), " tool call(s)", sep="")
                if want_stream:
                    var ch = SseChannel()
                    ch.push(SseEvent.message(chunk_role_json(s.model_id)))
                    if tc.content.byte_length() > 0:
                        ch.push(SseEvent.message(chunk_json(s.model_id, esc(tc.content), False, fin)))
                    for i in range(len(tc.calls)):
                        ch.push(SseEvent.message(chunk_toolcall_json(s.model_id, i, tc.calls[i])))
                    ch.push(SseEvent.message(chunk_json(s.model_id, "", True, "tool_calls")))
                    ch.push(SseEvent.message("[DONE]"))
                    ch.close()
                    return sse_response(ch)
                return ok_json(completion_tools_json(s.model_id, tc.content, tc.calls, len(ids), len(r.ids)))

        if want_stream:
            var ch = SseChannel()
            var deltas = stream_deltas(s, r.ids)
            for i in range(len(deltas)):
                ch.push(SseEvent.message(chunk_json(s.model_id, deltas[i], False, fin)))
            # The final chunk carries the millfolio stats (generation is already
            # done by here, so they're complete).
            ch.push(SseEvent.message(chunk_json(s.model_id, "", True, fin, stats)))
            ch.push(SseEvent.message("[DONE]"))
            ch.close()
            return sse_response(ch)

        var content = json_escape_str(s.tok.decode(r.ids))
        return ok_json(completion_json(s.model_id, content, len(ids), len(r.ids), fin, stats))

    def handle_responses(self, req: Request) raises -> Response:
        ref s = self.st[]
        var body = req.text()
        var bv0 = parse_json(body)
        var chat = responses_to_chat(bv0)
        if not chat:
            return bad_request('{"error":{"message":"responses: need messages or string input"}}')
        var bv = chat.value()
        var ids = s.tok.encode(to_bytes(render_value(s.tmpl, bv, s.cfg.family)))
        # Generation knobs live on the original Responses body, not the
        # synthesized messages Value. (`max_output_tokens` is the Responses
        # spelling; fall back to `max_tokens`.)
        var max_new = get_int(bv0, "max_output_tokens", get_int(bv0, "max_tokens", DEF_MAXNEW))
        var temp = Float32(get_float(bv0, "temperature", 0.0))
        var top_p = Float32(get_float(bv0, "top_p", Float64(DEF_TOPP)))
        var top_k = get_int(bv0, "top_k", DEF_TOPK)
        var want_stream = get_bool(bv0, "stream", False)

        var r = gen_full(s, ids, max_new, temp, top_k, top_p)
        # Gemma: clean the thinking channel out of the message text and surface it
        # as a reasoning item. Qwen: the decoded text is the message verbatim.
        var reasoning_e = String("")
        var full: String
        if s.cfg.tool_style == TOOL_GEMMA:
            var prg = parse_gemma_tool_calls(bytes_to_string(s.tok.decode(r.ids)))
            full = esc(prg.content)
            reasoning_e = esc(prg.reasoning)
        else:
            full = json_escape_str(s.tok.decode(r.ids))
        print("  responses: ", len(r.ids), " tokens", sep="")

        # Tool calls -> Responses `function_call` output items (only if requested).
        if req_has_tools(bv0):
            var _completion = bytes_to_string(s.tok.decode(r.ids))
            var tc = parse_gemma_tool_calls(_completion) if s.cfg.tool_style == TOOL_GEMMA else parse_tool_calls(_completion)
            if tc.has_calls():
                print("    -> ", len(tc.calls), " tool call(s)", sep="")
                var out_arr = function_calls_output_json(tc.calls)
                if not want_stream:
                    return ok_json(response_object_raw(s.model_id, out_arr, "completed", len(ids), len(r.ids)))
                var tch = SseChannel()
                tch.push(resp_event("response.created",
                    '"response":' + response_object_raw(s.model_id, "[]", "in_progress", len(ids), 0)))
                for i in range(len(tc.calls)):
                    var nm = tc.calls[i].name
                    var ar = tc.calls[i].arguments
                    tch.push(resp_event("response.output_item.added",
                        '"output_index":' + String(i) + ',"item":'
                        + function_call_item_json(i, nm, "", "in_progress")))
                    tch.push(resp_event("response.function_call_arguments.delta",
                        '"item_id":"fc_' + String(i) + '","output_index":' + String(i)
                        + ',"delta":"' + esc(ar) + '"'))
                    tch.push(resp_event("response.function_call_arguments.done",
                        '"item_id":"fc_' + String(i) + '","output_index":' + String(i)
                        + ',"arguments":"' + esc(ar) + '"'))
                    tch.push(resp_event("response.output_item.done",
                        '"output_index":' + String(i) + ',"item":'
                        + function_call_item_json(i, nm, ar, "completed")))
                tch.push(resp_event("response.completed",
                    '"response":' + response_object_raw(s.model_id, out_arr, "completed", len(ids), len(r.ids))))
                tch.close()
                return sse_response(tch)

        # Output array (a reasoning item first, when present, then the message).
        var out_arr = String("[")
        if reasoning_e.byte_length() > 0:
            out_arr += output_reasoning_json(reasoning_e, "completed") + ","
        out_arr += output_message_json(full, "completed") + "]"

        if not want_stream:
            return ok_json(response_object_raw(s.model_id, out_arr, "completed", len(ids), len(r.ids)))

        var ch = SseChannel()
        ch.push(resp_event("response.created",
            '"response":' + response_object_raw(s.model_id, "[]", "in_progress", len(ids), 0)))
        var oidx = 0
        if reasoning_e.byte_length() > 0:
            ch.push(resp_event("response.output_item.added",
                '"output_index":' + String(oidx) + ',"item":' + output_reasoning_json("", "in_progress")))
            ch.push(resp_event("response.output_item.done",
                '"output_index":' + String(oidx) + ',"item":' + output_reasoning_json(reasoning_e, "completed")))
            oidx += 1
        ch.push(resp_event("response.output_item.added",
            '"output_index":' + String(oidx) + ',"item":{"type":"message","id":"' + MSG_ID
            + '","status":"in_progress","role":"assistant","content":[]}'))
        ch.push(resp_event("response.content_part.added",
            '"item_id":"' + MSG_ID + '","output_index":' + String(oidx)
            + ',"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}'))
        # Gemma's text is cleaned post-hoc, so it can't be streamed token-incrementally
        # (the raw ids carry channel markers); send it as one delta. Qwen streams live.
        if s.cfg.tool_style == TOOL_GEMMA:
            if full.byte_length() > 0:
                ch.push(resp_event("response.output_text.delta",
                    '"item_id":"' + MSG_ID + '","output_index":' + String(oidx)
                    + ',"content_index":0,"delta":"' + full + '"'))
        else:
            var deltas = stream_deltas(s, r.ids)
            for i in range(len(deltas)):
                ch.push(resp_event("response.output_text.delta",
                    '"item_id":"' + MSG_ID + '","output_index":' + String(oidx)
                    + ',"content_index":0,"delta":"' + deltas[i] + '"'))
        ch.push(resp_event("response.output_text.done",
            '"item_id":"' + MSG_ID + '","output_index":' + String(oidx)
            + ',"content_index":0,"text":"' + full + '"'))
        ch.push(resp_event("response.content_part.done",
            '"item_id":"' + MSG_ID + '","output_index":' + String(oidx)
            + ',"content_index":0,"part":{"type":"output_text","text":"' + full + '","annotations":[]}'))
        ch.push(resp_event("response.output_item.done",
            '"output_index":' + String(oidx) + ',"item":' + output_message_json(full, "completed")))
        ch.push(resp_event("response.completed",
            '"response":' + response_object_raw(s.model_id, out_arr, "completed", len(ids), len(r.ids))))
        ch.close()
        return sse_response(ch)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _detect_family(ckpt_dir: String) raises -> Int:
    """Model family from the checkpoint's config.json `model_type` (FAMILY_GEMMA
    for any gemma* type, else FAMILY_QWEN). Drives tokenizer + chat-template
    selection so a Gemma checkpoint serves with Gemma's SentencePiece-style BPE
    and turn-format template instead of Qwen's byte-level BPE + ChatML."""
    var cfg = ckpt_dir + "/config.json"
    if not exists(cfg):
        return FAMILY_QWEN
    if read_text(cfg).find("gemma") >= 0:   # model_type "gemma4_unified*"
        return FAMILY_GEMMA
    return FAMILY_QWEN


def _dirname(path: String) -> String:
    """Directory component of `path` (everything before the last '/'), or '.'."""
    var b = path.as_bytes()
    var cut = -1
    for i in range(len(b)):
        if b[i] == 47:           # '/'
            cut = i
    if cut < 0:
        return String(".")
    if cut == 0:
        return String("/")
    var out = List[UInt8]()
    for i in range(cut):
        out.append(b[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _slug(model_id: String) -> String:
    """HF repo id -> cache dir suffix: 'Qwen/Qwen2.5-3B-Instruct' -> 'Qwen--Qwen2.5-3B-Instruct'."""
    var b = model_id.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        if b[i] == 47:           # '/'
            out.append(45); out.append(45)   # '--'
        else:
            out.append(b[i])
    return bytes_to_string(out)

def hf_cache_path(model_id: String) raises -> String:
    """Local snapshot dir of an already-downloaded HF model, mirroring
    huggingface_hub's layout: <hub>/models--<slug>/snapshots/<refs/main>. Raises if
    not cached (no refs/main) — caller then treats the arg as a literal path."""
    var home = String(getenv("HF_HOME"))
    var hub = (home + "/hub") if home.byte_length() > 0 else (String(getenv("HOME")) + "/.cache/huggingface/hub")
    var repo = hub + "/models--" + _slug(model_id)
    var commit = String(read_text(repo + "/refs/main")).strip()
    return repo + "/snapshots/" + String(commit)


struct Config(Copyable, Movable):
    """Server config from ~/.config/millfolio/config.json (+ env). All keys optional;
    precedence is: env var > config file > built-in default."""
    var port: Int
    var model: String       # default chat model/checkpoint (when no CLI arg / $QWEN_SAFETENSORS)
    var embed_model: String # default embedding model/checkpoint (when no $EMBED_SAFETENSORS)
    var q4: Bool            # group-128 int4 projection weights
    var kv_budget_mb: Int   # disk KV-cache LRU cap, in MiB

    def __init__(out self, port: Int, var model: String, var embed_model: String,
                 q4: Bool, kv_budget_mb: Int):
        self.port = port
        self.model = model^
        self.embed_model = embed_model^
        self.q4 = q4
        self.kv_budget_mb = kv_budget_mb


def _config_atoi(s: String, default: Int) -> Int:
    var n = 0
    var any = False
    for cp in s.codepoints():
        var v = Int(cp)
        if v < 48 or v > 57:
            return default
        n = n * 10 + (v - 48)
        any = True
    return n if any else default


def load_config() -> Config:
    """Load ~/.config/millfolio/config.json (override path: $MILLFOLIO_CONFIG).
    Recognized keys: port (int), model (str), embed_model (str), q4 (bool),
    kv_budget_mb (int, MiB). `model` is the chat model; `embed_model` is the
    secondary embedding model (HF id or checkpoint path, same treatment as
    `model`). Parsed with the same jinja2.mojo json the server uses for requests."""
    var port = Int(PORT)
    var model = String("")
    var embed_model = String("")
    var q4 = False
    var kv_mb = Int(KV_BUDGET_BYTES) // (1024 * 1024)   # default 8 GiB -> 8192 MiB

    var path = String(getenv("MILLFOLIO_CONFIG"))
    if path.byte_length() == 0:
        path = String(getenv("HOME")) + "/.config/millfolio/config.json"
    try:
        var v = parse_json(read_text(path))
        port = get_int(v, "port", port)
        var m = get_str(v, "model")
        if m.byte_length() > 0:
            model = m^
        var em = get_str(v, "embed_model")
        if em.byte_length() > 0:
            embed_model = em^
        q4 = get_bool(v, "q4", q4)
        kv_mb = get_int(v, "kv_budget_mb", kv_mb)
    except:
        pass  # missing / unreadable / malformed config -> defaults

    # env overrides (where one exists)
    var ep = String(getenv("MILLFOLIO_PORT"))
    if ep.byte_length() > 0:
        port = _config_atoi(ep, port)
    if String(getenv("QWEN_Q4")) == "1":
        q4 = True
    return Config(port, model^, embed_model^, q4, kv_mb)


def main() raises:
    # Config: ~/.config/millfolio/config.json (+ env). Path override: $MILLFOLIO_CONFIG.
    var cfg = load_config()
    # Checkpoint selection: `serve <hf-id-or-path>` (CLI) > $QWEN_SAFETENSORS >
    # config `model` > meta.txt. An HF id resolves to its cached snapshot dir; the
    # served model id (reported by /v1/models) is that id, else derived from the arch.
    var ckpt: String
    var model_id = String("")
    if len(argv()) > 1:
        var spec = String(argv()[1])
        try:
            ckpt = hf_cache_path(spec)
            model_id = spec
            print("model: ", spec, sep="")
        except:
            ckpt = spec   # not in the HF cache — use as a literal checkpoint path
            print("model: ", spec, " (path)", sep="")
    else:
        var env = String(getenv("QWEN_SAFETENSORS"))
        if env.byte_length() > 0:
            ckpt = env
        elif cfg.model.byte_length() > 0:
            try:
                ckpt = hf_cache_path(cfg.model)
                model_id = cfg.model.copy()
                print("model: ", cfg.model, " (config)", sep="")
            except:
                ckpt = cfg.model.copy()
                print("model: ", cfg.model, " (config path)", sep="")
        else:
            ckpt = String(String(read_text("tests/fixtures/forward/meta.txt").split("\n")[1]).strip())

    # Optional group-128 int4 weights (QWEN_Q4=1). Projection weights become int4
    # (embed/lm-head stays bf16); ~4x smaller + ~2x faster decode, at a quality
    # cost that is coherent on the 3B but degrades the 0.5B (see model.QMat).
    var q4 = cfg.q4   # config `q4` (with $QWEN_Q4 override), applied in load_config()
    print("loading tokenizer + weights…")
    # Prefer the checkpoint's own HuggingFace tokenizer.json (what the native
    # downloader fetches) so a freshly downloaded model serves with no tok-capture;
    # fall back to the tok-capture .tsv fixtures (dev/tests). ckpt is the snapshot
    # dir (sharded/HF cache) or a single .safetensors path — look beside either.
    var ckpt_dir = ckpt if isdir(ckpt) else _dirname(ckpt)
    # Family (from config.json model_type) selects both the tokenizer flavour and
    # the chat template: Gemma uses SentencePiece-style BPE + the turn-format
    # template; Qwen uses byte-level BPE + ChatML. Defaults to Qwen so the
    # existing path is unchanged when no/Qwen config is present.
    var family = _detect_family(ckpt_dir)
    var tok_json = ckpt_dir + "/tokenizer.json"
    var tok: Tokenizer
    if exists(tok_json):
        print("  tokenizer: ", tok_json, " (", "gemma" if family == FAMILY_GEMMA else "qwen", ")", sep="")
        if family == FAMILY_GEMMA:
            tok = load_gemma_tokenizer_json(tok_json)
        else:
            tok = load_tokenizer_json(tok_json)
    else:
        tok = load_tokenizer("tests/fixtures/tokenizer/")
    # The Qwen template drives render_value for Qwen; Gemma renders in pure Mojo
    # (render_gemma) and ignores tmpl, but ServerState still needs a Template, so
    # this is loaded as a placeholder for the Gemma path.
    var tmpl = load_chat_template(TEMPLATE)
    var ctx = DeviceContext()

    # Primary (chat) model, loaded by family into a Variant (both weight structs
    # conform to ModelWeights). The rest of the server reads `p_cfg` for behavior +
    # dispatches once on the Variant's active type — no per-family `if`s. Capture the
    # scalars main()/the banner need before the weights move into the Variant.
    var p_nlayers: Int
    var p_nkv: Int
    var p_arch: Int
    var p_quant: Bool
    var p_simd_ok: Bool
    var p_maxseq: Int
    var p_cfg: ModelConfig
    var model: Variant[Weights, GemmaWeights]
    var gemm_path = String("simdgroup-matrix (~4.5x)")
    if family == FAMILY_GEMMA:
        # The 12B bf16 model is ~24 GB and won't fit a 24 GB GPU, so int4 is forced.
        var alllayers = List[Int]()
        for i in range(G_NLAYERS):
            alllayers.append(i)
        var gw = load_gemma_weights(ctx, ckpt, alllayers, True)
        gw.simd_ok = probe_simd_gemm(ctx)
        p_cfg = gw.config()
        p_nlayers = p_cfg.nlayers; p_nkv = p_cfg.nkv
        p_arch = -1; p_quant = True; p_simd_ok = gw.simd_ok; p_maxseq = GEMMA_MAX_SEQ
        if not gw.simd_ok:
            gemm_path = String("scalar tiled (simd probe failed)")
        if model_id.byte_length() == 0:
            model_id = String(MODEL_GEMMA) + "-int4"
        print("  serving ", model_id, "  (hidden=", gw.hidden, ", layers=", gw.nlayers,
              ", q-heads=", gw.hq, ", kv=8/1, head_dim=256/512, max_seq=", p_maxseq, ")", sep="")
        model = gw^
    else:
        var w = load_weights(ctx, ckpt, q4)
        # Probe the simdgroup-matrix GEMM once; on success prefill GEMMs take the
        # ~4.5× faster path, else fall back to the scalar tiled kernel (see mm()).
        w.simd_ok = probe_simd_gemm(ctx)
        p_cfg = w.config()
        p_nlayers = p_cfg.nlayers; p_nkv = p_cfg.nkv
        p_arch = w.arch; p_quant = w.quant; p_simd_ok = w.simd_ok; p_maxseq = MAX_SEQ
        # The KV cache is f32 and sized by max_seq regardless of weight quant. For
        # the large Qwen3 chat models (arch 3=8B/4=14B, nkv=1024) a full MAX_SEQ KV
        # is ~9.6-10.7 GB; combined with weights this can over-commit the 24 GB GPU
        # and silently corrupt the KV. Cap to fit:
        #   - 8B int4 (~5 GB weights): fits MAX_SEQ -> no cap.
        #   - 8B bf16 (~16 GB weights): over-commits -> cap.
        #   - 14B (~11 GB int4 weights + ~11 GB KV): over-commits even int4 -> cap.
        if w.arch == 4 or (w.arch == 3 and not w.quant):
            p_maxseq = 16384
            print("  note: capping max_seq to ", p_maxseq,
                  " so the KV cache fits GPU memory (large model; 8B-int4 keeps ", MAX_SEQ, ")", sep="")
        if not w.simd_ok:
            gemm_path = String("scalar tiled (simd probe failed)")
        if model_id.byte_length() == 0:   # default id from detected arch (+quant tag)
            model_id = String(MODEL_3B) if w.arch == 1 else String(MODEL_05B)
            if w.quant:
                model_id += "-int4"       # distinct id + KV-cache dir from the bf16 build
        print("  serving ", model_id, "  (hidden=", w.hidden, ", layers=", w.nlayers,
              ", heads=", w.hq, "/", w.hkv, ", head_dim=", w.head_dim, ")", sep="")
        model = w^
    print("  prefill GEMM: ", gemm_path, sep="")
    var wprec = String("group-128 int4 (proj) + bf16 (embed)") if p_quant else String("bf16")
    print("  weights: ", wprec, sep="")

    # One persistent KV cache for the process, sized to the model.
    var sess = new_session(ctx, p_maxseq, p_nlayers, p_nkv)

    # Disk-backed prefix cache (per model), survives restarts.
    var kvdir = String(getenv("HOME")) + "/.cache/millfolio/kv/" + _slug(model_id)
    var bcache = BlockCache(kvdir, BLOCK_TOK, p_nkv, p_nlayers, cfg.kv_budget_mb * 1024 * 1024, model_id)  # MiB -> bytes
    if bcache.enabled:
        print("  kv-cache: ", kvdir, " (", len(bcache.order), " blocks cached, cap ",
              bcache.max_blocks, " blocks)", sep="")
    else:
        print("  kv-cache: disabled")

    # SECONDARY embedding model so one process/port serves /v1/embeddings too.
    # Checkpoint precedence (mirrors the chat model's id->snapshot resolution):
    #   $EMBED_SAFETENSORS (path) > config `embed_model` (HF id or path) >
    #   default Qwen/Qwen3-Embedding-0.6B from the HF cache. If none resolves the
    #   field stays unset and /v1/embeddings 503s (chat still works). Skipped when
    #   the PRIMARY is itself the embedding arch (arch==2) — it serves embeddings.
    var embed_w = Optional[Weights](None)
    var embed_tok = Optional[Tokenizer](None)
    var embed_id = String("")
    if p_arch == 2:
        print("  embeddings: served by the primary model (arch==2)")
    else:
        var eckpt = String("")
        var eid = String("")
        var esrc = String("")
        var eenv = String(getenv("EMBED_SAFETENSORS"))
        if eenv.byte_length() > 0:
            eckpt = eenv; esrc = "EMBED_SAFETENSORS"   # literal path
        elif cfg.embed_model.byte_length() > 0:
            try:
                eckpt = hf_cache_path(cfg.embed_model); eid = cfg.embed_model.copy()
                esrc = "config embed_model"
            except:
                eckpt = cfg.embed_model.copy(); eid = cfg.embed_model.copy()
                esrc = "config embed_model (path)"
        else:
            try:
                eckpt = hf_cache_path(MODEL_EMBED); eid = String(MODEL_EMBED)
                esrc = "default (HF cache)"
            except:
                pass  # not cached either -> embed model stays unset
        if eckpt.byte_length() > 0:
            try:
                var edir = eckpt if isdir(eckpt) else _dirname(eckpt)
                var etok_json = edir + "/tokenizer.json"
                var et: Tokenizer
                if exists(etok_json):
                    et = load_tokenizer_json(etok_json)
                else:
                    et = load_tokenizer("tests/fixtures/tokenizer/")
                var ew = load_weights(ctx, eckpt, False)   # embed weights stay bf16
                ew.simd_ok = p_simd_ok
                if eid.byte_length() == 0:
                    eid = String(MODEL_EMBED)
                embed_id = eid.copy()
                print("  embed model: ", embed_id, "  (dims=", ew.hidden,
                      ", layers=", ew.nlayers, ", arch=", ew.arch, ", ", esrc,
                      ")", sep="")
                embed_tok = Optional[Tokenizer](et^)
                embed_w = Optional[Weights](ew^)
            except:
                print("  embed model: failed to load ", eckpt, " (", esrc,
                      ") — /v1/embeddings will 503", sep="")
        else:
            print("  embed model: none resolved — /v1/embeddings will 503")

    # Capture the banner values before the (moving) ServerState construction.
    var banner_embed_id = embed_id.copy()
    var banner_model_id = model_id.copy()
    var primary_is_embed = p_arch == 2

    var state = ServerState(ctx^, model^, p_cfg, p_arch,
                            p_maxseq, tok^, tmpl^, sess^, model_id^, bcache^,
                            embed_w^, embed_tok^, embed_id^)
    var sp = alloc[ServerState](1)
    sp.init_pointee_move(state^)
    var api = Api(sp)

    print("millfolio serving on http://127.0.0.1:", cfg.port, "  (flare)  v", MILLFOLIO_VERSION, sep="")
    print("  GET  /v1/models")
    print("  GET  /v1/version")
    print("  POST /v1/chat/completions  (stream + non-stream)")
    print("  POST /v1/responses         (stream + non-stream)")
    if banner_embed_id.byte_length() > 0:
        print("  POST /v1/embeddings        (", banner_embed_id, ")", sep="")
    elif primary_is_embed:
        print("  POST /v1/embeddings        (", banner_model_id, ")", sep="")
    else:
        print("  POST /v1/embeddings        (no embed model — 503)")
    var srv = HttpServer.bind(SocketAddr.localhost(UInt16(cfg.port)))
    srv.serve(api^)
