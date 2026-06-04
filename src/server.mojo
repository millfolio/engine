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

from flare.prelude import *
from flare.http import Handler, SseChannel, SseEvent, sse_response

from model import (
    Weights, load_weights, probe_simd_gemm, EOS1, EOS2,
    Session, new_session, sess_prefill_suffix, sess_step,
    argmax_f, process_logits, sample,
)
from tokenizer import Tokenizer, load_tokenizer
from chat import load_chat_template, render_value, json_escape_str
from toolcall import parse_tool_calls, ToolCall
from blockcache import BlockCache
from template import Template
from value import Value
from json import parse_json, bytes_to_string

# Persistent KV-cache capacity (tokens). One Session of this size lives on
# ServerState for the whole process so successive requests in an agent loop reuse
# the prefix they share instead of re-prefilling it. 32768 = Qwen2.5's native
# context; the cache is ~MAX_SEQ * 24 KiB ≈ 805 MB resident on the GPU.
comptime MAX_SEQ = 32768

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
comptime PORT = 8000

# minja2 Value tags (value.mojo)
comptime VBOOL = 2
comptime VINT = 3
comptime VFLOAT = 4
comptime VSTR = 5
# sampling defaults (generation_config.json) when temperature > 0
comptime DEF_TOPK = 20
comptime DEF_TOPP = Float32(0.8)
comptime DEF_REP = Float32(1.1)
comptime DEF_MAXNEW = 256
comptime SEED = UInt64(0x9E3779B97F4A7C15)

# Responses-API ids (opencode / Vercel AI SDK)
comptime RESP_ID = "resp_millrace"
comptime MSG_ID = "msg_millrace"


# ── Shared model state ───────────────────────────────────────────────────────


struct ServerState(Movable):
    """The one model, loaded once and reached by the (borrowed-self) handler
    through a pointer so generation can still take `mut w`."""

    var ctx: DeviceContext
    var w: Weights
    var tok: Tokenizer
    var tmpl: Template
    var sess: Session      # one long-lived KV cache, reused across requests
    var cached: List[Int]  # token ids currently held in sess rows [0, len)
    var model_id: String   # id reported by /v1/models + every response
    var bcache: BlockCache # disk-backed prefix cache (survives restarts)

    def __init__(out self, var ctx: DeviceContext, var w: Weights,
                 var tok: Tokenizer, var tmpl: Template, var sess: Session,
                 var model_id: String, var bcache: BlockCache):
        self.ctx = ctx^
        self.w = w^
        self.tok = tok^
        self.tmpl = tmpl^
        self.sess = sess^
        self.cached = List[Int]()
        self.model_id = model_id^
        self.bcache = bcache^


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
    rather than constructing minja2 Values by hand."""
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

    def __init__(out self, var ids: List[Int], stopped: Bool):
        self.ids = ids^
        self.stopped = stopped


def gen_full(mut s: ServerState, ids: List[Int], max_new: Int,
             temp: Float32, top_k: Int, top_p: Float32) raises -> Reply:
    """Run the GPU decode loop to completion for `ids`, honoring OpenAI knobs.

    Reuses the longest prefix already resident in the persistent KV cache (the
    common case in an agent loop, where each turn appends to the same growing
    conversation) and only prefills the diverging suffix. Times prefill vs decode
    separately — each `sess_*` call ends in a device→host logits copy, so the GPU
    is synced at the boundary — and logs a terse per-request line."""
    # Clamp generation so prefill + decode never overrun the cache.
    var room = MAX_SEQ - len(ids) - 1
    if room < 1:
        raise Error("prompt of " + String(len(ids)) + " tokens exceeds context "
                    + String(MAX_SEQ))
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
    var logits = sess_prefill_suffix(s.ctx, s.w, s.sess, suffix, reuse, True)
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
    while len(gen) < cap:
        var nxt = (
            sample(process_logits(logits, context, temp, top_k, top_p, DEF_REP), rng)
            if temp > 0.0 else argmax_f(logits)
        )
        if nxt == EOS1 or nxt == EOS2:
            stopped = True
            break
        gen.append(nxt)
        context.append(nxt)
        if len(gen) >= cap:
            break
        logits = sess_step(s.ctx, s.w, s.sess, nxt)
        # sess_step already synced (host logits copy), so this is real wall-clock.
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
    return Reply(gen^, stopped)


# ── JSON envelopes ───────────────────────────────────────────────────────────


def models_json(model: String) -> String:
    return (
        '{"object":"list","data":[{"id":"' + model
        + '","object":"model","created":0,"owned_by":"millrace"}]}'
    )

def completion_json(model: String, content: String, n_prompt: Int, n_gen: Int, finish: String) -> String:
    return (
        '{"id":"chatcmpl-millrace","object":"chat.completion","created":0,"model":"'
        + model + '","choices":[{"index":0,"message":{"role":"assistant","content":"'
        + content + '"},"finish_reason":"' + finish + '"}],'
        + '"usage":{"prompt_tokens":' + String(n_prompt)
        + ',"completion_tokens":' + String(n_gen)
        + ',"total_tokens":' + String(n_prompt + n_gen) + "}}"
    )

def chunk_json(model: String, delta: String, finish: Bool, fin: String) -> String:
    var delta_obj = String("{}")
    var finish_reason = String("null")
    if finish:
        finish_reason = '"' + fin + '"'
    else:
        delta_obj = '{"content":"' + delta + '"}'
    return (
        '{"id":"chatcmpl-millrace","object":"chat.completion.chunk","created":0,"model":"'
        + model + '","choices":[{"index":0,"delta":' + delta_obj
        + ',"finish_reason":' + finish_reason + "}]}"
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
                          n_prompt: Int, n_gen: Int) -> String:
    var content_field = String("null")
    if content.byte_length() > 0:
        content_field = '"' + esc(content) + '"'
    return (
        '{"id":"chatcmpl-millrace","object":"chat.completion","created":0,"model":"'
        + model + '","choices":[{"index":0,"message":{"role":"assistant","content":'
        + content_field + ',"tool_calls":' + tool_calls_array_json(calls)
        + '},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":' + String(n_prompt)
        + ',"completion_tokens":' + String(n_gen)
        + ',"total_tokens":' + String(n_prompt + n_gen) + "}}"
    )

def chunk_role_json(model: String) -> String:
    """Opening streaming chunk announcing the assistant role (content null)."""
    return (
        '{"id":"chatcmpl-millrace","object":"chat.completion.chunk","created":0,"model":"'
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
        '{"id":"chatcmpl-millrace","object":"chat.completion.chunk","created":0,"model":"'
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

def response_object_json(model: String, content: String, status: String, with_output: Bool,
                         n_prompt: Int, n_gen: Int) -> String:
    var output = String("[]")
    if with_output:
        output = "[" + output_message_json(content, "completed") + "]"
    return response_object_raw(model, output, status, n_prompt, n_gen)

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

        if path == "/" or path == "/health":
            return ok("millrace ok")
        if path == "/v1/models":
            return ok_json(models_json(self.st[].model_id))
        if is_post and path == "/v1/chat/completions":
            return self.handle_chat(req)
        if is_post and path == "/v1/responses":
            return self.handle_responses(req)
        return not_found("no route for " + req.method + " " + path)

    def handle_chat(self, req: Request) raises -> Response:
        ref s = self.st[]
        var body = req.text()
        var bv = parse_json(body)
        var ids = s.tok.encode(to_bytes(render_value(s.tmpl, bv)))
        var max_new = get_int(bv, "max_tokens", DEF_MAXNEW)
        var temp = Float32(get_float(bv, "temperature", 0.0))
        var top_p = Float32(get_float(bv, "top_p", Float64(DEF_TOPP)))
        var top_k = get_int(bv, "top_k", DEF_TOPK)
        var want_stream = get_bool(bv, "stream", False)

        var r = gen_full(s, ids, max_new, temp, top_k, top_p)
        var fin = String("stop") if r.stopped else String("length")
        print("  chat: ", len(r.ids), " tokens [", fin, "]", sep="")

        # Tool calls: only when the request advertised tools. Lift the model's
        # <tool_call> blocks into OpenAI `tool_calls` rather than leaking the XML.
        if req_has_tools(bv):
            var tc = parse_tool_calls(bytes_to_string(s.tok.decode(r.ids)))
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
            ch.push(SseEvent.message(chunk_json(s.model_id, "", True, fin)))
            ch.push(SseEvent.message("[DONE]"))
            ch.close()
            return sse_response(ch)

        var content = json_escape_str(s.tok.decode(r.ids))
        return ok_json(completion_json(s.model_id, content, len(ids), len(r.ids), fin))

    def handle_responses(self, req: Request) raises -> Response:
        ref s = self.st[]
        var body = req.text()
        var bv0 = parse_json(body)
        var chat = responses_to_chat(bv0)
        if not chat:
            return bad_request('{"error":{"message":"responses: need messages or string input"}}')
        var bv = chat.value()
        var ids = s.tok.encode(to_bytes(render_value(s.tmpl, bv)))
        # Generation knobs live on the original Responses body, not the
        # synthesized messages Value. (`max_output_tokens` is the Responses
        # spelling; fall back to `max_tokens`.)
        var max_new = get_int(bv0, "max_output_tokens", get_int(bv0, "max_tokens", DEF_MAXNEW))
        var temp = Float32(get_float(bv0, "temperature", 0.0))
        var top_p = Float32(get_float(bv0, "top_p", Float64(DEF_TOPP)))
        var top_k = get_int(bv0, "top_k", DEF_TOPK)
        var want_stream = get_bool(bv0, "stream", False)

        var r = gen_full(s, ids, max_new, temp, top_k, top_p)
        var full = json_escape_str(s.tok.decode(r.ids))
        print("  responses: ", len(r.ids), " tokens", sep="")

        # Tool calls -> Responses `function_call` output items (only if requested).
        if req_has_tools(bv0):
            var tc = parse_tool_calls(bytes_to_string(s.tok.decode(r.ids)))
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

        if not want_stream:
            return ok_json(response_object_json(s.model_id, full, "completed", True, len(ids), len(r.ids)))

        var ch = SseChannel()
        ch.push(resp_event("response.created",
            '"response":' + response_object_json(s.model_id, "", "in_progress", False, len(ids), 0)))
        ch.push(resp_event("response.output_item.added",
            '"output_index":0,"item":{"type":"message","id":"' + MSG_ID
            + '","status":"in_progress","role":"assistant","content":[]}'))
        ch.push(resp_event("response.content_part.added",
            '"item_id":"' + MSG_ID
            + '","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}'))
        var deltas = stream_deltas(s, r.ids)
        for i in range(len(deltas)):
            ch.push(resp_event("response.output_text.delta",
                '"item_id":"' + MSG_ID
                + '","output_index":0,"content_index":0,"delta":"' + deltas[i] + '"'))
        ch.push(resp_event("response.output_text.done",
            '"item_id":"' + MSG_ID + '","output_index":0,"content_index":0,"text":"' + full + '"'))
        ch.push(resp_event("response.content_part.done",
            '"item_id":"' + MSG_ID
            + '","output_index":0,"content_index":0,"part":{"type":"output_text","text":"'
            + full + '","annotations":[]}'))
        ch.push(resp_event("response.output_item.done",
            '"output_index":0,"item":' + output_message_json(full, "completed")))
        ch.push(resp_event("response.completed",
            '"response":' + response_object_json(s.model_id, full, "completed", True, len(ids), len(r.ids))))
        ch.close()
        return sse_response(ch)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


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


def main() raises:
    # Checkpoint selection: `serve <hf-id-or-path>` (CLI) > $QWEN_SAFETENSORS > meta.txt.
    # An HF id resolves to its cached snapshot dir (weights assumed downloaded); the
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
        else:
            ckpt = String(String(read_text("tests/fixtures/forward/meta.txt").split("\n")[1]).strip())

    print("loading tokenizer + weights…")
    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ctx = DeviceContext()
    var w = load_weights(ctx, ckpt)
    # Probe the simdgroup-matrix GEMM once; on success prefill GEMMs take the
    # ~4.5× faster path, else fall back to the scalar tiled kernel (see mm()).
    w.simd_ok = probe_simd_gemm(ctx)
    if model_id.byte_length() == 0:   # default id from detected arch
        model_id = String(MODEL_3B) if w.arch == 1 else String(MODEL_05B)
    # One persistent KV cache for the process, sized to the detected arch.
    var sess = new_session(ctx, MAX_SEQ, w.nlayers, w.nkv)
    print("  serving ", model_id, "  (hidden=", w.hidden, ", layers=", w.nlayers,
          ", heads=", w.hq, "/", w.hkv, ", head_dim=", w.head_dim, ")", sep="")
    var gemm_path = String("simdgroup-matrix (~4.5x)")
    if not w.simd_ok:
        gemm_path = String("scalar tiled (simd probe failed)")
    print("  prefill GEMM: ", gemm_path, sep="")

    # Disk-backed prefix cache (per model), survives restarts.
    var kvdir = String(getenv("HOME")) + "/.cache/millrace/kv/" + _slug(model_id)
    var bcache = BlockCache(kvdir, BLOCK_TOK, w.nkv, w.nlayers, KV_BUDGET_BYTES, model_id)
    if bcache.enabled:
        print("  kv-cache: ", kvdir, " (", len(bcache.order), " blocks cached, cap ",
              bcache.max_blocks, " blocks)", sep="")
    else:
        print("  kv-cache: disabled")

    var state = ServerState(ctx^, w^, tok^, tmpl^, sess^, model_id^, bcache^)
    var sp = alloc[ServerState](1)
    sp.init_pointee_move(state^)
    var api = Api(sp)

    print("millrace serving on http://127.0.0.1:", PORT, "  (flare)", sep="")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions  (stream + non-stream)")
    print("  POST /v1/responses         (stream + non-stream)")
    var srv = HttpServer.bind(SocketAddr.localhost(UInt16(PORT)))
    srv.serve(api^)
