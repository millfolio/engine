"""Minimal OpenAI-compatible HTTP server, pure Mojo on the GPU (ARCHITECTURE.md §6).

flare (max-backend's HTTP layer) pins Mojo 1.0.0b1, but this engine needs the
1.0.0b2 nightly's std.gpu API — an unresolved version conflict (§11 #11). And the
Mojo stdlib has no sockets. So this server talks to libc directly via FFI: a
single-threaded blocking accept loop that loads the model once and answers
`POST /v1/chat/completions` and `GET /v1/models`.

Scope: minimal. One request at a time (no streaming/SSE, no concurrency — see
max-backend §10 #4 for why even flare stays single-worker here). Request parsing
is a crude last-`"content"` extraction, not a full JSON parser; the response is a
non-streaming ChatCompletion. Enough to point a client at and get real text.

    pixi run serve            # listens on 127.0.0.1:8000
    curl -s localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
"""

from std.ffi import external_call, c_int
from std.gpu.host import DeviceContext

from model import Weights, load_weights, generate, generate_sample, EOS1, EOS2
from tokenizer import Tokenizer, load_tokenizer
from chat import load_chat_template, render_value, json_escape_str
from value import Value
from json import parse_json, bytes_to_string

comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"
# minja2 Value tags (value.mojo)
comptime VBOOL = 2
comptime VINT = 3
comptime VFLOAT = 4
# sampling defaults (generation_config.json) when temperature > 0
comptime DEF_TOPK = 20
comptime DEF_TOPP = Float32(0.8)
comptime DEF_REP = Float32(1.1)
comptime DEF_MAXNEW = 256

comptime PORT_HI = 0x1F        # 8000 = 0x1F40, big-endian
comptime PORT_LO = 0x40
comptime SOL_SOCKET = 0xFFFF   # macOS
comptime SO_REUSEADDR = 0x0004


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

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


def http_body(req: String) -> String:
    """The bytes after the blank line separating HTTP headers from the body."""
    var idx = req.find("\r\n\r\n")
    if idx < 0:
        return String("")
    var rb = req.as_bytes()
    var out = String("")
    for i in range(idx + 4, len(rb)):
        out += chr(Int(rb[i]))
    return out^


def send_str(conn: c_int, s: String):
    var b = s.as_bytes()
    _ = external_call["send", Int](conn, b.unsafe_ptr(), len(b), c_int(0))

def http_response(conn: c_int, body: String):
    var resp = String("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
    resp += "Content-Length: " + String(len(body.as_bytes())) + "\r\nConnection: close\r\n\r\n" + body
    send_str(conn, resp)


def main() raises:
    var ckpt = String(read_text("tests/fixtures/forward/meta.txt").split("\n")[1]).strip()
    print("loading tokenizer + weights…")
    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ctx = DeviceContext()
    var w = load_weights(ctx, String(ckpt))

    var fd = external_call["socket", c_int](c_int(2), c_int(1), c_int(0))
    var one = List[Int32](length=1, fill=1)
    _ = external_call["setsockopt", c_int](fd, c_int(SOL_SOCKET), c_int(SO_REUSEADDR), one.unsafe_ptr().bitcast[UInt8](), c_int(4))
    var sa = List[UInt8](length=16, fill=0)
    sa[0] = 2
    sa[2] = PORT_HI
    sa[3] = PORT_LO
    sa[4] = 127
    sa[7] = 1
    if Int(external_call["bind", c_int](fd, sa.unsafe_ptr(), c_int(16))) < 0:
        raise Error("bind failed (port 8000 in use?)")
    _ = external_call["listen", c_int](fd, c_int(16))
    print("millrace serving on http://127.0.0.1:8000  (POST /v1/chat/completions)")

    while True:
        var peer = List[UInt8](length=16, fill=0)
        var plen = List[Int32](length=1, fill=16)
        var conn = external_call["accept", c_int](fd, peer.unsafe_ptr(), plen.unsafe_ptr().bitcast[UInt8]())
        if Int(conn) < 0:
            continue
        var buf = List[UInt8](length=65536, fill=0)
        var n = external_call["recv", Int](conn, buf.unsafe_ptr(), 65536, c_int(0))
        var req = String("")
        for i in range(Int(n)):
            req += chr(Int(buf[i]))

        if req.find("/v1/models") >= 0 and req.find("GET") >= 0:
            http_response(conn, String('{"object":"list","data":[{"id":"qwen2.5-0.5b-instruct","object":"model","owned_by":"millrace"}]}'))
        else:
            try:
                var body_v = parse_json(http_body(req))
                var ids = tok.encode(to_bytes(render_value(tmpl, body_v)))

                # OpenAI request knobs: greedy unless temperature > 0.
                var max_new = get_int(body_v, "max_tokens", DEF_MAXNEW)
                var temp = Float32(get_float(body_v, "temperature", 0.0))
                var gen: List[Int]
                if temp > 0.0:
                    var top_p = Float32(get_float(body_v, "top_p", Float64(DEF_TOPP)))
                    var top_k = get_int(body_v, "top_k", DEF_TOPK)
                    gen = generate_sample(ctx, w, ids, max_new, temp, top_k, top_p, DEF_REP, UInt64(0))
                else:
                    gen = generate(ctx, w, ids, max_new)

                var body_ids = List[Int]()
                var stopped = False
                for i in range(len(gen)):
                    if gen[i] == EOS1 or gen[i] == EOS2:
                        stopped = True
                        break
                    body_ids.append(gen[i])
                var dec = tok.decode(body_ids)
                print("  reply:  ", bytes_to_string(dec), sep="")
                var finish = String("stop") if stopped else String("length")
                var json = String('{"id":"chatcmpl-millrace","object":"chat.completion","model":"qwen2.5-0.5b-instruct",')
                json += '"choices":[{"index":0,"message":{"role":"assistant","content":"'
                json += json_escape_str(dec)
                json += '"},"finish_reason":"' + finish + '"}],'
                json += '"usage":{"prompt_tokens":' + String(len(ids))
                json += ',"completion_tokens":' + String(len(body_ids))
                json += ',"total_tokens":' + String(len(ids) + len(body_ids)) + "}}"
                http_response(conn, json)
            except e:
                print("  error: ", String(e), sep="")
                http_response(conn, String('{"error":{"message":"') + json_escape_str(to_bytes(String(e))) + '"}}')
        _ = external_call["close", c_int](conn)
