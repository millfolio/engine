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

from model import Weights, load_weights, generate, EOS1, EOS2
from tokenizer import Tokenizer, load_tokenizer

comptime PORT_HI = 0x1F        # 8000 = 0x1F40, big-endian
comptime PORT_LO = 0x40
comptime MAX_NEW = 128
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

def ascii_str(bytes: List[UInt8]) -> String:
    var s = String("")
    for i in range(len(bytes)):
        s += chr(Int(bytes[i]))
    return s^

def chat_prompt(user: String) -> String:
    return String(
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
    ) + user + "<|im_end|>\n<|im_start|>assistant\n"


def json_escape(s: String) -> String:
    var out = String("")
    var sb = s.as_bytes()
    for i in range(len(sb)):
        var c = Int(sb[i])
        if c == 34:
            out += "\\\""
        elif c == 92:
            out += "\\\\"
        elif c == 10:
            out += "\\n"
        elif c == 13:
            out += "\\r"
        elif c == 9:
            out += "\\t"
        else:
            out += chr(c)
    return out^


def last_content(req: String) -> String:
    """Crude extraction of the final "content":"…" string value in the body."""
    var key = String('"content"')
    var rb = req.as_bytes()
    var kb = key.as_bytes()
    # find last occurrence of the key
    var at = -1
    for i in range(len(rb) - len(kb) + 1):
        var hit = True
        for j in range(len(kb)):
            if rb[i + j] != kb[j]:
                hit = False
                break
        if hit:
            at = i
    if at < 0:
        return String("")
    # advance to the opening quote of the value
    var p = at + len(kb)
    while p < len(rb) and Int(rb[p]) != 34:  # skip to '"'
        p += 1
    p += 1
    # read until unescaped '"'
    var out = String("")
    while p < len(rb):
        var c = Int(rb[p])
        if c == 92 and p + 1 < len(rb):  # backslash escape
            var n = Int(rb[p + 1])
            if n == 110:
                out += "\n"
            elif n == 116:
                out += "\t"
            elif n == 34:
                out += "\""
            elif n == 92:
                out += "\\"
            else:
                out += chr(n)
            p += 2
            continue
        if c == 34:
            break
        out += chr(c)
        p += 1
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
            var user = last_content(req)
            print("  prompt: ", user, sep="")
            var ids = tok.encode(to_bytes(chat_prompt(user)))
            var gen = generate(ctx, w, ids, MAX_NEW)
            var body_ids = List[Int]()
            for i in range(len(gen)):
                if gen[i] == EOS1 or gen[i] == EOS2:
                    break
                body_ids.append(gen[i])
            var text = ascii_str(tok.decode(body_ids))
            print("  reply:  ", text, sep="")
            var json = String('{"id":"chatcmpl-millrace","object":"chat.completion","model":"qwen2.5-0.5b-instruct",')
            json += '"choices":[{"index":0,"message":{"role":"assistant","content":"'
            json += json_escape(text)
            json += '"},"finish_reason":"stop"}]}'
            http_response(conn, json)
        _ = external_call["close", c_int](conn)
