"""Parse Qwen2.5 `<tool_call>` blocks out of generated text into structured calls.

The chat template (assets/qwen2.5-chat-template.jinja) instructs the model to
emit, for each function call:

    <tool_call>
    {"name": <fn-name>, "arguments": <args-json-object>}
    </tool_call>

This module lifts those blocks out of the raw decoded completion so the server
can frame them as OpenAI `tool_calls` (chat) / `function_call` items (responses)
instead of leaking the literal XML to the client. Any text *outside* the blocks
is returned as the message `content`; a block that isn't valid JSON (or lacks a
`name`) is left verbatim in the content so nothing the model said is dropped.

Pure CPU + JSON only (no GPU) — unit-tested via `pixi run test-toolcall`.
"""

from json import parse_json, to_json, string_to_bytes, bytes_to_string
from value import Value, VSTR

# Gemma 4 generated tool-call markers (single control tokens; decode to this
# literal text). Args use Gemma's serialization (NOT JSON): bare keys, strings
# quoted with the <|"|> token, bare numbers/true/false, nested {}/[].
comptime _G_QUOTE = '<|"|>'
comptime _G_OPEN = "<|tool_call>"
comptime _G_CLOSE = "<tool_call|>"


struct ToolCall(Copyable, Movable):
    var name: String  # function name
    var arguments: String  # arguments serialized as a JSON string (OpenAI shape)

    def __init__(out self, var name: String, var arguments: String):
        self.name = name^
        self.arguments = arguments^


struct ParsedReply(Movable):
    var content: String  # text outside any tool block (trimmed)
    var reasoning: String  # thinking-channel content ("" for Qwen / no channel)
    var calls: List[ToolCall]  # tool calls in emission order

    def __init__(
        out self,
        var content: String,
        var reasoning: String,
        var calls: List[ToolCall],
    ):
        self.content = content^
        self.reasoning = reasoning^
        self.calls = calls^

    def has_calls(self) -> Bool:
        return len(self.calls) > 0


def _find(b: List[UInt8], needle: List[UInt8], start: Int) -> Int:
    """First index >= start where `needle` occurs in `b`, or -1."""
    var n = len(b)
    var m = len(needle)
    if m == 0:
        return start
    var i = start
    while i + m <= n:
        var ok = True
        for j in range(m):
            if b[i + j] != needle[j]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _slice(b: List[UInt8], start: Int, stop: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(start, stop):
        out.append(b[i])
    return out^


def _strip_trailing_comma(mut out: List[UInt8]):
    """Drop trailing whitespace + one trailing comma from `out`, so closing a
    structure never produces `,}` / `,]`."""
    while len(out) > 0:
        var ch = out[len(out) - 1]
        if ch == 32 or ch == 9 or ch == 10 or ch == 13:
            _ = out.pop()
        else:
            break
    if len(out) > 0 and out[len(out) - 1] == 44:  # comma
        _ = out.pop()
        while len(out) > 0:
            var ch = out[len(out) - 1]
            if ch == 32 or ch == 9 or ch == 10 or ch == 13:
                _ = out.pop()
            else:
                break


def repair_json(s: String) -> String:
    """Best-effort repair of the not-quite-JSON small models emit in tool calls.

    A single forward pass with a stack of expected closers (respecting string
    literals): a closer that mismatches the stack top first shuts the inner
    structure with the *right* delimiter (`["x"}` → `["x"]`), anything still open
    at end-of-input gets its closers appended (truncated output), and stray
    closers + trailing commas are dropped. Conservative — callers still
    `parse_json` the result and fall back to verbatim text if it's beyond repair.
    Fixes the common 0.5B failures (`["q"}}` missing the `]`, cut-off generation)
    but not everything."""
    var b = string_to_bytes(s)
    var out = List[UInt8]()
    var stack = List[
        UInt8
    ]()  # expected closers ('}'=125 / ']'=93), innermost last
    var in_str = False
    var esc = False
    var i = 0
    var n = len(b)
    while i < n:
        var ch = b[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == 92:  # backslash
                esc = True
            elif ch == 34:  # closing quote
                in_str = False
            i += 1
            continue
        if ch == 34:  # opening quote
            out.append(ch)
            in_str = True
            i += 1
            continue
        if ch == 123:  # {
            out.append(ch)
            stack.append(125)
            i += 1
            continue
        if ch == 91:  # [
            out.append(ch)
            stack.append(93)
            i += 1
            continue
        if ch == 125 or ch == 93:  # } or ]
            if len(stack) == 0:
                i += 1  # stray closer — drop it
                continue
            var top = stack[len(stack) - 1]
            if ch == top:
                _strip_trailing_comma(out)
                out.append(ch)
                _ = stack.pop()
                i += 1
            else:
                # wrong closer: shut the inner structure properly, then re-handle ch
                _strip_trailing_comma(out)
                out.append(top)
                _ = stack.pop()
            continue
        out.append(ch)
        i += 1
    if in_str:
        out.append(34)  # close an unterminated string
    while len(stack) > 0:
        _strip_trailing_comma(out)
        out.append(stack.pop())
    return bytes_to_string(out)


def _try_parse(txt: String) -> Optional[Value]:
    """Repair to balanced JSON, then parse; None if still unparseable.

    We repair *first* rather than trying strict parse and falling back: jinja2.mojo's
    parse_json indexes past the end of truncated input and hard-*crashes* (assert)
    instead of raising, which `except` can't catch — so we must never hand it
    unbalanced text. repair_json is identity on already-valid JSON, so well-formed
    calls parse unchanged."""
    try:
        return parse_json(repair_json(txt))
    except:
        return None


def _extract_call(inner: List[UInt8], mut calls: List[ToolCall]) raises -> Bool:
    """Parse one <tool_call> body (strict, then repaired) and append it. Returns
    True iff a call with a `name` was extracted."""
    var opt = _try_parse(bytes_to_string(inner))
    if not opt:
        return False
    var v = opt.value()
    var nm = v.map_get("name")
    if nm and nm.value().tag == VSTR:
        var args_str = String("{}")
        var args = v.map_get("arguments")
        if args and not args.value().is_none():
            args_str = to_json(args.value(), 0)
        calls.append(ToolCall(nm.value().s, args_str^))
        return True
    return False


def parse_tool_calls(text: String) raises -> ParsedReply:
    """Split a raw completion into surrounding text + structured tool calls.

    Scans for `<tool_call> … </tool_call>` spans and pulls out `name` +
    `arguments` (re-serialized to the JSON *string* the OpenAI schema wants),
    via `_extract_call` which parses strictly and then falls back to a repair
    pass for the malformed JSON small models often emit. A block beyond repair
    (or name-less) is preserved verbatim in `content`. A trailing `<tool_call>`
    with no closing tag (truncated generation) is repaired from what's there."""
    var b = string_to_bytes(text)
    var OPEN = string_to_bytes(String("<tool_call>"))
    var CLOSE = string_to_bytes(String("</tool_call>"))
    var content = List[UInt8]()
    var calls = List[ToolCall]()
    var pos = 0
    var n = len(b)

    while pos < n:
        var o = _find(b, OPEN, pos)
        if o < 0:
            content += _slice(b, pos, n)
            break
        content += _slice(b, pos, o)  # text before the tag
        var inner_start = o + len(OPEN)
        var c = _find(b, CLOSE, inner_start)
        if c < 0:
            # no closing tag (truncated): try to recover the partial call, else verbatim
            if not _extract_call(_slice(b, inner_start, n), calls):
                content += _slice(b, o, n)
            break
        var advanced = c + len(CLOSE)
        if not _extract_call(_slice(b, inner_start, c), calls):
            content += _slice(b, o, advanced)  # beyond repair — keep verbatim
        pos = advanced

    return ParsedReply(
        String(bytes_to_string(content).strip()), String(""), calls^
    )


# ── Gemma tool-call parsing (format differs from Qwen's JSON blocks) ──────────


def _is_ws(c: UInt8) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def _at(b: List[UInt8], i: Int, needle: List[UInt8]) -> Bool:
    """Whether `needle` occurs at index `i` in `b`."""
    if i < 0 or i + len(needle) > len(b):
        return False
    for j in range(len(needle)):
        if b[i + j] != needle[j]:
            return False
    return True


def _g_string(b: List[UInt8], mut i: Int, quote: List[UInt8]) raises -> Value:
    """Parse `<|"|>…<|"|>`; `i` is at the opening quote token."""
    i += len(quote)
    var start = i
    while i < len(b) and not _at(b, i, quote):
        i += 1
    var s = String(bytes_to_string(_slice(b, start, i)))
    if i < len(b):
        i += len(quote)
    return Value.string(s^)


def _g_number(b: List[UInt8], mut i: Int) raises -> Value:
    var start = i
    var is_float = False
    while i < len(b):
        var c = b[i]
        if (c >= 48 and c <= 57) or c == 45 or c == 43:  # digit - +
            i += 1
        elif c == 46 or c == 101 or c == 69:  # . e E
            is_float = True
            i += 1
        else:
            break
    var s = String(bytes_to_string(_slice(b, start, i)))
    if is_float:
        return Value.float(atof(s))
    return Value.int(Int(atol(s)))


def _g_value(b: List[UInt8], mut i: Int, quote: List[UInt8]) raises -> Value:
    while i < len(b) and _is_ws(b[i]):
        i += 1
    if _at(b, i, quote):
        return _g_string(b, i, quote)
    if i < len(b) and b[i] == 123:  # {
        return _g_object(b, i, quote)
    if i < len(b) and b[i] == 91:  # [
        return _g_array(b, i, quote)
    if _at(b, i, string_to_bytes(String("true"))):
        i += 4
        return Value.bool(True)
    if _at(b, i, string_to_bytes(String("false"))):
        i += 5
        return Value.bool(False)
    if _at(b, i, string_to_bytes(String("null"))):
        i += 4
        return Value.none()
    return _g_number(b, i)


def _g_object(b: List[UInt8], mut i: Int, quote: List[UInt8]) raises -> Value:
    i += 1  # consume {
    var v = Value.mapping()
    while True:
        while i < len(b) and _is_ws(b[i]):
            i += 1
        if i >= len(b) or b[i] == 125:  # }
            break
        var ks = i
        while i < len(b) and b[i] != 58 and b[i] != 125:  # key until ':' / '}'
            i += 1
        var key = String(bytes_to_string(_slice(b, ks, i)).strip())
        if i < len(b) and b[i] == 58:  # consume ':'
            i += 1
        var val = _g_value(b, i, quote)
        v.map_set(key, val^)
        while i < len(b) and _is_ws(b[i]):
            i += 1
        if i < len(b) and b[i] == 44:  # consume ','
            i += 1
    if i < len(b) and b[i] == 125:  # consume }
        i += 1
    return v


def _g_array(b: List[UInt8], mut i: Int, quote: List[UInt8]) raises -> Value:
    i += 1  # consume [
    var items = List[Value]()
    while True:
        while i < len(b) and _is_ws(b[i]):
            i += 1
        if i >= len(b) or b[i] == 93:  # ]
            break
        items.append(_g_value(b, i, quote))
        while i < len(b) and _is_ws(b[i]):
            i += 1
        if i < len(b) and b[i] == 44:
            i += 1
    if i < len(b) and b[i] == 93:  # consume ]
        i += 1
    return Value.list_of(items^)


def _extract_gemma_call(inner: List[UInt8], mut calls: List[ToolCall]) -> Bool:
    """Parse one `call:NAME{…}` body; append a ToolCall (args → JSON string).
    Defensive — any malformed body returns False so it's kept verbatim."""
    try:
        var quote = string_to_bytes(String(_G_QUOTE))
        var i = 0
        var n = len(inner)
        while i < n and _is_ws(inner[i]):
            i += 1
        if not _at(inner, i, string_to_bytes(String("call:"))):
            return False
        i += 5
        var ns = i
        while i < n and inner[i] != 123:  # name until '{'
            i += 1
        var name = String(bytes_to_string(_slice(inner, ns, i)).strip())
        if name.byte_length() == 0:
            return False
        var args_json = String("{}")
        if i < n and inner[i] == 123:
            var obj = _g_object(inner, i, quote)
            args_json = to_json(obj, 0)
        calls.append(ToolCall(name^, args_json^))
        return True
    except:
        return False


def _split_gemma_channels(b: List[UInt8]) raises -> Tuple[String, String]:
    """Split Gemma output into (answer, reasoning). The model wraps its chain of
    thought in `<|channel>thought\\n…<channel|>` (empty in non-thinking mode); the
    text inside the channel(s) is the reasoning, everything outside is the answer.
    Stray channel markers are dropped. Both are trimmed."""
    var OPEN = string_to_bytes(String("<|channel>"))
    var CLOSE = string_to_bytes(String("<channel|>"))
    var out = List[UInt8]()
    var reason = List[UInt8]()
    var i = 0
    var n = len(b)
    while i < n:
        if _at(b, i, OPEN):
            # channel body runs to the matching close, or to end-of-text when the
            # thought was cut off mid-stream (still captured as reasoning).
            var j = _find(b, CLOSE, i + len(OPEN))
            var body_end = j if j >= 0 else n
            var cs = i + len(OPEN)
            var nl = cs
            while (
                nl < body_end and b[nl] != 10
            ):  # first newline ends the channel label
                nl += 1
            var rstart = nl + 1 if nl < body_end else cs
            if len(reason) > 0:
                reason.append(10)  # separate multiple channels with \n
            for k in range(rstart, body_end):
                reason.append(b[k])
            if j < 0:
                break  # unterminated → rest was all reasoning
            i = j + len(CLOSE)
        elif _at(b, i, CLOSE):
            i += len(CLOSE)  # stray close marker
        else:
            out.append(b[i])
            i += 1
    return (
        String(bytes_to_string(out).strip()),
        String(bytes_to_string(reason).strip()),
    )


def parse_gemma_tool_calls(text: String) raises -> ParsedReply:
    """Split a Gemma completion into surrounding text + structured tool calls.

    Scans for `<|tool_call>call:NAME{…}<tool_call|>` spans (Gemma's emitted
    format), parses NAME + the Gemma-serialized args into OpenAI `arguments`
    (a JSON string). Text outside spans is `content`; an unparseable span is
    kept verbatim. A trailing span with no closing token (truncated) is parsed
    from what's there."""
    var b = string_to_bytes(text)
    var OPEN = string_to_bytes(String(_G_OPEN))
    var CLOSE = string_to_bytes(String(_G_CLOSE))
    var content = List[UInt8]()
    var calls = List[ToolCall]()
    var pos = 0
    var n = len(b)
    # `<|tool_response>` is the model's turn boundary after emitting tool calls;
    # greedy decode often runs past it and hallucinates a tool result. Anything
    # from that marker on is not the model's real content, so cut it off.
    var tr = _find(b, string_to_bytes(String("<|tool_response>")), 0)
    if tr >= 0:
        n = tr
    while pos < n:
        var o = _find(b, OPEN, pos)
        if o < 0:
            content += _slice(b, pos, n)
            break
        content += _slice(b, pos, o)
        var inner_start = o + len(OPEN)
        var c = _find(b, CLOSE, inner_start)
        var end = c if c >= 0 else n
        if not _extract_gemma_call(_slice(b, inner_start, end), calls):
            var keep_end = (c + len(CLOSE)) if c >= 0 else n
            content += _slice(b, o, keep_end)
        pos = (c + len(CLOSE)) if c >= 0 else n
    var split = _split_gemma_channels(content)
    return ParsedReply(split[0], split[1], calls^)
