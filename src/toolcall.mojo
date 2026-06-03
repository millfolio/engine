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


struct ToolCall(Movable, Copyable):
    var name: String       # function name
    var arguments: String  # arguments serialized as a JSON string (OpenAI shape)

    def __init__(out self, var name: String, var arguments: String):
        self.name = name^
        self.arguments = arguments^


struct ParsedReply(Movable):
    var content: String        # text outside any <tool_call> block (trimmed)
    var calls: List[ToolCall]  # tool calls in emission order

    def __init__(out self, var content: String, var calls: List[ToolCall]):
        self.content = content^
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
    var stack = List[UInt8]()   # expected closers ('}'=125 / ']'=93), innermost last
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
            elif ch == 92:      # backslash
                esc = True
            elif ch == 34:      # closing quote
                in_str = False
            i += 1
            continue
        if ch == 34:            # opening quote
            out.append(ch)
            in_str = True
            i += 1
            continue
        if ch == 123:           # {
            out.append(ch)
            stack.append(125)
            i += 1
            continue
        if ch == 91:            # [
            out.append(ch)
            stack.append(93)
            i += 1
            continue
        if ch == 125 or ch == 93:   # } or ]
            if len(stack) == 0:
                i += 1              # stray closer — drop it
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
        out.append(34)             # close an unterminated string
    while len(stack) > 0:
        _strip_trailing_comma(out)
        out.append(stack.pop())
    return bytes_to_string(out)


def _try_parse(txt: String) -> Optional[Value]:
    """Repair to balanced JSON, then parse; None if still unparseable.

    We repair *first* rather than trying strict parse and falling back: minja2's
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

    return ParsedReply(String(bytes_to_string(content).strip()), calls^)
