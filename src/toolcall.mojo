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


def parse_tool_calls(text: String) raises -> ParsedReply:
    """Split a raw completion into surrounding text + structured tool calls.

    Scans for `<tool_call> … </tool_call>` spans, JSON-parses each inner object,
    and pulls out `name` + `arguments` (re-serialized to the JSON *string* the
    OpenAI schema wants). Malformed or name-less blocks, and any unterminated
    trailing `<tool_call>`, are preserved verbatim in `content`."""
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
            content += _slice(b, o, n)  # unterminated — keep verbatim
            break
        var advanced = c + len(CLOSE)

        var ok = False
        try:
            var v = parse_json(bytes_to_string(_slice(b, inner_start, c)))
            var nm = v.map_get("name")
            if nm and nm.value().tag == VSTR:
                var args_str = String("{}")
                var args = v.map_get("arguments")
                if args and not args.value().is_none():
                    args_str = to_json(args.value(), 0)
                calls.append(ToolCall(nm.value().s, args_str^))
                ok = True
        except:
            ok = False
        if not ok:
            content += _slice(b, o, advanced)  # not a real call — keep verbatim
        pos = advanced

    return ParsedReply(String(bytes_to_string(content).strip()), calls^)
