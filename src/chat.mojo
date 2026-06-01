"""Chat-template rendering via minja2 (ARCHITECTURE.md §5.3).

Renders the model's real Jinja chat template (assets/qwen2.5-chat-template.jinja)
with the ../minja2 engine, replacing the hardcoded no-tools template the CLI and
server used. The messages context is built as JSON and parsed into a minja2
`Value` (simpler than constructing values by hand). Compile once, render many.

Built with `-I ../minja2/src` so minja2's modules resolve (it compiles cleanly
under the same 1.0.0b2 nightly the GPU engine needs — unlike flare, §11 #11).
"""

from template import Template
from value import Value
from json import parse_json, bytes_to_string, string_to_bytes


def json_escape_str(b: List[UInt8]) -> String:
    """JSON-escape UTF-8 bytes into a string. Operates at the byte level so
    multibyte UTF-8 (é, emoji, CJK…) passes through intact — building the string
    with `chr(byte)` per byte would mojibake it."""
    var out = List[UInt8]()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 34:  # "
            out.append(92)
            out.append(34)
        elif c == 92:  # backslash
            out.append(92)
            out.append(92)
        elif c == 10:
            out.append(92)
            out.append(110)  # \n
        elif c == 13:
            out.append(92)
            out.append(114)  # \r
        elif c == 9:
            out.append(92)
            out.append(116)  # \t
        else:
            out.append(b[i])
    return bytes_to_string(out)


def load_chat_template(path: String) raises -> Template:
    with open(path, "r") as f:
        return Template.compile(f.read())


def render_value(tmpl: Template, req: Value) raises -> String:
    """Render the template from an already-parsed OpenAI request `Value`.

    The request's `messages` (full multi-turn history, with any `tool_calls`) and
    optional `tools` are exactly the shape the Qwen template consumes — the same
    inputs transformers' apply_chat_template takes — so we pass them straight
    through, adding `add_generation_prompt`.
    """
    var msgs = req.map_get("messages")
    if not msgs:
        raise Error("request has no 'messages' array")

    var ctx = Value.mapping()
    ctx.map_set("messages", msgs.value())
    ctx.map_set("add_generation_prompt", Value.bool(True))

    var tools = req.map_get("tools")
    if tools and not tools.value().is_none():
        ctx.map_set("tools", tools.value())
    else:
        ctx.map_set("tools", Value.none())

    return tmpl.render(ctx^, 0)


def render_request(tmpl: Template, body: String) raises -> String:
    """Parse an OpenAI request body and render it (CLI / single-shot use)."""
    return render_value(tmpl, parse_json(body))


def render_chat(tmpl: Template, user: String) raises -> String:
    """Convenience for a single user turn (the CLI), via `render_request`."""
    var body = (
        String('{"messages":[{"role":"user","content":"')
        + json_escape_str(string_to_bytes(user))
        + '"}]}'
    )
    return render_request(tmpl, body)
