"""Chat-template rendering via jinja2.mojo (ARCHITECTURE.md §5.3).

Renders the model's real Jinja chat template (assets/qwen2.5-chat-template.jinja)
with the ../jinja2.mojo engine, replacing the hardcoded no-tools template the CLI and
server used. The messages context is built as JSON and parsed into a jinja2.mojo
`Value` (simpler than constructing values by hand). Compile once, render many.

Built with `-I ../jinja2.mojo/src` so jinja2.mojo's modules resolve (it compiles cleanly
under the same 1.0.0b2 nightly the GPU engine needs — unlike flare, §11 #11).
"""

from template import Template
from value import Value
from json import parse_json, bytes_to_string, string_to_bytes
from runtime.model_iface import FAMILY_QWEN, FAMILY_GEMMA
from chat.gemma_chat import render_gemma


def _hex_nibble(n: Int) -> UInt8:
    return UInt8(48 + n) if n < 10 else UInt8(97 + n - 10)  # 0-9, a-f


def json_escape_str(b: List[UInt8]) -> String:
    """JSON-escape UTF-8 bytes into a string. Operates at the byte level so
    multibyte UTF-8 (é, emoji, CJK…) passes through intact — building the string
    with `chr(byte)` per byte would mojibake it. All control bytes < 0x20 must be
    escaped for valid JSON: the common ones get short escapes, the rest `\\u00XX`
    (the model can emit e.g. form-feed/vertical-tab, which a raw byte would make
    the response invalid JSON)."""
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
        elif c < 0x20:  # other control char -> \u00XX
            out.append(92)  # backslash
            out.append(117)  # u
            out.append(48)  # 0
            out.append(48)  # 0
            out.append(_hex_nibble((c >> 4) & 0xF))
            out.append(_hex_nibble(c & 0xF))
        else:
            out.append(b[i])
    return bytes_to_string(out)


def load_chat_template(path: String) raises -> Template:
    with open(path, "r") as f:
        return Template.compile(f.read())


def render_value(
    tmpl: Template, req: Value, family: Int = FAMILY_QWEN
) raises -> String:
    """Render the template from an already-parsed OpenAI request `Value`.

    The request's `messages` (full multi-turn history, with any `tool_calls`) and
    optional `tools` are exactly the shape the Qwen template consumes — the same
    inputs transformers' apply_chat_template takes — so we pass them straight
    through, adding `add_generation_prompt`.

    Gemma (`family == FAMILY_GEMMA`) does not go through Jinja at all — its prompt
    format (model-turn folding of tool messages, tool-call/result formatting, the
    thinking channel) is beyond jinja2.mojo, so it's rendered in Mojo by
    `render_gemma`; `tmpl` is ignored for Gemma.
    """
    if family == FAMILY_GEMMA:
        return render_gemma(req)

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
