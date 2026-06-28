"""Gemma 4 chat-prompt renderer in pure Mojo (replaces a Jinja template).

The upstream gemma4_unified chat template folds separate OpenAI `tool` messages
into the preceding assistant's *model* turn, continues one model turn across
consecutive assistant messages, leaves the turn open before generation when the
last message is a tool call/response, and formats tool-call args + tool results
with recursive `format_argument` over dictsort'd maps — none of which jinja2.mojo
can express (no macros, no lookahead-friendly slicing). So the whole Gemma prompt
is rendered here, byte-for-byte identical to transformers' apply_chat_template
(validated by tests/manual/gemma_template_test.mojo over tests/fixtures/gemma).

Scope: text + thinking channel + tool definitions + assistant tool_calls + tool
responses (the served paths). Not handled (no-op for the OpenAI inputs a client
sends): strip_thinking of past assistant content, and multimodal content parts
(text parts are concatenated; image/audio/video are dropped).
"""

from value import Value, VNONE, VBOOL, VINT, VFLOAT, VSTR, VLIST, VMAP
from chat.gemma_tools import format_argument, format_gemma_tools, dictsort, Q


def _role(msg: Value) -> String:
    var o = msg.map_get("role")
    if o and o.value().tag == VSTR:
        return o.value().s.copy()
    return String("")


def _content_value(msg: Value) -> Value:
    var o = msg.map_get("content")
    if o:
        return o.value()
    return Value.none()


def _content_text(msg: Value) raises -> String:
    """Assistant/user text content. String → itself; content-parts array → the
    text parts concatenated (image/audio/video parts dropped)."""
    var c = _content_value(msg)
    if c.tag == VSTR:
        return c.s.copy()
    if c.tag == VLIST:
        var out = String("")
        for i in range(len(c.c[].vals)):
            var part = c.c[].vals[i]
            var ty = part.map_get("type")
            if ty and ty.value().tag == VSTR and ty.value().s == "text":
                var tx = part.map_get("text")
                if tx and tx.value().tag == VSTR:
                    out += tx.value().s
        return out^
    return String("")


def _map_pairs(m: Value) raises -> String:
    """dictsort'd `key:format_argument(value, escape_keys=False)` pairs (no braces)
    — the shared body of tool-call args and tool-result maps."""
    var out = String("")
    var idx = dictsort(m)
    for ii in range(len(idx)):
        if ii > 0:
            out += ","
        out += (
            m.c[].keys[idx[ii]]
            + ":"
            + format_argument(m.c[].vals[idx[ii]], False)
        )
    return out^


def _tc_args(func: Value) raises -> String:
    """The inside of `call:NAME{…}`. OpenAI sends `arguments` as a JSON *string*
    (emitted verbatim, as upstream does → `{{…}}`); Gemma-native maps are
    formatted (bare keys, <|"|> strings)."""
    var a = func.map_get("arguments")
    if not a:
        return String("")
    var av = a.value()
    if av.tag == VSTR:
        return av.s.copy()
    if av.tag == VMAP:
        return _map_pairs(av)
    return String("")


def _resolve_tool_name(assistant: Value, tool_msg: Value) raises -> String:
    """Tool result's function name: match the tool message's tool_call_id against
    the assistant's tool_calls; else its own `name`; else 'unknown'."""
    var tcid = tool_msg.map_get("tool_call_id")
    var tcs = assistant.map_get("tool_calls")
    if tcid and tcid.value().tag == VSTR and tcs and tcs.value().tag == VLIST:
        var want = tcid.value().s
        for j in range(len(tcs.value().c[].vals)):
            var tc = tcs.value().c[].vals[j]
            var idv = tc.map_get("id")
            if idv and idv.value().tag == VSTR and idv.value().s == want:
                var func = tc.map_get("function")
                if func:
                    var nm = func.value().map_get("name")
                    if nm and nm.value().tag == VSTR:
                        return nm.value().s.copy()
    var own = tool_msg.map_get("name")
    if own and own.value().tag == VSTR:
        return own.value().s.copy()
    return String("unknown")


def _tool_response_block(name: String, body: Value) raises -> String:
    """`<|tool_response>response:NAME{…}<tool_response|>`. A map result formats
    its pairs; anything else is wrapped as `value:<formatted>`."""
    var out = String("<|tool_response>response:") + name + "{"
    if body.tag == VMAP:
        out += _map_pairs(body)
    else:
        out += "value:" + format_argument(body, False)
    out += "}<tool_response|>"
    return out^


def _prev_non_tool_role(M: Value, before: Int) -> String:
    var j = before - 1
    while j >= 0:
        var r = _role(M.c[].vals[j])
        if r != "tool":
            return r
        j -= 1
    return String("")


def render_gemma(req: Value) raises -> String:
    """Render an OpenAI request (`messages` + optional `tools`/`enable_thinking`)
    into the Gemma prompt, always adding the generation prompt (serving)."""
    var msgs_o = req.map_get("messages")
    if not msgs_o:
        raise Error("request has no 'messages' array")
    var M = msgs_o.value()
    var n = len(M.c[].vals)

    var et = req.map_get("enable_thinking")
    var thinking = Bool(et) and et.value().tag == VBOOL and et.value().b
    var to = req.map_get("tools")
    var has_tools = (
        Bool(to) and to.value().tag == VLIST and len(to.value().c[].vals) > 0
    )

    var out = String("<bos>")

    var first_is_system = n > 0 and (
        _role(M.c[].vals[0]) == "system" or _role(M.c[].vals[0]) == "developer"
    )
    if thinking or has_tools or first_is_system:
        out += "<|turn>system\n"
        if thinking:
            out += "<|think|>\n"
        if first_is_system:
            out += String(_content_text(M.c[].vals[0]).strip())
        if has_tools:
            out += format_gemma_tools(to.value())
        out += "<turn|>\n"

    var prev_type = String("none")  # tracks ns.prev_message_type across turns
    var i = 1 if first_is_system else 0
    while i < n:
        var msg = M.c[].vals[i]
        var role = _role(msg)
        if role == "tool":
            i += 1  # consumed by a preceding assistant's scan
            continue

        prev_type = "none"
        var label = String("model") if role == "assistant" else role
        var continue_turn = (label == "model") and (
            _prev_non_tool_role(M, i) == "assistant"
        )
        if not continue_turn:
            out += "<|turn>" + label + "\n"

        # assistant tool calls
        var tcs = msg.map_get("tool_calls")
        var has_tc = (
            Bool(tcs)
            and tcs.value().tag == VLIST
            and len(tcs.value().c[].vals) > 0
        )
        if has_tc:
            for j in range(len(tcs.value().c[].vals)):
                var tc = tcs.value().c[].vals[j]
                var func = tc.map_get("function").value()
                var nm = func.map_get("name").value().s
                out += (
                    "<|tool_call>call:"
                    + nm
                    + "{"
                    + _tc_args(func)
                    + "}<tool_call|>"
                )
            prev_type = "tool_call"

        # tool results: forward-scan the consecutive role:tool messages
        var tr_emitted = False
        if has_tc:
            var k = i + 1
            while k < n and _role(M.c[].vals[k]) == "tool":
                var tmsg = M.c[].vals[k]
                var tname = _resolve_tool_name(msg, tmsg)
                out += _tool_response_block(tname, _content_value(tmsg))
                tr_emitted = True
                prev_type = "tool_response"
                k += 1

        var content = String(_content_text(msg).strip())
        out += content
        var has_content = content.byte_length() > 0

        if prev_type == "tool_call" and not tr_emitted:
            out += "<|tool_response>"
        elif not (tr_emitted and not has_content):
            out += "<turn|>\n"
        i += 1

    if prev_type != "tool_response" and prev_type != "tool_call":
        out += "<|turn>model\n"
        if not thinking:
            out += "<|channel>thought\n<channel|>"
    return out^
