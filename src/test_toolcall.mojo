"""Gate: <tool_call> parsing (pure, no GPU). `pixi run test-toolcall`.

Drives `parse_tool_calls` over the shapes the model actually emits — single call,
multiple calls, a call wrapped in prose, no call at all, and a malformed block —
and checks the extracted name/arguments and leftover content.
"""

from toolcall import parse_tool_calls, ParsedReply


def expect(cond: Bool, msg: String, mut ok: Bool):
    if not cond:
        print("  FAIL: ", msg)
        ok = False


def main() raises:
    var all_ok = True

    # 1. single call, the canonical Qwen framing (newlines around the JSON)
    var r1 = parse_tool_calls(
        String('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Cluj"}}\n</tool_call>')
    )
    expect(r1.has_calls(), "1: expected a call", all_ok)
    expect(len(r1.calls) == 1, "1: expected exactly 1 call", all_ok)
    expect(r1.calls[0].name == "get_weather", "1: name", all_ok)
    expect(r1.calls[0].arguments == '{"city": "Cluj"}', "1: args=" + r1.calls[0].arguments, all_ok)
    expect(r1.content == "", "1: content should be empty, got [" + r1.content + "]", all_ok)

    # 2. two calls back to back
    var r2 = parse_tool_calls(
        String('<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>')
        + '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    expect(len(r2.calls) == 2, "2: expected 2 calls", all_ok)
    expect(r2.calls[0].name == "a" and r2.calls[1].name == "b", "2: names", all_ok)
    expect(r2.calls[1].arguments == "{}", "2: empty args=" + r2.calls[1].arguments, all_ok)

    # 3. prose around a call -> prose becomes content, call extracted
    var r3 = parse_tool_calls(
        String('Let me check.\n<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>')
    )
    expect(len(r3.calls) == 1, "3: expected 1 call", all_ok)
    expect(r3.content == "Let me check.", "3: content=[" + r3.content + "]", all_ok)

    # 4. plain answer, no tool call
    var r4 = parse_tool_calls(String("The capital of Romania is Bucharest."))
    expect(not r4.has_calls(), "4: should have no calls", all_ok)
    expect(r4.content == "The capital of Romania is Bucharest.", "4: content", all_ok)

    # 5. malformed block (not JSON) -> preserved verbatim, no call
    var r5 = parse_tool_calls(String("<tool_call>\nnot json\n</tool_call>"))
    expect(not r5.has_calls(), "5: malformed should yield no call", all_ok)
    expect(r5.content.find("not json") >= 0, "5: malformed kept in content=[" + r5.content + "]", all_ok)

    if all_ok:
        print("toolcall gate: PASS")
    else:
        print("toolcall gate: FAIL")
        raise Error("toolcall gate failed")
