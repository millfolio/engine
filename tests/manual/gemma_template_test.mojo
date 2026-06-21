"""Validate the Gemma prompt renderer (render_gemma) against transformers'
apply_chat_template, byte-for-byte, over tests/fixtures/gemma/chat_cases.json.
Covers text, thinking, tool definitions, assistant tool_calls, and tool results
(incl. multi-turn folding). Pure CPU (no weights/GPU).

  pixi run mojo run -I src -I ../jinja2.mojo/src -I ../flare tests/manual/gemma_template_test.mojo
"""

from chat.gemma_chat import render_gemma
from json import parse_json

comptime CASES = "tests/fixtures/gemma/chat_cases.json"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def main() raises:
    var cases = parse_json(_read(CASES))
    var n = len(cases.c[].vals)
    var fails = 0
    for i in range(n):
        var cs = cases.c[].vals[i]
        var name = cs.map_get("name").value().s
        var body = cs.map_get("body").value().s
        var want = cs.map_get("want").value().s
        var got = render_gemma(parse_json(body))
        if got == want:
            print("OK   :: ", name, sep="")
        else:
            fails += 1
            print("FAIL :: ", name, sep="")
            print("   got : ", repr(got))
            print("   want: ", repr(want))
    print("")
    if fails == 0:
        print("ALL ", n, " gemma render cases match transformers", sep="")
    else:
        print(fails, "/", n, " FAILED", sep="")
        raise Error("gemma render gate failed")
