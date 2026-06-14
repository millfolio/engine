"""Validate the Gemma chat template (incl. thinking + tool definitions) against
transformers' apply_chat_template, byte-for-byte. Renders each fixture case
through chat.render_value (FAMILY_GEMMA) — the exact server path — and compares
to the captured `want`. Pure CPU (no weights/GPU).

  pixi run mojo run -I src -I ../jinja2.mojo/src -I ../flare tests/manual/gemma_template_test.mojo
"""

from chat import load_chat_template, render_value
from model_iface import FAMILY_GEMMA
from json import parse_json
from value import VLIST, VSTR

comptime TMPL = "assets/gemma4-chat-template.jinja"
comptime CASES = "tests/fixtures/gemma/chat_cases.json"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def main() raises:
    var tmpl = load_chat_template(TMPL)
    var cases = parse_json(_read(CASES))
    var n = len(cases.c[].vals)
    var fails = 0
    for i in range(n):
        var cs = cases.c[].vals[i]
        var name = cs.map_get("name").value().s
        var body = cs.map_get("body").value().s
        var want = cs.map_get("want").value().s
        var got = render_value(tmpl, parse_json(body), FAMILY_GEMMA)
        if got == want:
            print("OK   :: ", name, sep="")
        else:
            fails += 1
            print("FAIL :: ", name, sep="")
            print("   got : ", repr(got))
            print("   want: ", repr(want))
    print("")
    if fails == 0:
        print("ALL ", n, " gemma template cases match transformers", sep="")
    else:
        print(fails, "/", n, " FAILED", sep="")
