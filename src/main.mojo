"""End-to-end pure-Mojo chat CLI: prompt → text, no Python on the path.

Ties the library together — chat-template render → tokenizer.encode → weight load
→ GPU greedy generate → tokenizer.decode — to answer a prompt entirely in Mojo on
the GPU (ARCHITECTURE.md §6 Phase 6, §1). This is the "it runs as a program"
milestone: the first fully pure-Mojo prompt→text path.

The chat template is the Qwen2.5 no-tools branch, hardcoded (a full Jinja render
via ../minja2 is the general path, §5.3). The checkpoint path comes from
$QWEN_SAFETENSORS or tests/fixtures/forward/meta.txt; the tokenizer tables from
tests/fixtures/tokenizer/ (run `tok-capture` once).

    pixi run chat -- "Your question here"
"""

from std.sys import argv
from std.os import getenv
from std.gpu.host import DeviceContext

from model import load_weights, generate, EOS1, EOS2
from tokenizer import load_tokenizer
from chat import load_chat_template, render_chat
from json import bytes_to_string

comptime MAX_NEW = 64
comptime TEMPLATE = "assets/qwen2.5-chat-template.jinja"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

def to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var sb = s.as_bytes()
    for i in range(len(sb)):
        out.append(sb[i])
    return out^

def main() raises:
    var user = String("What is the capital of France?")
    if len(argv()) > 1:
        var joined = String("")
        for i in range(1, len(argv())):
            if i > 1:
                joined += " "
            joined += String(argv()[i])
        user = joined

    var ckpt = String(getenv("QWEN_SAFETENSORS"))
    if ckpt.byte_length() == 0:
        ckpt = String(String(read_text("tests/fixtures/forward/meta.txt").split("\n")[1]).strip())

    var tok = load_tokenizer("tests/fixtures/tokenizer/")
    var tmpl = load_chat_template(TEMPLATE)
    var ids = tok.encode(to_bytes(render_chat(tmpl, user)))

    print("loading weights…")
    var ctx = DeviceContext()
    var w = load_weights(ctx, String(ckpt))
    var gen = generate(ctx, w, ids, MAX_NEW)

    # drop a trailing EOS for display
    var body = List[Int]()
    for i in range(len(gen)):
        if gen[i] == EOS1 or gen[i] == EOS2:
            break
        body.append(gen[i])

    print("\n>>> ", user, sep="")
    print(bytes_to_string(tok.decode(body)))
