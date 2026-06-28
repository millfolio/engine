"""Coherence smoke test for the gemma-4 e2b DRAFT model (int4): load all 35
layers, greedy-generate from a couple of chat prompts, print the decoded text.

There is no HF/MLX `gemma4` reference in this environment, so this is the primary
correctness signal for the PLE reconstruction: if the text is coherent, the forward
(PLE setup + per-layer integration + the sliding/full attention geometry) is
substantially right. `pixi run e2b-smoke`."""

from std.gpu.host import DeviceContext
from models.gemma_e2b import load_e2b_weights, GemmaE2bWeights, E_NLAYERS
from runtime.engine import (
    new_session,
    sess_prefill_suffix,
    sess_step,
    argmax_f,
    Session,
)
from chat import load_chat_template, render_value
from tokenizer import load_gemma_tokenizer_json, Tokenizer
from runtime.tensor_ops import probe_simd_gemm
from template import Template
from runtime.model_iface import FAMILY_GEMMA
from json import parse_json

comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"
comptime TMPL = (  # placeholder; render_value(FAMILY_GEMMA) renders in Mojo
    "assets/qwen2.5-chat-template.jinja"
)


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for byte in s.as_bytes():
        b.append(byte)
    return b^


def run(
    ctx: DeviceContext,
    mut gw: GemmaE2bWeights,
    mut sess: Session,
    tok: Tokenizer,
    tmpl: Template,
    eos1: Int,
    eos2: Int,
    idx: Int,
    body: String,
) raises:
    sess.pos = 0
    var ids = tok.encode(
        _bytes(render_value(tmpl, parse_json(body), FAMILY_GEMMA))
    )
    var logits = sess_prefill_suffix(ctx, gw, sess, ids, 0, False)
    var gen = List[Int]()
    var nxt = argmax_f(logits)
    var steps = 0
    while steps < 64 and nxt != eos1 and nxt != eos2:
        gen.append(nxt)
        logits = sess_step(ctx, gw, sess, nxt)
        nxt = argmax_f(logits)
        steps += 1
    var text = String(StringSlice(unsafe_from_utf8=Span(tok.decode(gen))))
    print("Q", idx, " prompt_toks=", len(ids), " gen=", len(gen), sep="")
    print("  -> ", text, sep="")


def main() raises:
    var ctx = DeviceContext()
    print("loading gemma-4 e2b (int4, 35 layers)…")
    var tok = load_gemma_tokenizer_json(SNAP + "/tokenizer.json")
    var tmpl = load_chat_template(TMPL)
    var alllayers = List[Int]()
    for i in range(E_NLAYERS):
        alllayers.append(i)
    var gw = load_e2b_weights(ctx, SNAP, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    print(
        "  simd_ok=",
        gw.simd_ok,
        " nlayers=",
        cfg.nlayers,
        " eos=",
        cfg.eos1,
        "/",
        cfg.eos2,
        sep="",
    )
    var sess = new_session(ctx, 2048, cfg.nlayers, cfg.nkv)

    run(
        ctx,
        gw,
        sess,
        tok,
        tmpl,
        cfg.eos1,
        cfg.eos2,
        0,
        (
            '{"messages":[{"role":"user","content":"What is the capital of'
            ' France? Answer in one word."}]}'
        ),
    )
    run(
        ctx,
        gw,
        sess,
        tok,
        tmpl,
        cfg.eos1,
        cfg.eos2,
        1,
        (
            '{"messages":[{"role":"user","content":"Name the first five prime'
            ' numbers."}]}'
        ),
    )
    run(
        ctx,
        gw,
        sess,
        tok,
        tmpl,
        cfg.eos1,
        cfg.eos2,
        2,
        (
            '{"messages":[{"role":"user","content":"Write one sentence about'
            ' why the sky is blue."}]}'
        ),
    )
