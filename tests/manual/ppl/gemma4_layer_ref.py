"""Capture the HF Gemma-4 12B per-layer reference fixtures for the Mojo validators.

Runs the HF Gemma-4 12B text decoder on a fixed 12-token input and dumps, for
EVERY decoder layer L, the input and output hidden states ([1,12,3840], f32) to
tests/fixtures/gemma/layer<L>_{in,out}.bin — the fixtures `pixi run gemma-layer-test`
compares against — plus ids.txt + ref_argmax.txt (last-position argmax, for
`pixi run gemma-dump`) and meta.json.

FITS 24 GB. The full bf16 12B is ~24 GB and won't co-reside with the OS on a 24 GB
box (a naive load gets OOM-killed). So this drives the model ONE LAYER AT A TIME:
it instantiates the library's own `Gemma4TextDecoderLayer` / rotary / norm classes
(so the math is HF's, not hand-rolled), loads just that layer's weights, runs it,
captures in/out, and frees it. Peak resident ≈ the token-embedding matrix (~2 GB)
+ one layer (~0.5 GB). The 12B config has PLE and KV-sharing OFF
(hidden_size_per_layer_input=0, num_kv_shared_layers=0), so per_layer_input=None
and shared_kv_states={} — the dense path `src/models/gemma.mojo` implements.

Self-check: it asserts the layers it re-derives match any already-committed
fixtures (layer0/layer5), so a mistake in this driver fails loudly rather than
silently producing a wrong reference.

Run:  pixi run -e oracle gemma-ref     (GEMMA_REF_DEVICE=cpu default, or cuda)
"""
import os
import glob
import json
import numpy as np
import torch
from transformers import AutoTokenizer
import transformers.models.gemma4.modeling_gemma4 as mg
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from safetensors import safe_open

torch.set_grad_enabled(False)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "tests", "fixtures", "gemma")
os.makedirs(FIX, exist_ok=True)
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/*"))[0]
DEVICE = os.environ.get("GEMMA_REF_DEVICE", "cpu")
# f32 compute is the reference precision: the committed fixtures were captured in
# f32, and running in f32 reproduces them EXACTLY (bf16 accumulation scatters ~2%).
# Weights load from the bf16 checkpoint and upcast to f32. Peak ~5 GB, fits 24 GB.
DT = torch.float32
S = 12

cfg = Gemma4TextConfig(**json.load(open(os.path.join(SNAP, "config.json")))["text_config"])
cfg._attn_implementation = "eager"  # additive float causal mask path

# ── key -> shard index, and the text-decoder prefix ──────────────────────────
SHARDS = glob.glob(os.path.join(SNAP, "*.safetensors"))
key2shard = {}
for f in SHARDS:
    with safe_open(f, "pt") as h:
        for k in h.keys():
            key2shard[k] = f
PFX = next(k[: -len("embed_tokens.weight")] for k in key2shard
           if k.endswith("embed_tokens.weight"))  # e.g. "language_model.model."


def get(key):
    with safe_open(key2shard[key], "pt") as h:
        return h.get_tensor(key).to(DT)


# ── tiny always-resident modules: embed (~2 GB), norm, rotary ────────────────
embed = mg.Gemma4TextScaledWordEmbedding(
    cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, embed_scale=cfg.hidden_size ** 0.5)
embed.weight.data = get(PFX + "embed_tokens.weight")
embed = embed.to(DEVICE).eval()

norm = mg.Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
norm.weight.data = get(PFX + "norm.weight")
norm = norm.to(DEVICE).eval()

rotary = mg.Gemma4TextRotaryEmbedding(cfg).to(DEVICE)

# ── fixed 12-token input (exactly S ids: BOS + S-1 content tokens) ───────────
tok = AutoTokenizer.from_pretrained(SNAP)
content = tok("The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
              add_special_tokens=False)["input_ids"]
assert len(content) >= S - 1, f"need >= {S-1} content tokens, got {len(content)}"
ids = [2] + content[:S - 1]
ids_t = torch.tensor([ids], device=DEVICE)
pos = torch.arange(S, device=DEVICE).unsqueeze(0)

hidden = embed(ids_t)  # [1, S, hidden], already ×sqrt(hidden)
pe = {
    "full_attention": rotary(hidden, pos, "full_attention"),
    "sliding_attention": rotary(hidden, pos, "sliding_attention"),
}
# Build the per-type masks with the SAME library builders the model's forward
# uses (create_*_causal_mask), so the format matches the configured attention
# impl exactly — a hand-rolled additive mask silently diverged.
_mkw = dict(config=cfg, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=pos)
maskmap = {
    "full_attention": create_causal_mask(**_mkw),
    "sliding_attention": create_sliding_window_causal_mask(**_mkw),
}


def _dump(name, t):
    t.detach().float().cpu().numpy().reshape(-1).astype("<f4").tofile(os.path.join(FIX, name))


def build_layer(L):
    layer = mg.Gemma4TextDecoderLayer(cfg, L)  # constructed float32 by default
    lpfx = PFX + f"layers.{L}."
    sd = {k[len(lpfx):]: get(k) for k in key2shard if k.startswith(lpfx)}
    layer.load_state_dict(sd, strict=False)
    return layer.to(device=DEVICE, dtype=DT).eval()  # unify dtype (f32) + device


def run_layer(layer, L, hin):
    typ = cfg.layer_types[L]
    return layer(hin, None, shared_kv_states={},
                 position_embeddings=pe[typ],
                 attention_mask=maskmap[typ], position_ids=pos, past_key_values=None)


# ── driver self-check vs the ALREADY-committed fixtures (independent oracle) ──
# Feed each committed layer<L>_in into this driver and confirm it reproduces the
# committed layer<L>_out. Those fixtures came from a *separate* full-model HF run,
# so a match proves this layer-by-layer driver == the full HF model (breaking any
# circularity) BEFORE we overwrite the fixtures below.
for L in (0, 5):
    pin, pout = os.path.join(FIX, f"layer{L}_in.bin"), os.path.join(FIX, f"layer{L}_out.bin")
    if os.path.exists(pin) and os.path.exists(pout):
        hin = torch.tensor(np.fromfile(pin, dtype="<f4").reshape(1, S, cfg.hidden_size),
                           dtype=DT, device=DEVICE)
        got = run_layer(build_layer(L), L, hin).detach().float().cpu().numpy().reshape(-1)
        d = float(np.max(np.abs(got - np.fromfile(pout, dtype="<f4"))))
        print(f"  driver self-check L{L} vs committed fixture: max|Δ|={d:.2e}  "
              f"{'ok' if d < 5e-3 else 'MISMATCH'}")
        if d >= 5e-3:
            raise SystemExit(f"driver disagrees with trusted L{L} ({d:.2e}) — NOT overwriting fixtures")

# ── main capture: propagate through all layers, one resident at a time ───────
captured = 0
for L in range(cfg.num_hidden_layers):
    typ = cfg.layer_types[L]
    _dump(f"layer{L}_in.bin", hidden)
    hidden = run_layer(build_layer(L), L, hidden)
    _dump(f"layer{L}_out.bin", hidden)
    captured += 1
    print(f"  captured L{L} ({typ.split('_')[0]})")

# ── final norm + tied lm head → argmax (softcap is monotonic, so skip it) ─────
final = norm(hidden)
logits = (final[0, -1].float() @ embed.weight.data.float().t())
argmax = int(logits.argmax())

open(os.path.join(FIX, "ids.txt"), "w").write(" ".join(str(i) for i in ids))
open(os.path.join(FIX, "ref_argmax.txt"), "w").write(str(argmax))
json.dump(
    {"S": S, "hidden": cfg.hidden_size, "layers_captured": list(range(captured)),
     "ids": ids, "final_argmax": argmax},
    open(os.path.join(FIX, "meta.json"), "w"),
)
print(f"wrote {captured} layers x (in,out) + ids/argmax to {FIX}")
print(f"final-position argmax = {argmax}")
