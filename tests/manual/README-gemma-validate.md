# Gemma 4 12B — dump-and-compare validation

Numeric validation of the Mojo Gemma-4 12B decoder (`src/models/gemma.mojo`)
against the HF `Gemma4ForCausalLM` reference, split so the two 12B-sized things
are **never co-resident** — which is what lets it work on a 24 GB machine.

The full bf16 HF reference is ~24 GB and the Mojo bf16 model is another ~24 GB, so
running either whole model OOMs a 24 GB box. The trick is to touch **one layer at
a time** on both sides:

1. **Capture the reference** — drive the HF model layer-by-layer (one
   `Gemma4TextDecoderLayer` resident, ~5 GB) → commit ~18 MB of fixtures.
2. **Validate** against those fixtures, loading only one Mojo layer (or the
   ~7 GB int4 model) at a time.

Everything below runs on a 24 GB machine. **Status: all 48 layers match
(max\|Δ\| ≈ 1e-5); int4 full-forward integration OK.**

## The three pieces

| Command | What it does | Memory |
|---|---|---|
| `pixi run -e oracle gemma-ref` | Drives HF Gemma-4 **one decoder layer at a time** (f32, the reference precision) → per-layer in/out hidden states + final argmax → `tests/fixtures/gemma/`. Self-checks it reproduces any committed L0/L5 fixture before overwriting. | ~5 GB |
| `pixi run gemma-layer-test` | Loads **one bf16 layer at a time**, feeds it the reference input, checks output (max\|Δ\| < 1e-2). Auto-covers every captured layer. | tiny (one layer) |
| `pixi run gemma-dump` | Full **int4** 48-layer forward on the same ids. Rank-based integration check: the f32 reference token must stay within the int4 top-6 (int4 quantization over 48 layers can flip the greedy token, so an *exact* argmax match isn't required). Catches embed scale / KV+RoPE / final-norm / softcap bugs. | ~7 GB |

`gemma-layer-test` **localizes** a broken layer (each layer gets the reference
input, so errors don't accumulate). `gemma-dump` catches **integration** bugs the
isolated per-layer test can't. Run both.

## Step 1 — capture the reference (once)

Needs the `oracle` pixi env (transformers 5.9.x, which has `gemma4`; the default
env has no torch/transformers). It drives the model one layer at a time, so it
fits 24 GB (~5 GB resident):

```bash
pixi run -e oracle gemma-ref                       # device=cpu by default
GEMMA_REF_DEVICE=cuda pixi run -e oracle gemma-ref  # if you have a CUDA GPU
```

It writes `layer<L>_{in,out}.bin` for all 48 layers, `ids.txt`, `ref_argmax.txt`,
`meta.json`, and self-checks that it reproduces any already-committed L0/L5 fixture
(max\|Δ\|=0) before overwriting — so a driver mistake fails loudly. The fixtures
are machine-independent; commit them and you rarely re-run this.

## Step 2 — validate locally (repeat freely, fits 24 GB)

```bash
pixi run gemma-layer-test    # per-layer bf16 parity, one layer at a time (all 48)
pixi run gemma-dump          # full int4 forward, rank-based integration check
```

`gemma-layer-test` auto-discovers whichever `layer<L>_in.bin` fixtures exist, so it
covers all 48 captured layers.

## If a layer FAILS

`gemma-layer-test` prints `max|Δ|` per layer and the first one over 1e-2 is the
culprit. This is exactly how the e2b bugs were found (wrong `layer_scalar` order,
double-wide MLP). Compare the Mojo layer's math to
`transformers/models/gemma4/modeling_gemma4.py` for that layer type
(sliding vs full — `_is_full_layer(L)`).
