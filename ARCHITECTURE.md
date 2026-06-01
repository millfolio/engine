# mojo-backend — Architecture

A **native-Mojo** re-implementation of the millrace inference path. Where
[`../max-backend`](../max-backend/ARCHITECTURE.md) drives MAX's *Python* pipeline
over Mojo↔Python interop (its Phase-2 pivot, since no MAX *Mojo* Graph API
ships), this repo takes on the piece max-backend explicitly deferred: **writing
the transformer forward pass by hand in Mojo** (max-backend §7 #1, §8 #1).

The first milestone is deliberately narrow: a **hardcoded Mojo implementation of
`Qwen/Qwen2.5-0.5B-Instruct`** — one model, one architecture, no config-driven
generality — that loads the real HuggingFace weights and produces coherent,
reference-matching text **on the Apple Silicon GPU (Metal), with hand-written
Mojo kernels**. Generality comes later, only once one model is proven end-to-end.

This is a bet max-backend's own findings underwrite: its isolation ladder
(its §8 #2, 2026-05-31) showed that **hand-written Mojo GPU kernels are correct**
on the M4 Metal backend (rungs 1–2: elementwise + matmul exact vs. CPU), as are
MAX-graph primitives at f32 *and* bf16 (rungs 3–5). The *only* thing that
produced garbage was MAX's **own fused attention / RoPE / KV-cache Metal kernel**
(rung 6) — precisely the kernel we replace by hand here. So writing our own
forward pass on Metal is expected to be coherent where driving MAX's was not.

## 1. Goals & non-goals

### Goals

1. **Load the real weights.** Read `model.safetensors` (bf16, ~988 MB) directly
   — no torch, no transformers, no MAX at runtime.
2. **Run the forward pass in pure Mojo on the GPU.** Embed → 24 Qwen2 decoder
   layers → final norm → tied LM head, computed with hand-written Mojo Metal
   kernels on Apple Silicon.
3. **Tokenize in pure Mojo.** Byte-level BPE (`Qwen2Tokenizer`) for
   encode/decode, from the shipped `vocab.json` + `merges.txt`.
4. **Template in pure Mojo.** Render the chat prompt via
   [`../minja2`](../minja2/docs/requirements.md) (the Jinja2-subset engine built
   for exactly this).
5. **Generate.** A greedy decode loop with a KV cache that produces text
   **token-for-token identical to the reference** — **MAX running on CPU/float32**
   (§7) — under greedy decoding.

### Design stance

- **No Python, no MAX, at runtime.** This is the "no Python at runtime" stance
  max-backend aspired to (its §2) but could not reach because MAX ships no Mojo
  graph API. Here we own every layer, so there is no interop boundary to begin
  with. Python is used **only as a build-time reference oracle** for conformance
  (§7), never on the request path.
- **Hardcode first, generalize never (yet).** Dimensions, layer count, head
  counts, RoPE base, the SwiGLU/GQA structure, and the weight-tensor names are
  all **compile-time constants** baked to Qwen2.5-0.5B (§3). No `config.json`
  parsing, no model registry, no architecture dispatch. A second model is a
  later, separate effort — and the seam it needs (a config struct) is called out
  in §6, not built now.
- **GPU-only, hand-written Mojo Metal kernels.** The forward pass runs on the
  Apple Silicon GPU through Mojo's native Metal backend — *not* MLX, *not* MAX.
  There is no production CPU compute path; CPU appears only as the conformance
  oracle (next bullet, §7). We compute in **float32** for v1 (weights are bf16 on
  disk, upcast on load) because that is what the CPU oracle runs and gives the
  cleanest diff; bf16-native GPU compute is a later optimization (§4), and
  max-backend rung 5 already showed Metal bf16 matmul/MLP bit-identical to CPU.
- **MAX-on-CPU is the conformance oracle.** max-backend established that MAX's
  CPU/float32 path is *coherent* while its Metal path returns garbage (its §8 #2,
  re-confirmed 2026-05-30). We exploit that asymmetry: MAX/CPU/float32 is the
  trusted reference our **GPU** output is diffed against (§7). MAX is a
  **test-time dependency only** — never on the runtime path.
- **Single-user, single-request.** No batching, no continuous batching, no
  concurrency. One prompt, one stream of tokens.

### Non-goals (v1)

- The HTTP/API server, OpenAI/Anthropic compatibility, streaming SSE — these
  already exist in max-backend (flare, pure Mojo) and are **ported in a later
  phase**, not rebuilt here. v1 is a CLI/library that generates from a prompt.
- Multiple model architectures / config-driven generality.
- A production **CPU** compute path. CPU is the conformance oracle only; the
  engine runs on the GPU.
- Tool calling, vision, embeddings.
- Quantization (the weights are bf16; we upcast to f32). No q4/gguf.
- The control plane (lifecycle, memory budget, KV tiering) — max-backend §3.3.
- Full per-request sampling parity (temperature/top_p/top_k/penalties) — greedy
  first for verifiability; sampling is a fast-follow (§5.6).

## 2. Target model: Qwen2.5-0.5B-Instruct

`Qwen2ForCausalLM`, from the shipped `config.json`. These are the constants the
implementation hardcodes:

| Hyperparameter | Value | Notes |
|---|---:|---|
| `vocab_size` | 151936 | byte-level BPE |
| `hidden_size` | 896 | model dim |
| `intermediate_size` | 4864 | SwiGLU MLP inner dim |
| `num_hidden_layers` | 24 | decoder blocks |
| `num_attention_heads` | 14 | → `head_dim` = 896/14 = **64** |
| `num_key_value_heads` | 2 | **GQA**: 7 query heads share each KV head; KV dim = 2×64 = 128 |
| `hidden_act` | `silu` | SwiGLU: `down(silu(gate(x)) * up(x))` |
| `rms_norm_eps` | 1e-6 | RMSNorm (not LayerNorm) |
| `rope_theta` | 1000000.0 | RoPE base |
| `max_position_embeddings` | 32768 | context limit |
| `tie_word_embeddings` | **true** | LM head reuses `embed_tokens.weight` — there is **no** `lm_head.weight` |
| `torch_dtype` | bfloat16 | on-disk dtype; compute in f32 |

**Attention bias quirk (Qwen2-specific).** The Q/K/V projections carry a **bias**
term; the output projection and the MLP do **not**. Confirmed from the
safetensors header:

```
model.embed_tokens.weight                  [151936, 896]   (also the tied LM head)
model.layers.{0..23}.input_layernorm.weight                [896]
model.layers.{0..23}.self_attn.q_proj.weight [896, 896]  .q_proj.bias [896]
model.layers.{0..23}.self_attn.k_proj.weight [128, 896]  .k_proj.bias [128]
model.layers.{0..23}.self_attn.v_proj.weight [128, 896]  .v_proj.bias [128]
model.layers.{0..23}.self_attn.o_proj.weight [896, 896]   (no bias)
model.layers.{0..23}.post_attention_layernorm.weight       [896]
model.layers.{0..23}.mlp.gate_proj.weight  [4864, 896]    (no bias)
model.layers.{0..23}.mlp.up_proj.weight    [4864, 896]    (no bias)
model.layers.{0..23}.mlp.down_proj.weight  [896, 4864]    (no bias)
model.norm.weight                                          [896]
```

290 tensors total. Linear weights are stored `[out, in]` (torch convention), so a
forward projection is `y = x · Wᵀ (+ b)`.

**Special tokens** (`tokenizer_config.json`, `generation_config.json`):
`<|im_start|>`=151644, `<|im_end|>`=151645 (the `eos_token`), `<|endoftext|>`=151643
(also a stop id). `bos_token` is `null` (`add_bos_token: false` — no BOS is
prepended). Generation stops on **either** 151645 or 151643.

## 3. Forward pass (one decoder layer)

Each of the 24 layers is the standard Qwen2/Llama-style pre-norm block:

```
h = x
# --- attention ---
n  = RMSNorm(h, input_layernorm.weight)
q  = n·q_projᵀ + q_bias        # [T, 896] → 14 heads × 64
k  = n·k_projᵀ + k_bias        # [T, 128] →  2 heads × 64
v  = n·v_projᵀ + v_bias        # [T, 128] →  2 heads × 64
q, k = RoPE(q, k, pos, theta=1e6)
attn = causal_GQA_softmax(q, k, v)   # 7 q-heads per kv-head; scale 1/sqrt(64)
h  = h + (attn·o_projᵀ)
# --- MLP (SwiGLU) ---
n  = RMSNorm(h, post_attention_layernorm.weight)
h  = h + down_proj( silu(gate_proj·nᵀ) * (up_proj·nᵀ) )
x  = h
```

Model head:
```
embed_tokens[token_ids]  → x          # [T, 896]
... 24 layers ...
x = RMSNorm(x, model.norm.weight)
logits = x · embed_tokens.weightᵀ      # tied head → [T, 151936]
```

Kernels to implement as **Mojo GPU (Metal) kernels**, f32: RMSNorm, dense matmul
(`y=xWᵀ+b`), RoPE (rotate-half on 64-dim heads), causal scaled-dot-product
attention with GQA head-grouping + softmax, SiLU, elementwise mul/add, and an
argmax over the vocab. max-backend's ladder already validated hand-written Mojo
Metal matmul + the MLP/norm/softmax primitives on the M4 (rungs 1–5); the kernel
we must get right *ourselves* is the one MAX got wrong — **fused causal GQA
attention + RoPE** (its rung-6 suspect). Start with a simple, correct kernel per
op (one matmul, one attention pass); fusion and tiling are optimizations, not a
v1 requirement.

## 4. Numerics

- **Storage:** bf16 (as shipped). **Compute:** float32 on the GPU — bf16 is
  widened to f32 on load (uploaded to device as f32) for v1. The CPU oracle runs
  MAX at float32 (its proven-coherent path, max-backend §8 #2), so diffing our
  **GPU/f32** against **MAX-CPU/f32** isolates any fault to our kernels, not a
  dtype gap.
- **Accumulation:** f32 throughout. Softmax in f32 with the standard max-subtract
  for stability.
- A **bf16-native GPU** compute path (half the device footprint and bandwidth) is
  the main later optimization, gated on matching the f32 reference within
  tolerance — max-backend rung 5 found Metal bf16 matmul/MLP bit-identical to CPU,
  so the precision headroom is there.

## 5. Components

```
   prompt (messages)                                   text out
        │                                                 ▲
        ▼                                                 │
  ┌───────────┐   ids   ┌──────────────┐  logits  ┌──────────────┐
  │ minja2    │────────►│ Tokenizer    │─────────►│ Qwen2 model  │
  │ (template)│         │ (BPE encode) │          │ 24 layers,KV │
  │  (host)   │         │   (host)     │          │ GPU / Metal  │
                              ▲                    └──────┬───────┘
                              │ decode(ids→text)          │ argmax / sample
                              └───────────────────────────┘
                                       decode loop
        ┌──────────────────────────────────────────────────────┐
        │ Weight loader: mmap safetensors, bf16→f32, → GPU device │
        └──────────────────────────────────────────────────────┘
```

### 5.1 Weight loader
Parse the safetensors header (8-byte little-endian length + JSON), `mmap` the
blob, and expose each named tensor as a typed view. Names/shapes are hardcoded
to §2's table; the loader asserts they match. bf16 is upcast to f32 and the
tensors are **uploaded to GPU device buffers once at load** (per §4); the forward
pass reads only device memory. The mmap'd host copy can be released after upload.

### 5.2 Tokenizer (byte-level BPE)
`Qwen2Tokenizer` is GPT-2-style byte-level BPE: a byte→unicode mapping, a
`vocab.json` (token→id), and ranked `merges.txt`. Encode = byte-map → greedy
lowest-rank merges → ids; decode = ids → tokens → byte-unmap → UTF-8. Special
tokens (`<|im_start|>` etc.) are matched verbatim before BPE. This is a real
component to build in Mojo (max-backend let MAX own it); pre-tokenization uses
Qwen's GPT-2 regex split.

### 5.3 Chat templating (minja2)
The prompt string is produced by [`../minja2`](../minja2/docs/requirements.md)
rendering the Qwen2.5 `chat_template` (shipped in `tokenizer_config.json`, also
in `minja2/tests/fixtures/chat_templates/Qwen__Qwen2.5-0.5B-Instruct.jinja`) with
`add_generation_prompt=true`. minja2 already targets byte-identical output vs.
`transformers.apply_chat_template`, so the prompt string is trusted; we feed it
to the tokenizer (§5.2). No BOS is added (`bos_token: null`).

### 5.4 KV cache
Single-sequence cache of per-layer K and V (`[num_kv_heads=2, T, head_dim=64]`),
**resident in GPU device memory**, grown one position per decode step. Prefill
processes the whole prompt in one pass and fills positions `0..P-1`; each
subsequent step appends one. Sized to `max_position_embeddings` or a CLI cap. No
paging, no eviction, no SSD tier (those are max-backend control-plane concerns,
out of scope).

### 5.5 Decode loop
1. Render prompt (§5.3) → encode (§5.2) → `ids` (host).
2. **Prefill:** forward all `ids` on the GPU, fill device KV, take logits at the
   last position.
3. **Greedy:** `argmax(logits)` runs on-device; only the resulting **token id**
   crosses back to the host. Append it to the KV at the next position; repeat from
   the new token's logits. (Tokenization/templating stay on the host; only ids and
   the final logits/argmax cross the boundary.)
4. Stop on `<|im_end|>` (151645) or `<|endoftext|>` (151643), or at `max_tokens`.
5. Decode generated ids → text (host).

### 5.6 Sampling (fast-follow, not v1)
v1 is **greedy/argmax** — deterministic, so it diffs cleanly against the
reference (§7). The model's `generation_config.json` (temperature 0.7, top_p 0.8,
top_k 20, repetition_penalty 1.1) defines the eventual sampling target; add
temperature → top-k → top-p → repetition penalty once greedy parity holds. (Note
max-backend could only honor `max_tokens` because MAX's offline `LLM` hard-codes
sampling — its §9 #1, §10 #1; here we own the sampler, so full per-request
sampling is *available* to us, just sequenced after correctness.)

### 5.7 CLI / entrypoint
A thin `main.mojo`: `mojo run main.mojo "<prompt>"` → renders, generates, prints.
Flags for `--max-tokens`, model path, and (later) sampling. The OpenAI/Anthropic
HTTP surface is ported from max-backend in a later phase (§6).

## 6. Phased roadmap

The ordering is deliberately **risk-first**: prove the one kernel that can sink
the GPU-only thesis (attention + RoPE — the exact piece MAX got wrong on Metal,
max-backend §8 #2 rung 6) *before* investing in the loader, tokenizer, and the
known-reachable kernels around it. Attention is a pure function of (Q, K, V,
positions), so it is verifiable in isolation with no other component built.

| Phase | Delivers |
|---|---|
| **0 — Scaffold** *(✅ done)* | pixi env (Mojo + GPU/Metal, nightly), repo layout, this doc. **A Mojo GPU hello kernel runs on the M4** — re-confirms max-backend rung 1 (hand-written Mojo Metal kernels execute *and* compute correctly) on a clean env, before betting the project on it. `pixi run gpu-hello` ✅. |
| **1 — Attention + RoPE spike (the go/no-go gate)** *(✅ done — passed)* | The risky kernel, first and alone. A Mojo Metal kernel for RoPE (split-half, θ=1e6) + causal GQA attention at the real Qwen2 dims (head_dim 64, 14:2 heads), **diffed against a from-scratch NumPy reference** on both synthetic inputs and **captured-real layer Q/K/V fixtures from the model run** (so it is tested on realistic magnitudes, not just random). **Result: GPU output matches the reference to ≤ 8.4e-6 abs on synthetic + real layer-0 + real layer-23 — the kernel MAX got wrong on Metal, we got right.** See §11. |
| **2 — Remaining kernels + loader + tokenizer** *(✅ done)* | matmul/RMSNorm/SwiGLU GPU kernels (matmul bit-exact, others ≤ 4.5e-6 vs NumPy on real layer-0); safetensors loader (bf16→f32, bit-exact vs torch on 6 real tensors); byte-level BPE encode/decode **byte-identical to transformers** on the English corpus (ASCII-correct pretokenizer; non-ASCII `\p{L}`/`\p{N}` deferred). See §11 #4–6. |
| **3 — Forward pass** *(✅ done)* | embed → 24 layers → final norm → tied head on GPU/f32, real weights loaded from safetensors. **Greedy next-token argmax agrees with HF/CPU** (785, `'The'`); per-layer hidden drift ≤ 2.5e-3 over 24 layers (pure f32 accumulation). Per-layer comparison gives the layer-bisection (max-backend §8 #2 rung 6). See §11 #7. |
| **4 — Decode loop + KV cache** *(✅ done)* | Prefill + incremental greedy decode on-device (per-layer KV cache, RoPE-by-cache-row so a step is O(positions)), EOS handling. **Token-for-token greedy parity with HF**: "What is the capital of France?" → `The capital of France is Paris.` + EOS, all 8 ids identical. See §11 #8. |
| **5 — Sampling** | temperature/top-k/top-p/repetition-penalty per `generation_config.json` (§5.6). |
| **6 — Serve** | Port max-backend's flare HTTP layer (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) onto this engine — now a *pure-Mojo, GPU* request path end to end, **no Python/MAX at runtime**. |
| **Later — bf16 GPU** | bf16-native device compute (§4) once it matches the f32 reference. |
| **Later — Generalize** | Replace the hardcoded constants with a parsed `config.json` + a weight-name scheme; add a second architecture. The hardcoding in §2 is the seam this phase widens. |

## 7. Conformance methodology

Same philosophy as minja2's byte-equality (its §9): **one authoritative signal**,
checked against a trusted oracle.

- **Two oracles, by altitude (both CPU/f32, numerically equivalent):**
  - **Per-kernel → HF transformers (CPU/f32).** For an isolated op, HF eager
    attention is the better reference: it exposes per-layer Q/K/V and the
    post-attention context through forward hooks (MAX's compiled graph does not),
    and we own a from-scratch NumPy reference of the math that is first
    cross-checked against HF on real activations. Used in Phase 1 (§11 #2).
  - **Whole-model → MAX running on CPU/float32**, greedy (`do_sample=false`) —
    the path max-backend proved coherent (its §8 #2). For end-to-end logits and
    token parity (Phase 3–4), where per-layer hooks aren't needed and agreement
    with a full production engine is the point.
  Our **GPU/f32** output is diffed against these, so a mismatch points at our
  Metal kernels, not a dtype or a wrong reference.
- **Signals, in order of strictness:**
  1. **Tokenizer:** our `encode(prompt)` == reference token ids; `decode` round-trips.
     **Done — byte-identical on the English corpus (§11 #6).**
  2. **Per-kernel:** each GPU op (matmul, norm, SiLU, MLP, RoPE, attention) diffed
     against the reference for the same inputs — the rung-by-rung ladder
     max-backend built (its §8 #2 rungs 1–6), now applied to *our* kernels.
     **Attention+RoPE done — ≤ 8.4e-6 (§11 #1).**
  3. **Logits:** GPU last-position logits ≈ reference within f32 tolerance, and
     `argmax` agrees exactly. Bisect by layer count to localize any divergence
     (max-backend §8 #2 rung 6). **Done — argmax agrees, per-layer drift ≤ 2.5e-3
     (§11 #7).**
  4. **Generation:** greedy continuation is **token-for-token identical** to
     the reference for N tokens on a fixed prompt set. **Done — 8/8 ids match HF
     incl. EOS (§11 #8).**
- **Corpus:** a handful of fixed prompts (single-turn, multi-turn,
  system+user) rendered through minja2, mirroring minja2's context corpus.
- Because these checks need the **GPU + weights + MAX**, they live under
  `tests/manual/` and run via pixi tasks (e.g. `pixi run gpu-check`), per repo
  `CLAUDE.md` — not in the pure-Python/Mojo unittest suite.

The MAX oracle is a **build/test-time dependency only**; it never ships in the
runtime path.

## 8. Build & dependency management (pixi)

- `pixi.toml`: Mojo toolchain from Modular's channel, with the GPU/Metal backend
  (the same toolchain max-backend's isolation ladder ran Mojo GPU kernels on).
  **No `max` runtime dependency** for the engine itself (the whole point). A
  separate `dev`/`test` feature pulls `max` (the CPU oracle, §7) — and/or
  `transformers` + `torch` — plus `huggingface_hub` to fetch weights.
- minja2 is consumed as a pixi git dependency (or path dep during co-development),
  like max-backend pulls flare.
- **Env caveat (inherited from max-backend §8 #5):** MAX bakes *absolute* paths
  into its env at install time, so the test/oracle env is not relocatable —
  moving the repo breaks the kernel-package resolution; the fix is
  `rm -rf .pixi && pixi install`. This only affects the MAX *oracle* feature, not
  the runtime engine.
- The model is read from the HF cache
  (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`) or a path flag.
- `pixi run` tasks: `build`, `gen` (run a prompt on the GPU), `test` (pure
  unit tests), `gpu-check` (the GPU-vs-MAX-CPU conformance ladder, §7).

## 9. Key risks / open questions

1. **byte-level BPE exactness.** Tokenizer mismatches shift the prompt off the
   training distribution silently. Phase-1's id-equality test (§7 #1) is the
   guard; the GPT-2 pre-tokenization regex and the byte↔unicode table are the
   fiddly parts.
2. **Numeric drift across 24 layers.** Small per-op f32 differences can
   accumulate; greedy `argmax` is robust to tiny logit deltas but ties/near-ties
   can flip a token. Layer-bisection (§7 #2) localizes it; bf16-vs-f32 reference
   choice matters (we diff f32-vs-f32).
3. **RoPE / attention correctness — the load-bearing risk.** This is *exactly*
   the kernel where max-backend localized MAX's own GPU failure (its §8 #2 rung
   6). We are rewriting it by hand on the same Metal backend, so the rotate-half
   convention, the f32 inv-freq, and the GQA head-grouping (7:1) are where a
   subtly wrong-but-plausible result hides. Per-kernel isolation against MAX-CPU
   (§7 signal 2) is the mitigation; it is the **first kernel built (Phase 1)** —
   the whole roadmap is ordered to hit this risk before anything else.
4. **Mojo GPU / Metal maturity.** Mojo's native Metal backend is early
   (max-backend §2). The reassurance: max-backend's ladder ran hand-written Mojo
   GPU kernels *correctly* on this M4 (rungs 1–2), so the primitives we need are
   reachable — but stdlib GPU coverage (reductions, atomics, layout helpers) may
   have gaps we hit. Surface them early in Phase 0/1.
5. **GPU performance.** A naive kernel-per-op with host syncs each step may be
   slow; acceptable for v1 (correctness first), revisit with fusion/tiling and
   keeping the decode loop on-device (§5.5).
6. **Mojo language churn.** Mojo is pre-1.0; the std (esp. `String`, SIMD, GPU,
   file/mmap APIs) shifts between nightlies. Pin the toolchain in `pixi.toml`.

## 10. Relationship to max-backend

max-backend is the working, shipped reference: a pure-Mojo flare API layer in
front of MAX inference driven over Python interop (its §3, §8). It proved the
five user features and the API shape, but its inference is **not** pure Mojo —
that was the conscious deviation forced by the missing MAX Mojo Graph API (its
§7 #1, §8 #1). This repo closes exactly that gap for one model — and goes
straight to the GPU, the path max-backend could *not* use because MAX's Metal
kernels were broken (its §8 #2), but which its isolation ladder showed is open to
hand-written Mojo kernels. When the engine here is proven (Phase 4) and the
server is ported (Phase 6), the millrace request path is pure Mojo from socket to
logits, running on the Apple Silicon GPU, with **no Python and no MAX at
runtime** — the stance max-backend set out with and had to defer. MAX survives
only as the CPU oracle that tells us our GPU kernels are right.

## 11. Phase-1 findings (empirical)

Run on this machine (osx-arm64, Apple M4, Mojo 1.0.0b2 nightly).

1. **Hand-written Mojo Metal RoPE + causal GQA attention is correct — the
   go/no-go gate passed.** The spike kernel (`src/attention_spike.mojo`: one GPU
   thread per (query position, query head), split-half RoPE at θ=1e6, online
   softmax, GQA 14:2) matches a from-scratch NumPy reference to **≤ 8.4e-6
   absolute** on three fixtures: synthetic random Q/K/V, and **real** Q/K/V
   captured from Qwen2.5-0.5B layer 0 and layer 23. This is the exact kernel
   class MAX's Metal backend got *wrong* (max-backend §8 #2 rung 6), so the
   load-bearing risk (§9 #3) is retired: writing our own attention/RoPE on Metal
   is coherent. Reproduce: `pixi run attn-capture` then `pixi run attn-kernel`.
2. **The oracle is HF transformers (CPU/f32), not MAX — a deliberate, equivalent
   substitution for *this* spike.** ARCHITECTURE called for MAX-CPU (§7), but for
   a single-kernel check HF eager attention is the better oracle: it exposes the
   per-layer Q/K/V and post-attention context through forward hooks (MAX's
   compiled graph does not), and CPU/f32 numerics are equivalent to MAX-CPU/f32.
   The NumPy reference is the actual comparison target; it was first
   **cross-checked against HF's own attention output to ~1e-6** on the captured
   activations (`capture.py`), so "matches the reference" means "matches a
   trusted real implementation on real data". MAX-CPU remains the **whole-model**
   oracle from Phase 3 (logits, token parity), where per-layer hooks aren't
   needed and end-to-end MAX agreement is what matters. §7 updated to reflect
   this split.
3. **Mojo nightly API notes (1.0.0b2).** `def` does **not** imply `raises` (add
   it explicitly); `fn` is removed. Runtime-extent layouts use `row_major(n)`
   with a plain `Int` (not `Idx(n)`). `open()` takes `"r"`/`"w"`, not `"rb"`;
   `FileHandle.read_bytes()` returns the raw bytes regardless. List values must
   be transferred out of functions with `^`. A `TileTensor` over a borrowed
   `DeviceBuffer` is read-only (`mut=False`) and won't bind to a kernel param
   typed `MutAnyOrigin` — pass buffers a helper writes through as `mut`. These
   are captured by the installed `mojo-syntax` / `mojo-gpu-fundamentals` skills.

### Phase-2 progress

4. **Building-block kernels match the reference (Phase 2, partial).** Mojo Metal
   `matmul (+bias)`, `RMSNorm`, and the composed `SwiGLU MLP` (matmul → silu·mul
   → matmul) agree with the NumPy references — **matmul bit-exact (0.0)**, RMSNorm
   ≤ 1.8e-6, SwiGLU ≤ 4.5e-6 — on synthetic dims *and* real Qwen2 layer-0 weights
   + activations. References cross-checked against HF to ≤ 2.4e-7. With Phase-1
   attention, every compute kernel the forward pass needs is now verified on
   Metal. `pixi run kernels-capture` then `pixi run kernels-spike`.
5. **Safetensors loader reads the real checkpoint (Phase 2).** A hand-written
   Mojo header parser (8-byte LE length + JSON) locates tensors and decodes bf16
   → f32 (`f32_bits = u16 << 16`, exact). Verified **bit-exact (0.0)** vs torch
   on 6 real Qwen2 tensors (embeddings, norms, q-proj weight+bias, deep-layer
   MLP) — dtype `BF16`, shapes, and first elements all match.
   `pixi run loader-capture` then `pixi run loader-spike`.
6. **Byte-level BPE tokenizer is byte-identical to transformers (Phase 2,
   complete).** Encode *and* decode match on an 8-prompt English corpus (the full
   chat template with special tokens + newlines, multi-space, digits+punctuation,
   newline runs). Trick: the byte↔unicode map is a bijection and every BPE symbol
   is a vocab token, so the whole tokenizer runs in **integer id-space** — no
   unicode in Mojo. The pretokenizer implements the Qwen regex ASCII-correct;
   **non-ASCII `\p{L}`/`\p{N}` is deferred** (a UTF-8 letter run would mis-split)
   — fine for the English corpus, to revisit before multilingual input.
   `pixi run tok-capture` then `pixi run tok-spike`. **Phase 2 is complete; every
   piece the forward pass needs is verified.**

### Phase-3 result

7. **The whole model runs on the GPU and matches HF (Phase 3, ✅).** The full
   Qwen2.5-0.5B forward pass — real safetensors weights (bf16→f32, bulk-loaded
   via host buffer + `memcpy` + a GPU convert kernel) → embed → 24 decoder layers
   → final RMSNorm → tied LM head — runs entirely on the M4 in float32. Verified
   against HF/CPU/f32 (`forward-capture`): embedding **exact**, per-layer
   residual-stream hidden drift **≤ 2.5e-3** across all 24 layers (f32
   accumulation, monotone and tiny — no bisection failure), final-norm 1.4e-3,
   and the **greedy next-token argmax agrees exactly** (785 → `'The'`). Kernels
   live in the reusable `src/kernels.mojo`. `pixi run forward-capture` then
   `pixi run forward-spike`. **The GPU-only native-Mojo Qwen2 is real; only the
   decode loop (Phase 4) stands between this and generated text.**

### Phase-4 result

8. **The model generates correct text, token-for-token with HF (Phase 4, ✅).**
   A greedy decode loop with a per-layer KV cache (raw K/V cached at row = absolute
   position; `attn_cached_kernel` applies RoPE by row, so each decode step costs
   O(positions), not O(T²)) runs prefill → incremental decode → EOS stop entirely
   on the GPU. For "What is the capital of France?" it produces
   `The capital of France is Paris.` + `<|im_end|>` — **all 8 token ids identical
   to HF greedy generation** (`[785, 6722, 315, 9625, 374, 12095, 13, 151645]`).
   `pixi run generate-capture` then `pixi run generate-spike`. **End to end, the
   pure-Mojo GPU engine reproduces the reference — matching logits became matching
   text.** Remaining: per-request sampling (Phase 5) and the flare HTTP server
   (Phase 6); the inference core is done.
