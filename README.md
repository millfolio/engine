# inference-server

> Part of [**millrace**](https://millrace.me) — local-first LLM inference on Apple Silicon.

A from-scratch, **pure-Mojo** GPU inference engine for **Qwen2.5** (0.5B and 3B)
on Apple Silicon (Metal), served over an OpenAI-compatible HTTP API. Every GPU
kernel — matmul, attention, RMSNorm, RoPE, SwiGLU, the int4 dequant path — is
custom-written in Mojo (Apple's `simdgroup_matrix` units reached via AIR
`external_call`); there are **no C++ / CUDA / Metal-shader GPU dependencies**.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## How it compares

This is a learning/research engine: one language end to end, no external GPU
libraries, a small readable codebase. The mature frameworks are faster — that
gap is the interesting part, and it's documented honestly below.

**Approach**

|                     | **millrace**                          | [MLX](https://github.com/ml-explore/mlx) ([`mlx-lm`](https://github.com/ml-explore/mlx-lm)) | [Ollama](https://github.com/ollama/ollama) ([`llama.cpp`](https://github.com/ggml-org/llama.cpp)) |
|---------------------|---------------------------------------|------------------------------|-----------------------------|
| Implementation      | pure Mojo                             | C++/Metal core, Python API   | C/C++, Metal backend        |
| GPU kernels         | custom-written Mojo (Metal via AIR)   | MLX framework                | llama.cpp Metal shaders     |
| GPU dependencies    | **none**                              | MLX                          | llama.cpp                   |
| Weights             | bf16, or group-128 **int4**           | 4-bit affine (grouped)       | GGUF (Q4_K_M, …)            |
| Models              | Qwen2.5 0.5B / 3B (one build)         | many                         | many (GGUF)                 |
| API                 | OpenAI-compatible (+ prefix cache)    | OpenAI-compatible            | OpenAI-compatible           |

**Performance** — Qwen2.5-3B, all ~4-bit, Apple M4, each engine measured in
isolation (`pixi run bench`; two-point method). Lower-is-better for prefill,
higher-is-better for decode.

| metric (3B, 4-bit)            | **millrace** (int4) | MLX (4-bit) | Ollama (4-bit) |
|-------------------------------|--------------------:|------------:|---------------:|
| decode (tok/s)                |               ~18   |        52   |          47    |
| prefill, ~70-tok prompt (ms)  |              540    |       220   |         165    |
| prefill, ~1.5K-tok prompt (s) |               22    |       2.8   |         2.9    |

We're ~3× slower on decode and several× on prefill. The decode gap is **per-token
Metal dispatch overhead** (closed from ~5× to ~3× by the kernel fusions in this
repo). The prefill gap is a **fragment-ABI ceiling** — verified on the latest
nightly (`1.0.0b2.dev2026060906`) on this M4, *not* a question of intrinsic
access:

- We **do** reach the 8×8 `simdgroup_matrix` units via `external_call` (the
  shipped 32×32 kernel, ~1.1 TFLOP/s — ~4.5× a scalar matmul).
- MLX's 3–4 TFLOP/s comes from keeping **compact** fragments (2 floats/thread)
  register-resident across the K-loop. Mojo's `external_call` only exposes the
  **full** 8×8 fragment (`SIMD[f32,64]`), so register-blocking — the MLX lever —
  *spills* and runs **~8× slower** (`.scratch/simd2_gemm.mojo`: 0.14 vs 1.14
  TFLOP/s, current *and* latest nightly). So ~1.1 TFLOP/s is the best reachable
  with this ABI.
- The compact 16×16 op (returns `SIMD[f32,8]`) now **compiles** via
  `llvm_intrinsic` (the Mojo-side gate is gone), but the **M4 GPU rejects it**:
  `simdgroup_matrix<16,16x16>` needs GPUFamily10 — i.e. **M5+ silicon**
  (`.scratch/mma16_test.mojo`).

So closing the prefill gap on the M4 needs Modular to expose the **compact 8×8
fragment** representation; the 16×16 shortcut is hardware-gated to M5. Details +
raw numbers in [`bench/results/`](bench/results/).

## Prerequisites

- Apple Silicon Mac (Metal GPU).
- [pixi](https://pixi.sh) — the environment is pinned in `pixi.toml` (nightly
  Mojo + a separate `oracle` env with torch/transformers for fixture capture).

## Start the server

The server loads the tokenizer tables and the checkpoint path from captured
fixtures, so generate those once (these run in the `oracle` env and download the
HF model on first use):

```sh
pixi run -e oracle tok-capture       # tokenizer vocab/merges -> tests/fixtures/tokenizer/
pixi run -e oracle forward-capture   # checkpoint path        -> tests/fixtures/forward/meta.txt
```

Then launch the server:

```sh
pixi run serve
```

It compiles `src/server.mojo`, builds the native TLS helper (`libflare_tls.so`),
loads the weights onto the GPU, and listens on **http://127.0.0.1:8000**:

```
serving Qwen/Qwen2.5-0.5B-Instruct  (hidden=896, layers=24, heads=14/2, head_dim=64)
  prefill GEMM: simdgroup-matrix (~4.5x)
  weights: bf16
millrace serving on http://127.0.0.1:8000  (flare)
  GET  /v1/models
  POST /v1/chat/completions  (stream + non-stream)
  POST /v1/responses         (stream + non-stream)
```

Smoke-test it from another terminal:

```sh
curl -s localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}]}'
```

## Configuration

Optional config at `~/.config/millrace/config.json` (override the path with
`MILLRACE_CONFIG`), parsed with the same jinja2.mojo json the server uses for requests.
All keys are optional — see [`config.example.json`](config.example.json):

| key | default | notes / env override |
|---|---|---|
| `port` | `8000` | `MILLRACE_PORT` |
| `model` | (meta.txt fixture) | chat model — HF id or checkpoint path; below CLI arg + `$QWEN_SAFETENSORS` |
| `embed_model` | `Qwen/Qwen3-Embedding-0.6B` (from HF cache) | embedding model for `/v1/embeddings` — HF id or checkpoint path; `EMBED_SAFETENSORS` |
| `q4` | `false` | group-128 int4 projection weights; `QWEN_Q4=1` |
| `kv_budget_mb` | `8192` (8 GiB) | disk KV-cache LRU cap, in MiB |

The `embed_model` is a **secondary** model loaded alongside `model`, so one
process/port serves both `/v1/chat/completions` (the chat model) and
`/v1/embeddings` (the embedding model). If no embedding checkpoint resolves
(`$EMBED_SAFETENSORS` → `embed_model` → the default from the HF cache),
`/v1/embeddings` returns 503 and chat still works.

**Precedence: env / CLI arg > config file > built-in default.** So
`pixi run serve <model>` and the existing env vars still take priority; the file
is a default layer underneath.

## Models (0.5B / 3B)

The engine auto-detects the architecture from the checkpoint — **Qwen2.5-0.5B**
(the default) and **Qwen2.5-3B** are both supported from one build. The 0.5B and
3B share a tokenizer and chat template, so only the checkpoint changes; the loader
handles both a single `.safetensors` file and a sharded checkpoint
(`model.safetensors.index.json` + shards).

To run the larger model, download its weights and point the engine at it (an HF id
resolves to its cached snapshot directory):

```sh
pixi run -e oracle download-model -- Qwen/Qwen2.5-3B-Instruct   # download to the HF cache
pixi run serve -- Qwen/Qwen2.5-3B-Instruct                     # serve it
```

You can also set `QWEN_SAFETENSORS=<snapshot-dir>` for one run, or put the dir on
line 2 of `tests/fixtures/forward/meta.txt` to make it the default. The 3B needs
more memory (~6 GB bf16 weights) and decodes slower than the 0.5B.

## int4 quantization

Set `QWEN_Q4=1` to load the projection weights (q/k/v/o/gate/up/down) as
**group-128 int4** instead of bf16 (the embedding / LM head stays bf16):

```sh
QWEN_Q4=1 pixi run serve -- Qwen/Qwen2.5-3B-Instruct
```

On the 3B this gives **~2× faster decode GEMVs and ~4× smaller projection
weights** at coherent quality (~84% top-1 vs bf16). The startup banner reports
`weights: group-128 int4 (proj) + bf16 (embed)` and tags the model id `-int4`.
int4 only holds quality group-wise and is intended for the **3B** — the 0.5B
degrades noticeably, so keep it bf16. Validate with `pixi run q4-validate` (int4
vs bf16 agreement) and `pixi run q4-kernels` (kernel correctness + speed).

## Benchmark

`pixi run bench` measures prefill latency, decode tok/s, and cold-vs-warm prefix
reuse against any running OpenAI-compatible servers (millrace, `mlx_lm.server`,
Ollama) — see [`bench/README.md`](bench/README.md) for how to start each engine
and read the numbers, and [`bench/results/`](bench/results/) for a captured run.

## Connect OpenCode

With the server running in another terminal:

```sh
pixi run opencode                       # interactive
pixi run opencode -- run "your prompt"   # one-shot
```

The task queries the server's `/v1/models`, generates an OpenCode config that
declares a `millrace` provider (`@ai-sdk/openai-compatible`, pointed at
`http://127.0.0.1:8000/v1`) listing **exactly the model the server is serving**
(`opencode_config.py`), and points OpenCode at it via `OPENCODE_CONFIG`. So
whatever you launched `serve` with — 0.5B, `serve -- Qwen/Qwen2.5-3B-Instruct`,
or that plus `QWEN_Q4=1` — shows up in OpenCode's picker automatically. (It errors
if the server isn't up; start `serve` first.)

To point an existing OpenCode install at the server by hand, run
`python opencode_config.py http://127.0.0.1:8000/v1` and merge the `provider`
block from the file it prints into your `~/.config/opencode/opencode.json`.
