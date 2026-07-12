# Cross-engine local inference benchmark

`bench.py` measures **prefill latency**, **decode tok/s** (steady-state
generation), and a **cold-vs-warm prefix** test across any local
OpenAI-compatible servers — millfolio, `mlx_lm.server`, and Ollama all expose
POST `/v1/chat/completions`, so one harness drives all three identically.

**Method — the two-point trick** (engine-agnostic, robust to how a server
streams): time two non-streaming completions, `T(1)` with `max_tokens=1` and
`T(N)` with `max_tokens=N`; then prefill ≈ `T(1)` and decode tok/s ≈
`(tokens@N − tokens@1) / (T(N) − T(1))`. Differencing total latencies avoids
relying on client-side first-token timing, which is meaningless for a server
that generates fully then emits its stream in a burst. Every request gets a
unique nonce so server prompt-caches don't turn prefill into a cache hit; the
cold/warm test deliberately reuses a prefix to measure caching.

The harness does **not** start servers — launch the ones you want, then run it.
Unreachable targets are skipped, so benchmark whatever is up.

## Quick start

```sh
pixi run bench --doctor                 # which endpoints are reachable
pixi run bench                          # benchmark all reachable targets
pixi run bench -- --only millfolio,ollama-3b --repeats 7 --out bench/results/run.json
pixi run bench -- --only mlx --verbose --timeout 120   # debug a hang: logs each request
```

`--verbose` logs every request start/finish with a timestamp, so a hang shows
the last STARTED request instead of silence; `--timeout` caps each request
(default 600 s — lower it when diagnosing so a stuck request fails fast).

Besides the prompt suite and the cold/warm prefix test, each target runs a
**batch-classify** scenario — the product workload: 40 canned transaction
descriptions, chunked 10 per request under one shared instruction prefix,
answered as short per-line yes/no. It reports total time, distinct/s,
per-chunk seconds (chunk 1 pays the cold instruction prefill; later chunks
show what the engine's prefix cache is worth), and how many snippets parsed
to a yes/no answer.

## Starting each engine

One model per server instance for millfolio/MLX (Ollama serves many). Default
ports: millfolio 8000, MLX 8080, Ollama 11434.

**millfolio** (this repo):
```sh
pixi run serve                                              # 0.5B bf16
QWEN_SAFETENSORS=<path-to-Qwen2.5-3B-Instruct> QWEN_Q4=1 pixi run serve   # 3B int4
```

**MLX** (`mlx-lm`, installed in `.scratch/mlxenv`):
```sh
.scratch/mlxenv/bin/python -m mlx_lm server \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8080
.scratch/mlxenv/bin/python -m mlx_lm server \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit  --port 8080
```

**Ollama**:
```sh
ollama serve &                          # if not already running
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
```

## Reading the results

- **TTFT** rises with prompt length (prefill is compute over the whole prompt);
  **tok/s** is roughly prompt-length-independent (decode is per-token).
- **cold→warm**: a lower *warm* TTFT means the engine reused the cached
  conversation prefix instead of re-prefilling it.
- **Quantization differs by engine** and is printed per target (model id):
  millfolio 0.5B is bf16 / 3B is group-128 int4; MLX & Ollama default to ~4-bit.
  A 4-bit engine moving fewer weight bytes will show faster decode — that is a
  quantization gap, not purely an engine-efficiency gap. Compare like with like
  (e.g. millfolio-3B-int4 vs mlx/ollama 4-bit 3B).

Results JSON (with `--out`) records every median/min/max plus the resolved model
id and token-count source (`usage` if the server reports it, else streamed-delta
count) for reproducibility.
