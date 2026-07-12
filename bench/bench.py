#!/usr/bin/env python3
"""Reusable cross-engine LLM benchmark for local OpenAI-compatible servers.

Measures, per target, the two things that matter for an interactive coding agent:
  * TTFT  — time-to-first-token = prefill latency (scales with prompt length)
  * decode tok/s — steady-state generation rate (memory-bandwidth bound locally)
plus a cold-vs-warm prefix test (does a follow-up turn reuse the cached prefix?).

Method — the **two-point** trick, engine-agnostic (works whether a server streams
token-by-token like MLX/Ollama or batches-then-streams like millfolio):
  * T(1) = wall time of a non-streaming completion with max_tokens=1  ≈ prefill
  * T(N) = wall time with max_tokens=N
  * decode tok/s = (tokens@N − tokens@1) / (T(N) − T(1))
So prefill and decode are separated by *differencing total latencies*, not by
client-side stream timing (which is unreliable: a batch-then-emit server reports
a meaningless first-token time). All three expose POST /v1/chat/completions;
servers are NOT managed here — start them (see bench/README.md) and point this at
the running endpoints. Unreachable targets are skipped.

Methodology guards: N warmup runs discarded, R repeats, report median + min/max,
temperature 0 (greedy) for stable decode lengths, optional cooldown between
targets to limit M4 thermal throttling. Quantization differs across engines by
default (millfolio 0.5B bf16 / 3B int4; MLX & Ollama ~4-bit) — that mismatch is
real and is printed in the report header; read speed numbers with it in mind.

Stdlib only (urllib) so it runs under any python3. Usage:

    python3 bench/bench.py                      # all targets in targets.json
    python3 bench/bench.py --only millfolio,mlx  # subset
    python3 bench/bench.py --doctor             # just check which endpoints are up
    python3 bench/bench.py --repeats 7 --max-tokens 256 --out bench/results/run.json
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "targets.json")


# ── prompt suite ──────────────────────────────────────────────────────────────
# Varying prompt length probes prefill scaling; max_tokens bounds decode so tok/s
# is measured over a comparable window. Keep these stable across runs to compare.
_LONG_CODE = (
    "Here is a Python module:\n\n```python\n"
    + "\n".join(
        f"def feature_{i}(x, y):\n"
        f"    # combine inputs with a rolling transform\n"
        f"    acc = 0\n"
        f"    for k in range(len(x)):\n"
        f"        acc += x[k] * y[(k + {i}) % len(y)]\n"
        f"    return acc / (len(x) or 1)\n"
        for i in range(24)
    )
    + "\n```\n\nExplain what this module does and suggest two concrete improvements."
)

PROMPTS = [
    {"id": "short", "text": "What is a hash map and why are lookups O(1) on average?"},
    {
        "id": "medium",
        "text": (
            "I'm building a rate limiter for an HTTP API. Compare the token bucket "
            "and sliding window log algorithms: how each works, their memory cost, "
            "and which you'd pick for a 10k-req/s gateway. Be concise."
        ),
    },
    {"id": "long-code", "text": _LONG_CODE},
]


VERBOSE = [False]


def _log(msg):
    """Step-level progress (--verbose): timestamped, flushed immediately so a
    hang shows the LAST STARTED request, not silence."""
    if VERBOSE[0]:
        print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _complete(base_url, model, messages, max_tokens, temperature, timeout, tag=""):
    """One non-streaming completion. Returns dict: total_s, prompt_tokens,
    completion_tokens (from usage if present, else the requested max_tokens), text."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    nchars = sum(len(m.get("content") or "") for m in messages)
    _log(f"→ {tag or 'request'}: POST max_tokens={max_tokens} prompt~{nchars}ch (timeout {timeout}s)")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read().decode())
    except Exception as e:
        _log(f"✗ {tag or 'request'}: failed after {time.perf_counter() - t0:.1f}s — {e}")
        raise
    total_s = time.perf_counter() - t0
    usage = doc.get("usage") or {}
    choices = doc.get("choices") or [{}]
    text = ((choices[0].get("message") or {}).get("content")) or ""
    _log(
        f"← {tag or 'request'}: {total_s * 1000:.0f} ms  "
        f"(prompt {usage.get('prompt_tokens')} tok, completion {usage.get('completion_tokens')} tok)"
    )
    return {
        "total_s": total_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens") or max_tokens,
        "text": text,
    }


_NONCE = [0]


def _with_nonce(messages):
    """Copy messages, prepending a unique tag to the first user turn so the
    request can't hit any server-side prompt/prefix cache → true prefill compute.
    (Same-length-ish tag, so it doesn't materially change prompt length.)"""
    _NONCE[0] += 1
    tag = f"[req {time.time_ns()}-{_NONCE[0]}] "
    out = [dict(m) for m in messages]
    out[0]["content"] = tag + out[0]["content"]
    return out


def _two_point(base_url, model, messages, max_tokens, temperature, timeout, tag=""):
    """Prefill + decode via the two-point method. T(1) and T(N) get DISTINCT
    nonces so neither reuses the other's cached prefix (which would corrupt the
    decode differencing). Returns (prefill_ms, decode_tps, prompt_tokens, completion_tokens)."""
    r1 = _complete(base_url, model, _with_nonce(messages), 1, temperature, timeout, tag=tag + " T(1)")
    rn = _complete(base_url, model, _with_nonce(messages), max_tokens, temperature, timeout, tag=tag + " T(N)")
    prefill_ms = r1["total_s"] * 1000.0
    dtoks = (rn["completion_tokens"] or 0) - (r1["completion_tokens"] or 0)
    dt = rn["total_s"] - r1["total_s"]
    decode_tps = (dtoks / dt) if dt > 1e-6 and dtoks > 0 else 0.0
    return prefill_ms, decode_tps, rn["prompt_tokens"], rn["completion_tokens"]


def _resolve_model(base_url, model, timeout=5):
    if model and model != "auto":
        return model
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode())
    items = doc.get("data") or []
    if not items:
        raise RuntimeError("no models reported by " + url)
    return items[0]["id"]


def _is_up(base_url, timeout=3):
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout).read()
        return True
    except Exception:
        return False


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def bench_target(t, cfg):
    """Run the prompt suite + cold/warm test for one target."""
    base_url, label = t["base_url"], t["label"]
    model = _resolve_model(base_url, t.get("model", "auto"))
    max_tokens = cfg["max_tokens"]
    temp = cfg["temperature"]
    timeout = cfg.get("timeout", 600)
    repeats, warmup = cfg["repeats"], cfg["warmup"]
    print(f"\n== {label}  (model={model}) ==", flush=True)

    per_prompt = {}
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p["text"]}]
        for w in range(warmup):
            try:
                _two_point(base_url, model, msgs, max_tokens, temp, timeout, tag=f"{p['id']} warmup{w + 1}")
            except Exception as e:
                print(f"  [{p['id']}] warmup failed: {e}", flush=True)
        prefills, tps, ptoks = [], [], []
        for rep in range(repeats):
            try:
                pf, dtps, pt, _ = _two_point(
                    base_url, model, msgs, max_tokens, temp, timeout, tag=f"{p['id']} run{rep + 1}"
                )
                prefills.append(pf)
                tps.append(dtps)
                ptoks.append(pt)
            except Exception as e:
                print(f"  [{p['id']}] run failed: {e}", flush=True)
        pf_a, tps_a = _agg(prefills), _agg(tps)
        pt = _agg([x for x in ptoks if x is not None])
        per_prompt[p["id"]] = {
            "prefill_ms": pf_a, "decode_tps": tps_a,
            "prompt_tokens": (pt["median"] if pt else None),
        }
        if pf_a and tps_a:
            print(
                f"  {p['id']:<10} prefill {pf_a['median']:7.1f} ms "
                f"[{pf_a['min']:.0f}-{pf_a['max']:.0f}]   "
                f"decode {tps_a['median']:6.1f} tok/s "
                f"[{tps_a['min']:.1f}-{tps_a['max']:.1f}]   "
                f"(prompt~{per_prompt[p['id']]['prompt_tokens']} tok)",
                flush=True,
            )

    # cold vs warm: a 2-turn conversation; the long context is the cacheable prefix.
    cw = _cold_warm(base_url, model, max_tokens, temp, timeout, cfg["repeats"])
    per_prompt["_cold_warm"] = cw
    if cw and cw.get("cold_ms") and cw.get("warm_ms"):
        speedup = cw["cold_ms"]["median"] / cw["warm_ms"]["median"] if cw["warm_ms"]["median"] else 0
        print(
            f"  cold/warm  cold prefill {cw['cold_ms']['median']:7.1f} ms  ->  "
            f"warm prefill {cw['warm_ms']['median']:7.1f} ms  ({speedup:.2f}x prefix reuse)",
            flush=True,
        )

    # batch-classify: the PRODUCT workload — chunked yes/no over transaction
    # snippets with a shared instruction prefix (what the AI-tags classifier does).
    bc = _batch_classify(base_url, model, temp, timeout)
    per_prompt["_batch_classify"] = bc
    if bc and bc.get("total_s"):
        chunk_s = bc["chunk_s"]
        print(
            f"  batch      {bc['n_snippets']} snippets in {len(chunk_s)} chunks of "
            f"{bc['chunk_size']}: total {bc['total_s']:.1f}s  ·  "
            f"{bc['distinct_per_s']:.1f} distinct/s  ·  "
            f"chunk1 {chunk_s[0]:.1f}s → chunk{len(chunk_s)} {chunk_s[-1]:.1f}s  ·  "
            f"{bc['answered']}/{bc['n_snippets']} answered",
            flush=True,
        )
    return {"model": model, "prompts": per_prompt}


# ── batch-classify scenario ──────────────────────────────────────────────────
# 40 deterministic transaction descriptions, half grocery-ish — kept stable
# across runs so the numbers compare. This mirrors the vault's ask_local_batch
# classifier: one instruction prefix shared by every chunk (so an engine with a
# prefix cache amortizes it) + N numbered snippets per request + short per-line
# answers (decode is a few tokens per snippet, not a paragraph).
_BATCH_INSTRUCTION = (
    "You are labeling bank-transaction descriptions. For each numbered snippet "
    "answer exactly 'yes' if it is a grocery store or supermarket purchase (food "
    "shopping), else exactly 'no' (dining out, coffee, gas, and any non-grocery "
    "charge are 'no'). Reply with one line per snippet, formatted exactly as "
    "'<number>: yes' or '<number>: no', nothing else.\n\n"
)

_BATCH_SNIPPETS = [
    "WHOLE FOODS MARKET #10236 SEATTLE WA",
    "SHELL OIL 57444231 PORTLAND OR",
    "SAFEWAY STORE 1729 DEBIT PURCHASE",
    "STARBUCKS #4821 CARD PURCHASE",
    "TRADER JOES #517 POS DEBIT",
    "NETFLIX.COM MONTHLY SUBSCRIPTION",
    "KROGER #442 GROCERY BAKERY",
    "UNITED AIRLINES TICKET 016240",
    "COSTCO WHSE #0745 MEMBER PURCHASE",
    "CHIPOTLE ONLINE ORDER 8842",
    "ALBERTSONS MARKET #3310",
    "SHELL SERVICE STATION CAR WASH",
    "H MART FOOD MARKET 220",
    "AMC THEATRES 0091 CONCESSIONS",
    "SPROUTS FARMERS MKT #86",
    "UBER TRIP 55XJ2 SAN FRANCISCO",
    "ALDI 470 GROCERY DEBIT",
    "PELOTON MEMBERSHIP RENEWAL",
    "PUBLIX SUPER MARKET #1451",
    "CHEVRON 0209155 GAS PURCHASE",
    "WEGMANS FOOD MARKETS 084",
    "DUNKIN #349811 MOBILE ORDER",
    "FRED MEYER GROCERY 00071",
    "DELTA AIR BAGGAGE FEE 006",
    "LIDL US STORE 2205",
    "OLIVE GARDEN 0044 DINNER",
    "WINCO FOODS #58 POS",
    "APPLE.COM/BILL ICLOUD STORAGE",
    "PIGGLY WIGGLY 0672 GROCERY",
    "MARATHON PETRO 66011 FUEL",
    "GIANT EAGLE STORE 6521",
    "PANERA BREAD 204561 LUNCH",
    "FOOD LION #1622 DEBIT",
    "HILTON HOTELS FRONT DESK",
    "MEIJER STORE #254 GROCERY",
    "LYFT RIDE THU 10PM",
    "STOP & SHOP 0631 PURCHASE",
    "REGAL CINEMAS TICKETS 1109",
    "VONS GROCERY STORE 2118",
    "EXXONMOBIL 97514 GAS",
]

_BATCH_CHUNK = 10


def _batch_classify(base_url, model, temp, timeout):
    """Send the 40 snippets in sequential chunks of _BATCH_CHUNK, each request =
    the SAME instruction prefix + that chunk's numbered snippets. Reports total
    wall time, distinct/s, per-chunk seconds (chunk 1 pays the cold instruction
    prefill; later chunks show what the engine's prefix cache is worth), and how
    many snippets came back with a parseable yes/no line."""
    chunks = [
        _BATCH_SNIPPETS[i : i + _BATCH_CHUNK]
        for i in range(0, len(_BATCH_SNIPPETS), _BATCH_CHUNK)
    ]
    session = f"[batch {time.time_ns()}] "  # fresh cache lineage per run
    chunk_s, answered = [], 0
    t0 = time.perf_counter()
    for ci, chunk in enumerate(chunks):
        lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
        msgs = [{"role": "user", "content": session + _BATCH_INSTRUCTION + lines}]
        try:
            r = _complete(
                base_url, model, msgs, 8 * len(chunk) + 16, temp, timeout,
                tag=f"batch chunk{ci + 1}/{len(chunks)}",
            )
        except Exception as e:
            print(f"  batch chunk{ci + 1} failed: {e}", flush=True)
            return None
        chunk_s.append(r["total_s"])
        for line in (r["text"] or "").splitlines():
            token = line.split(":", 1)[-1].strip().lower().rstrip(".")
            if token in ("yes", "no"):
                answered += 1
    total_s = time.perf_counter() - t0
    n = len(_BATCH_SNIPPETS)
    return {
        "n_snippets": n,
        "chunk_size": _BATCH_CHUNK,
        "total_s": total_s,
        "distinct_per_s": n / total_s if total_s > 0 else 0.0,
        "chunk_s": chunk_s,
        "answered": answered,
    }


def _cold_warm(base_url, model, max_tokens, temp, timeout, repeats):
    """Cold = first turn over a long context (unique nonce → no stale cache hit).
    Warm = a follow-up turn appended to that conversation, so the long prefix
    should be served from cache. Reports both prefills (T(1), max_tokens=1)."""
    cold_ms, warm_ms = [], []
    base_ctx = _LONG_CODE
    for i in range(repeats):
        nonce = f"[session {time.time_ns()}-{i}] "
        turn1 = [{"role": "user", "content": nonce + base_ctx + "\n\nSummarize in one sentence."}]
        try:
            # cold prefill = T(1) of the fresh long conversation
            c = _complete(base_url, model, turn1, 1, temp, timeout, tag=f"cold/warm run{i + 1} cold")
            cold_ms.append(c["total_s"] * 1000.0)
            # produce a real assistant turn, then a short follow-up; warm prefill = T(1)
            a = _complete(base_url, model, turn1, max_tokens, temp, timeout, tag=f"cold/warm run{i + 1} gen")
            turn2 = turn1 + [
                {"role": "assistant", "content": a["text"]},
                {"role": "user", "content": "Now list two risks in two bullets."},
            ]
            w = _complete(base_url, model, turn2, 1, temp, timeout, tag=f"cold/warm run{i + 1} warm")
            warm_ms.append(w["total_s"] * 1000.0)
        except Exception as e:
            print(f"  cold/warm run failed: {e}", flush=True)
    return {"cold_ms": _agg(cold_ms), "warm_ms": _agg(warm_ms)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--only", default="", help="comma-separated target labels to include")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--cooldown", type=float, help="seconds to sleep between targets")
    ap.add_argument("--doctor", action="store_true", help="only report which targets are reachable")
    ap.add_argument("--out", default="", help="write JSON results to this path")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log every request start/finish (a hang shows the last started request)")
    ap.add_argument("--timeout", type=float,
                    help="per-request timeout in seconds (default from config, 600)")
    args = ap.parse_args()
    VERBOSE[0] = args.verbose

    with open(args.config) as f:
        cfg = json.load(f)
    for k, v in (("repeats", args.repeats), ("warmup", args.warmup), ("max_tokens", args.max_tokens)):
        if v is not None:
            cfg[k] = v
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("timeout", 600)
    if args.timeout is not None:
        cfg["timeout"] = args.timeout
    if args.cooldown is not None:
        cfg["cooldown"] = args.cooldown

    targets = cfg["targets"]
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        targets = [t for t in targets if t["label"] in want]

    print("checking endpoints…", flush=True)
    up = []
    for t in targets:
        ok = _is_up(t["base_url"])
        print(f"  {'UP  ' if ok else 'DOWN'}  {t['label']:<18} {t['base_url']}", flush=True)
        if ok:
            up.append(t)
    if args.doctor:
        return
    if not up:
        print("\nno reachable targets — start the servers (see bench/README.md)", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nconfig: repeats={cfg['repeats']} warmup={cfg['warmup']} "
        f"max_tokens={cfg['max_tokens']} temp={cfg['temperature']}"
    )
    print("NOTE: quantization differs by engine (see each model id) — compare with that in mind.")

    results = {}
    for i, t in enumerate(up):
        results[t["label"]] = bench_target(t, cfg)
        if cfg.get("cooldown") and i < len(up) - 1:
            time.sleep(cfg["cooldown"])

    _print_summary(results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": cfg, "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


def _print_summary(results):
    print("\n" + "=" * 78)
    print("SUMMARY  (median; prefill = T(1) latency, tok/s = decode rate)")
    print("=" * 78)
    for pid in [p["id"] for p in PROMPTS]:
        print(f"\n  prompt: {pid}")
        print(f"    {'engine':<20} {'model':<34} {'prefill ms':>10} {'tok/s':>8}")
        for label, r in results.items():
            pp = r["prompts"].get(pid, {})
            pf = pp.get("prefill_ms")
            tps = pp.get("decode_tps")
            model = (r["model"] or "")[:32]
            pf_s = f"{pf['median']:.1f}" if pf else "-"
            tps_s = f"{tps['median']:.1f}" if tps else "-"
            print(f"    {label:<20} {model:<34} {pf_s:>10} {tps_s:>8}")
    print(f"\n  batch-classify (the product workload: chunked yes/no, shared instruction prefix):")
    print(f"    {'engine':<20} {'total s':>8} {'distinct/s':>11} {'chunk1 s':>9} {'chunkN s':>9} {'answered':>9}")
    for label, r in results.items():
        bc = r["prompts"].get("_batch_classify")
        if bc and bc.get("total_s"):
            print(
                f"    {label:<20} {bc['total_s']:>8.1f} {bc['distinct_per_s']:>11.1f} "
                f"{bc['chunk_s'][0]:>9.1f} {bc['chunk_s'][-1]:>9.1f} "
                f"{str(bc['answered']) + '/' + str(bc['n_snippets']):>9}"
            )

    print(f"\n  cold->warm prefix reuse (TTFT ms, lower warm = caching works):")
    print(f"    {'engine':<20} {'cold':>9} {'warm':>9} {'reuse':>7}")
    for label, r in results.items():
        cw = r["prompts"].get("_cold_warm", {})
        c, w = cw.get("cold_ms"), cw.get("warm_ms")
        if c and w:
            sp = c["median"] / w["median"] if w["median"] else 0
            print(f"    {label:<20} {c['median']:>9.1f} {w['median']:>9.1f} {sp:>6.2f}x")
        else:
            print(f"    {label:<20} {'-':>9} {'-':>9} {'-':>7}")


if __name__ == "__main__":
    main()
