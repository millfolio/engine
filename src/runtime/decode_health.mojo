"""Decode-health tripwire — detect a wedged Metal decode queue.

Observed failure (see `_boot_worker` in server.mojo): after an idle stretch the
Metal command queue can wedge such that DECODE collapses to ~0.3 tok/s (≈3s per
token) while everything else looks fine — prefill still completes in ~1-3s, and
`/v1/models` / `/health` answer in sub-millisecond. A liveness check sees GREEN
the entire time; only the decode THROUGHPUT reveals it. This module records the
last real decode rate so `/v1/status` can expose it and the CLI/app can say
"engine decode wedged — restart" instead of the whole stack looking healthy.

Calibration (measured on the demo mini, Qwen2.5-3B): wedged ≈ 0.3 tok/s, healthy
≈ 10-15 tok/s. A floor of `WEDGE_FLOOR_TPS` cleanly separates them.

Cross-thread discipline mirrors BootState (server.mojo): the rate is a single
aligned Int64 word (atomic load/store on arm64/x86-64), written once per
generation with a `fence()` around the store and load. Stored as milli-tok/s
(tok/s × 1000) so it stays integral; -1 means "no real generation yet".
"""

from std.atomic import fence
from std.ffi import _Global

# Below this decode rate the engine is treated as wedged. wedged≈0.3, healthy≈10+,
# so 2 tok/s is a wide, unambiguous separator (not a perf SLA — a wedge tripwire).
comptime WEDGE_FLOOR_TPS = 2.0

# Ignore tiny generations: a 1-2 token decode is dominated by launch latency and
# reads artificially low even on a healthy engine. Only samples with at least this
# many generated tokens update the health cell.
comptime MIN_TOKENS_FOR_SAMPLE = 4

comptime _UNKNOWN: Int64 = -1


def _init_mtps() -> Int64:
    """Global cell initializer: -1 until the first real generation records a rate.
    """
    return _UNKNOWN


def _cell() raises -> UnsafePointer[Int64, MutUntrackedOrigin]:
    """The process-lifetime decode-rate cell (milli-tok/s, -1 = unknown).
    Type is `_Global.get_or_create_ptr()`'s ResultType (see kernel_cache)."""
    return _Global["millfolio_decode_mtps", _init_mtps].get_or_create_ptr()


def record_decode(tps: Float64, gen_tokens: Int):
    """Record the decode throughput of a just-finished generation. No-op for
    generations shorter than `MIN_TOKENS_FOR_SAMPLE` (launch-latency noise).
    Called from the generation path (server.gen_full) right after `tps` is
    computed. Best-effort — never raises, so health-tracking can't break a
    generation."""
    if gen_tokens < MIN_TOKENS_FOR_SAMPLE:
        return
    var mtps = Int64(tps * 1000.0 + 0.5)
    try:
        fence()
        _cell()[] = mtps
    except:
        pass  # allocation failure of the global cell — health is best-effort


def decode_mtps() -> Int64:
    """Last recorded decode rate in milli-tok/s, or -1 if none yet (or on the
    unreachable cell-allocation failure)."""
    try:
        var v = _cell()[]
        fence()
        return v
    except:
        return _UNKNOWN


def decode_healthy() -> Bool:
    """Whether decode looks healthy: True when unknown (no evidence of a wedge)
    or at/above the floor; False only when a real sample fell below it."""
    var m = decode_mtps()
    if m < 0:
        return True  # no generation yet — no evidence of a problem
    return Float64(m) / 1000.0 >= WEDGE_FLOOR_TPS


def decode_status_fields() -> String:
    """The `/v1/status` JSON fragment for decode health:
    `,"decode_tok_per_s":<n|null>,"decode_healthy":<bool>`. Leading comma so it
    appends inside the existing status object."""
    var m = decode_mtps()
    var tps_field: String
    if m < 0:
        tps_field = String("null")
    else:
        # One decimal, integral math (no float formatting dependency).
        tps_field = String(m // 1000) + "." + String((m % 1000) // 100)
    var healthy = "true" if decode_healthy() else "false"
    return ',"decode_tok_per_s":' + tps_field + ',"decode_healthy":' + healthy
