"""Gate: the decode-wedge tripwire (pure, no GPU/net).

Run: `mojo run -I src tests/gates/test_decode_health.mojo`

Verifies the health cell that backs `/v1/status`'s `decode_healthy` field: a real
wedge (~0.3 tok/s) trips it, a healthy rate (~10 tok/s) clears it, tiny
generations are ignored as launch-latency noise, and the status JSON fragment is
well-formed. This is the check that would have caught the demo's silent Metal
decode wedge (prefill + /v1/models stayed fast while decode collapsed).
"""

from runtime.decode_health import (
    record_decode,
    decode_mtps,
    decode_healthy,
    decode_status_fields,
    WEDGE_FLOOR_TPS,
    MIN_TOKENS_FOR_SAMPLE,
)


def expect(cond: Bool, msg: String, mut ok: Bool):
    print("  [" + ("PASS" if cond else "FAIL") + "] " + msg)
    if not cond:
        ok = False


def main() raises:
    var ok = True

    # Fresh process: no generation yet → unknown → healthy (no evidence of a
    # wedge), and the status field reports null.
    expect(decode_mtps() == -1, "unknown before any generation", ok)
    expect(decode_healthy(), "unknown is treated as healthy", ok)
    expect(
        decode_status_fields()
        == ',"decode_tok_per_s":null,"decode_healthy":true',
        "status fields: null + healthy when unknown",
        ok,
    )

    # A wedged decode (0.3 tok/s, real generation) must trip the tripwire.
    record_decode(0.3, 24)
    expect(decode_mtps() == 300, "0.3 tok/s recorded as 300 mtps", ok)
    expect(not decode_healthy(), "0.3 tok/s trips the wedge floor", ok)
    expect(
        decode_status_fields()
        == ',"decode_tok_per_s":0.3,"decode_healthy":false',
        "status fields: 0.3 + unhealthy when wedged",
        ok,
    )

    # A healthy rate clears it.
    record_decode(10.0, 24)
    expect(decode_mtps() == 10000, "10 tok/s recorded as 10000 mtps", ok)
    expect(decode_healthy(), "10 tok/s is healthy", ok)
    expect(
        decode_status_fields()
        == ',"decode_tok_per_s":10.0,"decode_healthy":true',
        "status fields: 10.0 + healthy",
        ok,
    )

    # A tiny generation (below MIN_TOKENS_FOR_SAMPLE) is launch-latency noise and
    # must NOT overwrite the last real sample — even if it looks slow.
    record_decode(0.3, MIN_TOKENS_FOR_SAMPLE - 1)
    expect(
        decode_mtps() == 10000,
        "sub-threshold generation ignored (stays 10 tok/s)",
        ok,
    )

    # Exactly at the floor is considered healthy (>= floor).
    record_decode(WEDGE_FLOOR_TPS, 24)
    expect(decode_healthy(), "exactly at the floor is healthy", ok)
    # Just under the floor is wedged.
    record_decode(WEDGE_FLOOR_TPS - 0.1, 24)
    expect(not decode_healthy(), "just under the floor is wedged", ok)

    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        raise Error("decode-health gate FAILED")
