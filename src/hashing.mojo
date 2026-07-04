"""SHA-256 (FIPS 180-4) + HuggingFace tree-API LFS-oid extraction.

The weights downloader (`src/download.mojo`) uses these to verify the integrity
of every LFS file (safetensors shard) it fetches: HuggingFace exposes each
git-LFS file's sha256 as the `lfs.oid` in the model tree API, and we recompute
the sha256 of the downloaded bytes and compare. A `Content-Length` match only
proves the transfer wasn't truncated — it does NOT detect a tampered or poisoned
weight (safetensors carry no embedded signature).

Everything here is **pure Mojo — no FFI, no subprocess, no new build inputs.**
That is deliberate: the install-time build compiles `src/download.mojo` with only
`-I ../flare` (see `vault/cli` Bootstrapper), so pulling in flare's OpenSSL-FFI
HMAC helper or a JSON library would change the install path and risk breaking
`mill install` for every user. A dependency-free check also can't be subverted by
a tampered helper binary.

Public API:
  - `sha256_hex(data) -> String`     — lowercase 64-char hex digest.
  - `hf_lfs_sha256(tree, name) -> String` — the `lfs.oid` (sha256) for the file
        named `name` in a HuggingFace tree-API JSON body, or "" if that file is
        absent or is a non-LFS blob (whose `oid` is a git SHA-1, not a sha256).
"""

from std.collections import List


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 (FIPS 180-4). Streaming over the input Span so a multi-GB safetensors
# shard is hashed WITHOUT copying it — only the <=128-byte tail block is
# materialised. UInt32 arithmetic wraps mod 2^32 (two's complement), which is
# exactly the modular addition SHA-256 specifies.
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
def _rotr(x: UInt32, n: Int) -> UInt32:
    """Rotate `x` right by `n` bits (0 < n < 32)."""
    return (x >> UInt32(n)) | (x << UInt32(32 - n))


def _sha256_k() -> List[UInt32]:
    """The 64 SHA-256 round constants (first 32 bits of the fractional parts of
    the cube roots of the first 64 primes)."""
    var k: List[UInt32] = [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ]
    return k^


def _compress(
    mut h: List[UInt32], data: Span[UInt8, _], off: Int, k: List[UInt32]
):
    """Fold one 64-byte block (at `data[off:off+64]`) into the state `h`."""
    var w = InlineArray[UInt32, 64](fill=UInt32(0))
    for t in range(16):
        var j = off + 4 * t
        w[t] = (
            (UInt32(data[j]) << 24)
            | (UInt32(data[j + 1]) << 16)
            | (UInt32(data[j + 2]) << 8)
            | UInt32(data[j + 3])
        )
    for t in range(16, 64):
        var s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
        var s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w[t] = w[t - 16] + s0 + w[t - 7] + s1

    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]

    for t in range(64):
        var S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        var ch = (e & f) ^ (~e & g)
        var t1 = hh + S1 + ch + k[t] + w[t]
        var S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        var maj = (a & b) ^ (a & c) ^ (b & c)
        var t2 = S0 + maj
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2

    h[0] += a
    h[1] += b
    h[2] += c
    h[3] += d
    h[4] += e
    h[5] += f
    h[6] += g
    h[7] += hh


def sha256(data: Span[UInt8, _]) -> List[UInt8]:
    """Return the 32-byte SHA-256 digest of `data`."""
    var h: List[UInt32] = [
        UInt32(0x6A09E667),
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]
    var k = _sha256_k()
    var n = len(data)

    # Full 64-byte blocks straight out of the input (no copy).
    var full = n // 64
    for blk in range(full):
        _compress(h, data, blk * 64, k)

    # Tail: remaining bytes + 0x80 + zero pad + 64-bit big-endian bit length,
    # materialised into a small (64- or 128-byte) buffer.
    var rem = n - full * 64
    var tail = List[UInt8]()
    for i in range(rem):
        tail.append(data[full * 64 + i])
    tail.append(0x80)
    while (len(tail) % 64) != 56:
        tail.append(0)
    var bitlen = UInt64(n) * 8
    for i in range(8):
        tail.append(UInt8((bitlen >> UInt64(56 - 8 * i)) & 0xFF))
    var tspan = Span(tail)
    var toff = 0
    while toff < len(tail):
        _compress(h, tspan, toff, k)
        toff += 64

    var out = List[UInt8]()
    for i in range(8):
        out.append(UInt8((h[i] >> 24) & 0xFF))
        out.append(UInt8((h[i] >> 16) & 0xFF))
        out.append(UInt8((h[i] >> 8) & 0xFF))
        out.append(UInt8(h[i] & 0xFF))
    return out^


def sha256_hex(data: Span[UInt8, _]) -> String:
    """Return the lowercase 64-character hex SHA-256 digest of `data`."""
    var digest = sha256(data)
    comptime HEX = "0123456789abcdef"
    var hb = HEX.as_bytes()
    var out = List[UInt8]()
    for i in range(len(digest)):
        var b = Int(digest[i])
        out.append(hb[b >> 4])
        out.append(hb[b & 0xF])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model tree-API extraction. The body of
# `GET /api/models/<repo>/tree/<rev>` is a JSON array of file entries. An LFS
# file (e.g. a safetensors shard) looks like:
#   {"type":"file","oid":"<git-sha1>","size":N,
#    "lfs":{"oid":"<sha256>","size":N,"pointerSize":P},"xetHash":"<64hex>",
#    "path":"model.safetensors"}
# The canonical sha256 is `lfs.oid`. NOTE two look-alikes we must NOT confuse it
# with: the top-level `oid` is a 40-hex git SHA-1, and `xetHash` is ALSO 64-hex.
# So we resolve `lfs` -> `oid` structurally, not by "the 64-hex value". A non-LFS
# blob (config.json, tokenizer) has no `lfs` object at all -> we return "" and the
# caller keeps the existing size check (its content sha256 is simply not
# published by HF).
#
# A small depth- and string-aware scanner (below) avoids adding a JSON-library
# build dependency to the install path.
# ─────────────────────────────────────────────────────────────────────────────


def _is_ws(b: UInt8) -> Bool:
    return b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D


def _scan_string(t: Span[UInt8, _], at: Int) -> Int:
    """`at` points at an opening `"`. Return the index just past the closing `"`,
    honouring backslash escapes."""
    var i = at + 1
    var n = len(t)
    while i < n:
        if t[i] == UInt8(ord("\\")):
            i += 2
            continue
        if t[i] == UInt8(ord('"')):
            return i + 1
        i += 1
    return n


def _read_string(t: Span[UInt8, _], at: Int) -> String:
    """Return the (unescaped-assumed-plain) contents of the JSON string starting
    at `at` (which points at the opening `"`). Our keys/values here — path names,
    "lfs", "oid", hex digests — contain no escapes."""
    var end = _scan_string(t, at)  # past closing quote
    var out = List[UInt8]()
    for i in range(at + 1, end - 1):
        out.append(t[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _match_bracket(t: Span[UInt8, _], start: Int) -> Int:
    """`start` points at `{` or `[`. Return the index of the matching close
    bracket (string-aware), or -1."""
    var n = len(t)
    var depth = 0
    var i = start
    while i < n:
        var c = t[i]
        if c == UInt8(ord('"')):
            i = _scan_string(t, i)
            continue
        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_key(
    t: Span[UInt8, _], obj_start: Int, obj_end: Int, key: String
) -> Int:
    """Scan the TOP level of the object `t[obj_start..obj_end]` (obj_start at `{`,
    obj_end at the matching `}`) for `"key"`. Return the index of that key's value
    (first non-ws byte after the `:`), or -1. Depth-aware so nested keys don't
    match."""
    var i = obj_start + 1
    while i < obj_end:
        while i < obj_end and (_is_ws(t[i]) or t[i] == UInt8(ord(","))):
            i += 1
        if i >= obj_end or t[i] != UInt8(ord('"')):
            return -1
        var k = _read_string(t, i)
        i = _scan_string(t, i)  # past key's closing quote
        while i < obj_end and _is_ws(t[i]):
            i += 1
        if i >= obj_end or t[i] != UInt8(ord(":")):
            return -1
        i += 1
        while i < obj_end and _is_ws(t[i]):
            i += 1
        if i >= obj_end:
            return -1
        if k == key:
            return i
        # Skip this value to reach the next key.
        if t[i] == UInt8(ord('"')):
            i = _scan_string(t, i)
        elif t[i] == UInt8(ord("{")) or t[i] == UInt8(ord("[")):
            var close = _match_bracket(t, i)
            if close < 0:
                return -1
            i = close + 1
        else:
            while (
                i < obj_end
                and t[i] != UInt8(ord(","))
                and t[i] != UInt8(ord("}"))
            ):
                i += 1
    return -1


def _is_hex64(s: String) -> Bool:
    if s.byte_length() != 64:
        return False
    var b = s.as_bytes()
    for i in range(64):
        var c = b[i]
        var ok = (
            (c >= UInt8(ord("0")) and c <= UInt8(ord("9")))
            or (c >= UInt8(ord("a")) and c <= UInt8(ord("f")))
            or (c >= UInt8(ord("A")) and c <= UInt8(ord("F")))
        )
        if not ok:
            return False
    return True


def hf_lfs_sha256(tree: Span[UInt8, _], filename: String) raises -> String:
    """Return the git-LFS sha256 (`lfs.oid`) for `filename` from a HuggingFace
    tree-API JSON body, or "" if `filename` is absent or is a non-LFS blob.

    Args:
        tree: The raw bytes of `GET /api/models/<repo>/tree/<rev>`.
        filename: The top-level file path to look up (e.g. "model.safetensors").

    Returns:
        A lowercase 64-hex sha256 string, or "" when there is no LFS sha256 for
        `filename`.
    """
    var n = len(tree)
    var i = 0
    while i < n and tree[i] != UInt8(ord("[")):
        i += 1
    i += 1  # past '['
    while i < n:
        while i < n and (_is_ws(tree[i]) or tree[i] == UInt8(ord(","))):
            i += 1
        if i >= n or tree[i] == UInt8(ord("]")):
            break
        if tree[i] != UInt8(ord("{")):
            i += 1
            continue
        var es = i
        var ee = _match_bracket(tree, es)
        if ee < 0:
            break
        var pv = _find_key(tree, es, ee, "path")
        if pv >= 0 and tree[pv] == UInt8(ord('"')):
            var name = _read_string(tree, pv)
            if name == filename:
                var lv = _find_key(tree, es, ee, "lfs")
                if lv < 0 or tree[lv] != UInt8(ord("{")):
                    return ""
                var le = _match_bracket(tree, lv)
                if le < 0:
                    return ""
                var ov = _find_key(tree, lv, le, "oid")
                if ov < 0 or tree[ov] != UInt8(ord('"')):
                    return ""
                var oid = _read_string(tree, ov)
                if _is_hex64(oid):
                    return oid^
                return ""
        i = ee + 1
    return ""
