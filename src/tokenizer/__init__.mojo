"""Byte-level BPE tokenizer for Qwen2 (ARCHITECTURE.md §5.2), pure Mojo.

Encode/decode in integer token-id space (the byte↔unicode map is a bijection and
every BPE symbol is a vocab token, so no unicode is needed in Mojo). Special-token
splitting + ASCII-correct Qwen pretokenization + rank-greedy BPE. Loads the tables
`tok-capture` dumps (vocab/merges resolved to ids). Verified by test_tokenizer.mojo.

Scope: ASCII-correct; non-ASCII \\p{L}/\\p{N} pretokenization is deferred.

Two loaders build the same `Tokenizer`:
  - `load_tokenizer(dir)`         — tok-capture's resolved `.tsv` dumps (dev/tests).
  - `load_tokenizer_json(path)`   — HuggingFace `tokenizer.json` directly (what the
                                    native downloader fetches), so a freshly
                                    downloaded checkpoint serves with no tok-capture.
"""

from json import parse_json

comptime MUL = 1 << 20         # pair key = left*MUL + right (ids < 262144 for Gemma,
                               # < 151936 for Qwen; 1<<20 == 1048576 exceeds both)
comptime APOS = 39
comptime SPACE = 32
comptime USCORE = 0x2581       # U+2581 "▁" — Gemma's normalized-space marker


def is_letter(b: Int) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)

def is_digit(b: Int) -> Bool:
    return b >= 48 and b <= 57

def is_space(b: Int) -> Bool:
    return b == 32 or b == 9 or b == 10 or b == 13 or b == 11 or b == 12

def is_nl(b: Int) -> Bool:
    return b == 10 or b == 13

def lower(b: Int) -> Int:
    return b + 32 if (b >= 65 and b <= 90) else b


@fieldwise_init
struct SpMatch(Copyable, Movable):
    var id: Int
    var length: Int


@fieldwise_init
struct Tokenizer(Movable):
    var byte_id: List[Int]                 # [256] id of each single-byte token
    var merge_rank: Dict[Int, Int]         # pairkey -> rank
    var merge_id: Dict[Int, Int]           # pairkey -> merged id
    var id_to_bytes: Dict[Int, List[UInt8]]
    var sp_text: List[List[UInt8]]         # special token texts
    var sp_id: List[Int]                   # parallel ids
    var cp_to_id: Dict[Int, Int]           # codepoint -> id (Gemma single-char symbols)
    var gemma: Bool                        # True -> SentencePiece-style path (no GPT2 regex)

    def bpe(self, mut ids: List[Int]) raises:
        while len(ids) >= 2:
            var best_rank = 1 << 60
            var best_i = -1
            for i in range(len(ids) - 1):
                var key = ids[i] * MUL + ids[i + 1]
                if key in self.merge_rank:
                    var r = self.merge_rank[key]
                    if r < best_rank:
                        best_rank = r
                        best_i = i
            if best_i < 0:
                break
            var key = ids[best_i] * MUL + ids[best_i + 1]
            ids[best_i] = self.merge_id[key]
            _ = ids.pop(best_i + 1)

    def next_chunk(self, buf: List[UInt8], pos: Int, end: Int) -> Int:
        var c = Int(buf[pos])

        # rule 1: contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == APOS:
            var sufs = [String("s"), String("t"), String("re"), String("ve"),
                        String("m"), String("ll"), String("d")]
            for s in sufs:
                var sb = s.as_bytes()
                if pos + 1 + len(sb) <= end:
                    var ok = True
                    for j in range(len(sb)):
                        if lower(Int(buf[pos + 1 + j])) != Int(sb[j]):
                            ok = False
                            break
                    if ok:
                        return pos + 1 + len(sb)

        # rule 2: [^\r\n\p{L}\p{N}]? \p{L}+
        if is_letter(c):
            var q = pos
            while q < end and is_letter(Int(buf[q])):
                q += 1
            return q
        elif (not is_digit(c)) and (not is_nl(c)):
            if pos + 1 < end and is_letter(Int(buf[pos + 1])):
                var q = pos + 1
                while q < end and is_letter(Int(buf[q])):
                    q += 1
                return q

        # rule 3: \p{N} (single digit)
        if is_digit(c):
            return pos + 1

        # rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        var p = pos
        if c == SPACE:
            p = pos + 1
        if p < end:
            var d = Int(buf[p])
            if (not is_space(d)) and (not is_letter(d)) and (not is_digit(d)):
                var q = p
                while q < end:
                    var e = Int(buf[q])
                    if is_space(e) or is_letter(e) or is_digit(e):
                        break
                    q += 1
                while q < end and is_nl(Int(buf[q])):
                    q += 1
                return q

        # rule 5: \s*[\r\n]
        if is_space(c):
            var w = pos
            while w < end and is_space(Int(buf[w])):
                w += 1
            var lastnl = -1
            for i in range(pos, w):
                if is_nl(Int(buf[i])):
                    lastnl = i
            if lastnl >= 0:
                return lastnl + 1
            # rule 6/7: \s+(?!\S) then \s+
            if w == end:
                return w
            if w - 1 > pos:
                return w - 1
            return w

        return pos + 1  # non-ASCII / unhandled byte

    def encode_normal(self, buf: List[UInt8], start: Int, stop: Int, mut out: List[Int]) raises:
        var p = start
        while p < stop:
            var e = self.next_chunk(buf, p, stop)
            if e <= p:
                e = p + 1
            var ids = List[Int]()
            for i in range(p, e):
                ids.append(self.byte_id[Int(buf[i])])
            self.bpe(ids)
            for x in ids:
                out.append(x)
            p = e

    def encode_gemma_gap(self, buf: List[UInt8], start: Int, stop: Int, mut out: List[Int]) raises:
        """SentencePiece-style symbolization for a non-special gap, then BPE.

        Walk the gap codepoint-by-codepoint (UTF-8). Replace literal space (0x20)
        with U+2581 ("▁"). For each resulting codepoint: if it is a single-codepoint
        vocab symbol, emit its id; else byte-fallback — emit `byte_id[b]` for each
        UTF-8 byte of the original codepoint. Then rank-greedy BPE over the gap."""
        var ids = List[Int]()
        var p = start
        while p < stop:
            var b0 = Int(buf[p])
            # decode one UTF-8 codepoint -> (cp, byte-length)
            var cp: Int
            var nb: Int
            if b0 < 0x80:
                cp = b0; nb = 1
            elif b0 >> 5 == 0b110:
                cp = (b0 & 0x1F) << 6 | (Int(buf[p + 1]) & 0x3F); nb = 2
            elif b0 >> 4 == 0b1110:
                cp = (b0 & 0x0F) << 12 | (Int(buf[p + 1]) & 0x3F) << 6 | (Int(buf[p + 2]) & 0x3F); nb = 3
            else:
                cp = (b0 & 0x07) << 18 | (Int(buf[p + 1]) & 0x3F) << 12 | (Int(buf[p + 2]) & 0x3F) << 6 | (Int(buf[p + 3]) & 0x3F); nb = 4

            var sym = USCORE if cp == SPACE else cp   # normalizer: space -> ▁
            if sym in self.cp_to_id:
                ids.append(self.cp_to_id[sym])
            else:
                # byte_fallback: emit the <0xNN> id for each UTF-8 byte. The space
                # was replaced by ▁ which is always a vocab symbol, so byte-fallback
                # only ever sees the original (non-space) bytes here.
                for j in range(nb):
                    ids.append(self.byte_id[Int(buf[p + j])])
            p += nb
        self.bpe(ids)
        for x in ids:
            out.append(x)

    def match_special(self, buf: List[UInt8], pos: Int) raises -> SpMatch:
        var best_id = -1
        var best_len = 0
        for s in range(len(self.sp_text)):
            ref t = self.sp_text[s]
            var n = len(t)
            if n > best_len and pos + n <= len(buf):
                var ok = True
                for j in range(n):
                    if buf[pos + j] != t[j]:
                        ok = False
                        break
                if ok:
                    best_id = self.sp_id[s]
                    best_len = n
        return SpMatch(best_id, best_len)

    def encode(self, buf: List[UInt8]) raises -> List[Int]:
        var out = List[Int]()
        var n = len(buf)
        var pos = 0
        var seg = 0
        while pos < n:
            var m = self.match_special(buf, pos)
            if m.id >= 0:
                if self.gemma:
                    self.encode_gemma_gap(buf, seg, pos, out)
                else:
                    self.encode_normal(buf, seg, pos, out)
                out.append(m.id)
                pos += m.length
                seg = pos
            else:
                pos += 1
        if self.gemma:
            self.encode_gemma_gap(buf, seg, n, out)
        else:
            self.encode_normal(buf, seg, n, out)
        return out^

    def decode(self, ids: List[Int]) raises -> List[UInt8]:
        var out = List[UInt8]()
        for k in range(len(ids)):
            var id = ids[k]
            if id in self.id_to_bytes:
                for b in self.id_to_bytes[id]:
                    out.append(b)
            else:
                for s in range(len(self.sp_id)):
                    if self.sp_id[s] == id:
                        for b in self.sp_text[s]:
                            out.append(b)
                        break
        return out^


def hex_val(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    return c - 97 + 10  # a-f (Python .hex() is lowercase)

def hex_to_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i + 2 <= len(sb):
        out.append(UInt8((hex_val(Int(sb[i])) << 4) | hex_val(Int(sb[i + 1]))))
        i += 2
    return out^

def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

def _byte_decoder() -> Dict[Int, Int]:
    """GPT-2 byte-level decoder: token-string codepoint -> raw byte (inverse of
    `bytes_to_unicode`). Printable bytes map to themselves; the rest to 256+n in
    order. HF `tokenizer.json` vocab keys are encoded with this bijection."""
    var dec = Dict[Int, Int]()
    var inbs = List[Bool]()
    for _ in range(256):
        inbs.append(False)
    for v in range(33, 127):        # '!'..'~'
        inbs[v] = True; dec[v] = v
    for v in range(161, 173):       # '¡'..'¬'
        inbs[v] = True; dec[v] = v
    for v in range(174, 256):       # '®'..'ÿ'
        inbs[v] = True; dec[v] = v
    var n = 0
    for b in range(256):
        if not inbs[b]:
            dec[256 + n] = b
            n += 1
    return dec^


def _decode_token(s: String, dec: Dict[Int, Int]) raises -> List[UInt8]:
    """Decode one byte-level token string back to its raw bytes."""
    var out = List[UInt8]()
    for cp in s.codepoints():
        var v = Int(cp)
        if v in dec:
            out.append(UInt8(dec[v]))
    return out^


def load_tokenizer_json(path: String) raises -> Tokenizer:
    """Build a `Tokenizer` straight from a HuggingFace `tokenizer.json` (byte-level
    BPE, `model.type == "BPE"`): decode each vocab token to raw bytes, resolve each
    `"A B"` merge to an (id,id)->merged-id rule, and take `added_tokens` as the
    special-token table. Produces the same tables as `load_tokenizer`."""
    var root = parse_json(_read(path))
    var model = root.map_get("model").value()
    var vocab = model.map_get("vocab").value()
    var merges = model.map_get("merges").value()
    var added = root.map_get("added_tokens").value()
    var dec = _byte_decoder()

    var byte_id = List[Int]()
    for _ in range(256):
        byte_id.append(-1)
    var id_to_bytes = Dict[Int, List[UInt8]]()
    var vocab_map = Dict[String, Int]()       # token string -> id (merge resolution)

    ref vkeys = vocab.c[].keys
    ref vvals = vocab.c[].vals
    for k in range(len(vkeys)):
        var tok = vkeys[k]
        var id = vvals[k].i
        vocab_map[tok] = id
        var raw = _decode_token(tok, dec)
        if len(raw) == 1:
            byte_id[Int(raw[0])] = id
        id_to_bytes[id] = raw^

    var merge_rank = Dict[Int, Int]()
    var merge_id = Dict[Int, Int]()
    ref mvals = merges.c[].vals
    comptime VLIST = 6   # json.Value tag for a list (Qwen3+ merges are ["A","B"])
    for r in range(len(mvals)):
        # Two on-disk encodings: legacy Qwen2.5 stores each merge as the string
        # "A B" (neither side contains a literal space — 0x20 encodes as 'Ġ'); newer
        # tokenizer.json (Qwen3) stores it as a 2-element list ["A", "B"].
        var a: String
        var b: String
        ref mr = mvals[r]
        if mr.tag == VLIST:
            ref pair = mr.c[].vals
            if len(pair) != 2:
                continue
            a = String(pair[0].s)
            b = String(pair[1].s)
        else:
            var parts = String(mr.s).split(" ")
            if len(parts) != 2:
                continue
            a = String(parts[0])
            b = String(parts[1])
        var merged = a + b
        if a not in vocab_map or b not in vocab_map or merged not in vocab_map:
            continue
        var key = vocab_map[a] * MUL + vocab_map[b]
        merge_rank[key] = r
        merge_id[key] = vocab_map[merged]

    var sp_text = List[List[UInt8]]()
    var sp_id = List[Int]()
    ref avals = added.c[].vals
    for a in range(len(avals)):
        ref obj = avals[a]
        sp_id.append(obj.map_get("id").value().i)
        var tb = List[UInt8]()
        var cb = String(obj.map_get("content").value().s).as_bytes()
        for i in range(len(cb)):
            tb.append(cb[i])
        sp_text.append(tb^)

    return Tokenizer(byte_id^, merge_rank^, merge_id^, id_to_bytes^, sp_text^,
                     sp_id^, Dict[Int, Int](), False)


def _is_byte_token(t: String) -> Bool:
    """True if `t` is a literal byte-fallback token of the form `<0xNN>`."""
    var b = t.as_bytes()
    if len(b) != 6:
        return False
    return (Int(b[0]) == ord("<") and Int(b[1]) == ord("0")
            and Int(b[2]) == ord("x") and Int(b[5]) == ord(">"))


def _gemma_token_bytes(t: String) -> List[UInt8]:
    """Raw decode bytes of a Gemma vocab token: literal UTF-8 with every U+2581
    ("▁") turned back into a real space, so `decode` emits genuine spaces."""
    var out = List[UInt8]()
    for cp in t.codepoints():
        if Int(cp) == USCORE:
            out.append(UInt8(SPACE))
        else:
            _utf8_emit(Int(cp), out)
    return out^


def _utf8_emit(cp: Int, mut out: List[UInt8]):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def load_gemma_tokenizer_json(path: String) raises -> Tokenizer:
    """Build a `Tokenizer` for Gemma's SentencePiece-style BPE straight from its
    HuggingFace `tokenizer.json` (`model.type == "BPE"`, `byte_fallback == true`).

    Vocab keys are RAW UTF-8 (with "▁" for space), NOT GPT2-byte-encoded. Tables:
      - `<0xNN>` token  -> `byte_id[NN] = id`, `id_to_bytes[id] = [NN]`.
      - any other token -> `id_to_bytes[id] = utf8(replace ▁ with space)`, and if it
        is exactly one codepoint, `cp_to_id[codepoint] = id` (keyed on the ORIGINAL
        codepoint incl. U+2581, since symbolization sees ▁).
      - merges are 2-element arrays `[A, B]`; resolve to `(id_A,id_B)->id_AB`.
      - `added_tokens` -> `sp_text`/`sp_id` (matched before normalization)."""
    var root = parse_json(_read(path))
    var model = root.map_get("model").value()
    var vocab = model.map_get("vocab").value()
    var merges = model.map_get("merges").value()
    var added = root.map_get("added_tokens").value()

    var byte_id = List[Int]()
    for _ in range(256):
        byte_id.append(-1)
    var id_to_bytes = Dict[Int, List[UInt8]]()
    var cp_to_id = Dict[Int, Int]()
    var vocab_map = Dict[String, Int]()       # token string -> id (merge resolution)

    ref vkeys = vocab.c[].keys
    ref vvals = vocab.c[].vals
    for k in range(len(vkeys)):
        var tok = vkeys[k]
        var id = vvals[k].i
        vocab_map[tok] = id
        if _is_byte_token(tok):
            var b = tok.as_bytes()
            var nn = (hex_val(lower(Int(b[3]))) << 4) | hex_val(lower(Int(b[4])))
            byte_id[nn] = id
            var bl = List[UInt8]()
            bl.append(UInt8(nn))
            id_to_bytes[id] = bl^
        else:
            id_to_bytes[id] = _gemma_token_bytes(tok)
            # single-codepoint symbol -> cp_to_id (keyed on the ORIGINAL codepoint).
            var ncp = 0
            var first = 0
            for cp in tok.codepoints():
                if ncp == 0:
                    first = Int(cp)
                ncp += 1
                if ncp > 1:
                    break
            if ncp == 1:
                cp_to_id[first] = id

    var merge_rank = Dict[Int, Int]()
    var merge_id = Dict[Int, Int]()
    ref mvals = merges.c[].vals
    for r in range(len(mvals)):
        # Gemma merges are 2-element arrays [A, B]; merged token = A concatenated B.
        ref pair = mvals[r].c[].vals
        if len(pair) != 2:
            continue
        var a = pair[0].s
        var b = pair[1].s
        var merged = a + b
        if a not in vocab_map or b not in vocab_map or merged not in vocab_map:
            continue
        var key = vocab_map[a] * MUL + vocab_map[b]
        merge_rank[key] = r
        merge_id[key] = vocab_map[merged]

    var sp_text = List[List[UInt8]]()
    var sp_id = List[Int]()
    ref avals = added.c[].vals
    for a in range(len(avals)):
        ref obj = avals[a]
        sp_id.append(obj.map_get("id").value().i)
        var tb = List[UInt8]()
        var cb = String(obj.map_get("content").value().s).as_bytes()
        for i in range(len(cb)):
            tb.append(cb[i])
        sp_text.append(tb^)

    return Tokenizer(byte_id^, merge_rank^, merge_id^, id_to_bytes^, sp_text^,
                     sp_id^, cp_to_id^, True)


def load_tokenizer(dir: String) raises -> Tokenizer:
    var byte_id = List[Int]()
    for _ in range(256):
        byte_id.append(-1)
    var id_to_bytes = Dict[Int, List[UInt8]]()

    for line in _read(dir + "vocab.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split("\t")
        if len(parts) < 2:
            continue
        var id = Int(atol(String(parts[0]).strip()))
        var hx = String(parts[1]).strip()
        if hx.byte_length() == 0:
            continue
        var bytes = hex_to_bytes(String(hx))
        if len(bytes) == 1:
            byte_id[Int(bytes[0])] = id
        id_to_bytes[id] = bytes^

    var merge_rank = Dict[Int, Int]()
    var merge_id = Dict[Int, Int]()
    var rank = 0
    for line in _read(dir + "merges.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split(" ")
        if len(parts) < 3:
            continue
        var key = Int(atol(String(parts[0]).strip())) * MUL + Int(atol(String(parts[1]).strip()))
        merge_rank[key] = rank
        merge_id[key] = Int(atol(String(parts[2]).strip()))
        rank += 1

    var sp_text = List[List[UInt8]]()
    var sp_id = List[Int]()
    for line in _read(dir + "specials.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split("\t")
        if len(parts) < 2:
            continue
        sp_id.append(Int(atol(String(parts[0]).strip())))
        var tb = List[UInt8]()
        var txt = String(parts[1]).as_bytes()
        for i in range(len(txt)):
            tb.append(txt[i])
        sp_text.append(tb^)

    return Tokenizer(byte_id^, merge_rank^, merge_id^, id_to_bytes^, sp_text^,
                     sp_id^, Dict[Int, Int](), False)
