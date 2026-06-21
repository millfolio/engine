"""Gate: byte-level BPE encode/decode vs transformers (ARCHITECTURE.md §6 Phase 2, §11 #6).

Runs the tokenizer library over the prompt corpus and checks token ids match
transformers (`pixi run tok-capture`) and decode round-trips. `pixi run test-tokenizer`.
"""

from tokenizer import load_tokenizer
from testio import read_text, read_bytes_file, ints_from


def ids_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def bytes_equal(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def ids_str(a: List[Int]) -> String:
    var s = String("")
    for i in range(len(a)):
        if i > 0:
            s += " "
        s += String(a[i])
    return s^


def main() raises:
    var dir = "tests/fixtures/tokenizer/"
    var tok = load_tokenizer(dir)
    print("loaded tokenizer; running corpus")

    var lines = read_text(dir + "expected.tsv").split("\n")
    var count = Int(atol(String(lines[0]).strip()))

    var all_ok = True
    for i in range(count):
        var parts = String(lines[1 + i]).split("\t")
        var expected = List[Int]()
        if len(parts) >= 2:
            expected = ints_from(String(parts[1]))

        var buf = read_bytes_file(dir + "prompts/p" + String(i) + ".txt")
        var got = tok.encode(buf)
        var enc_ok = ids_equal(got, expected)
        var dec_ok = bytes_equal(tok.decode(expected), buf)

        var tag = "OK" if (enc_ok and dec_ok) else "FAIL"
        print("  p", i, ": encode=", enc_ok, " decode=", dec_ok, " (", len(expected), " toks) [", tag, "]", sep="")
        if not enc_ok:
            print("     expected:", ids_str(expected))
            print("     got:     ", ids_str(got))
        all_ok = all_ok and enc_ok and dec_ok

    if not all_ok:
        raise Error("tokenizer does NOT match transformers — gate FAILED")
    print("OK — Mojo byte-level BPE matches transformers (encode + decode) on the corpus")
