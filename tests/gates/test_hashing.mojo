"""Gate: SHA-256 + HuggingFace tree-API lfs.oid extraction (pure, no GPU/net).

Run: `mojo run -I src tests/gates/test_hashing.mojo`

Verifies the two primitives the weights downloader relies on to detect a
tampered/poisoned safetensors shard: the pure-Mojo SHA-256 (against FIPS test
vectors + block-boundary cases) and the tree-API `lfs.oid` extractor (must pick
`lfs.oid`, never the top-level git-SHA-1 `oid` nor the look-alike 64-hex
`xetHash`, and must return "" for non-LFS blobs).
"""

from hashing import sha256_hex, hf_lfs_sha256


def expect(cond: Bool, msg: String, mut ok: Bool):
    if not cond:
        print("  FAIL: ", msg)
        ok = False


def check_hex(input: String, want: String, mut ok: Bool):
    var got = sha256_hex(input.as_bytes())
    expect(
        got == want,
        "sha256('" + input + "'...) = " + got + " want " + want,
        ok,
    )


# A representative HuggingFace tree-API body. Mirrors the real shape: a non-LFS
# blob (config.json — only a git-SHA-1 `oid`), a single-file LFS weight with a
# top-level git `oid` (40-hex), an `lfs.oid` (64-hex sha256), and a DIFFERENT
# 64-hex `xetHash`, plus two LFS shards. All hashes are real values captured from
# huggingface.co so the test doubles as a fixture.
comptime TREE = String(
    '[{"type":"file","oid":"0dbb161213629a23f0fc00ef286e6b1e366d180f",'
    '"size":659,"path":"config.json"},'
    '{"type":"file","oid":"d7db405a3f0d9bf1ba5bdd4e4211db8022ebe4eb",'
    '"size":988097824,'
    '"lfs":{"oid":'
    '"fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",'
    '"size":988097824,"pointerSize":134},'
    '"xetHash":'
    '"bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7",'
    '"path":"model.safetensors"},'
    '{"type":"file","oid":"aaaa111122223333444455556666777788889999",'
    '"size":1,"lfs":{"oid":'
    '"67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",'
    '"size":1,"pointerSize":135},'
    '"path":"model-00001-of-00002.safetensors"},'
    '{"type":"file","oid":"bbbb111122223333444455556666777788889999",'
    '"size":1,"lfs":{"oid":'
    '"a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",'
    '"size":1,"pointerSize":135},'
    '"path":"model-00002-of-00002.safetensors"},'
    '{"type":"file","oid":"07bfe0640cb5a0037f9322287fbfc682806cf672",'
    '"size":7305,"path":"tokenizer_config.json"}]'
)


def main() raises:
    var ok = True

    # ── SHA-256 known-answer tests ───────────────────────────────────────────
    check_hex(
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ok,
    )
    check_hex(
        "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ok,
    )
    # 56 bytes: forces the two-block tail (rem == 56, padding overflows a block).
    check_hex(
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ok,
    )
    # 64 bytes: one full block, then a single all-padding tail block.
    check_hex(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
        ok,
    )
    # 85 bytes: multiple blocks (one full + a partial tail).
    check_hex(
        (
            "The quick brown fox jumps over the lazy dog. Pack my box with"
            " five dozen liquor jugs."
        ),
        "d51712a8d1852b5acf942c19caddf168f80120d2f3a72c2d917227fd37f22788",
        ok,
    )

    # ── HuggingFace tree-API lfs.oid extraction ──────────────────────────────
    var tb = TREE.as_bytes()

    # Single-file LFS weight: must return lfs.oid, NOT the top-level git oid and
    # NOT the identical-length xetHash.
    var single = hf_lfs_sha256(tb, "model.safetensors")
    expect(
        single
        == "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "single lfs.oid = [" + single + "]",
        ok,
    )

    # Both shards resolve to their own lfs.oid.
    var s1 = hf_lfs_sha256(tb, "model-00001-of-00002.safetensors")
    expect(
        s1
        == "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
        "shard1 lfs.oid = [" + s1 + "]",
        ok,
    )
    var s2 = hf_lfs_sha256(tb, "model-00002-of-00002.safetensors")
    expect(
        s2
        == "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
        "shard2 lfs.oid = [" + s2 + "]",
        ok,
    )

    # Non-LFS blob (git-SHA-1 oid only): no sha256 to verify -> "".
    var cfg = hf_lfs_sha256(tb, "config.json")
    expect(
        cfg == "", "config.json should have no lfs.oid, got [" + cfg + "]", ok
    )
    var tokc = hf_lfs_sha256(tb, "tokenizer_config.json")
    expect(tokc == "", "tokenizer_config.json -> [" + tokc + "]", ok)

    # Absent file -> "".
    var miss = hf_lfs_sha256(tb, "does-not-exist.bin")
    expect(miss == "", "missing file -> [" + miss + "]", ok)

    # A filename that is a substring of another must not false-match.
    var partial = hf_lfs_sha256(tb, "model")
    expect(partial == "", "'model' partial -> [" + partial + "]", ok)

    if ok:
        print("hashing gate: PASS")
    else:
        print("hashing gate: FAIL")
        raise Error("hashing gate failed")
