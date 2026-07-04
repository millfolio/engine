"""Native-Mojo Qwen weights downloader — no huggingface_hub / no Python wheel.

Fetches a Qwen2.5 checkpoint straight from HuggingFace over HTTPS (via flare's
TLS client) and writes it into the same on-disk layout `huggingface_hub` uses, so
the server's `hf_cache_path()` finds it unchanged:

    <hub>/models--<slug>/snapshots/<commit>/<files>
    <hub>/refs/main is NOT used; instead:
    <hub>/models--<slug>/refs/main          -> contains <commit>

where <hub> is $HF_HOME/hub or ~/.cache/huggingface/hub, and <slug> turns
'Qwen/Qwen2.5-3B-Instruct' into 'Qwen--Qwen2.5-3B-Instruct'.

The commit hash is read from the `X-Repo-Commit` response header HF sends on every
`/resolve/<rev>/...` request, so a moving ref like `main` is pinned to the exact
revision actually downloaded.

Integrity: every git-LFS file (the safetensors weights) is verified by sha256
before it is written to disk. HF publishes each LFS file's sha256 as `lfs.oid` in
the model tree API (`/api/models/<repo>/tree/<commit>`); we recompute the sha256
of the downloaded bytes and refuse to write on a mismatch. A `Content-Length`
match alone only rules out truncation — it can't detect a tampered or poisoned
weight, and safetensors carry no embedded signature. Non-LFS blobs (config.json,
tokenizer assets) have no published content sha256 (their tree `oid` is a git
SHA-1 of the blob, not a sha256 of the bytes), so those keep the size check only.

Files fetched:
  - config.json, generation_config.json          (always)
  - model.safetensors                            (single-file: 0.5B)
  - model.safetensors.index.json + every shard   (sharded: 3B)
  - tokenizer.json, tokenizer_config.json,
    vocab.json, merges.txt                       (best-effort; HF tokenizer assets)

Build:  pixi run build-download   (mojo build src/download.mojo -I ../flare -o build/download)
Run:    build/download [Qwen/Qwen2.5-0.5B-Instruct] [--revision main]

NOTE: flare's HTTP client buffers each response body fully in memory before we
write it, so peak RSS is one shard (~1 GB for 0.5B, ~3 GB per shard for 3B). On
Apple unified memory that is fine for these sizes; true streaming would need the
lower-level TlsStream and is left for later if larger models are added.
"""

from std.sys import argv
from std.os import getenv, makedirs
from std.os.path import exists, getsize
from flare.http import HttpClient, Response
from hashing import sha256_hex, hf_lfs_sha256


comptime DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
"""HuggingFace repo id downloaded when none is given on the command line."""
# Generous per-file read+connect timeout (30 min) — a multi-GB shard on a slow
# link must not trip flare's default 30 s.
comptime TIMEOUT_MS = 1_800_000
"""Per-file HTTP read+connect timeout in milliseconds (30 min for large shards)."""


def slug(model_id: String) -> String:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'Qwen--Qwen2.5-3B-Instruct' (HF cache dir).

    Args:
        model_id: The HuggingFace repo id to slugify.

    Returns:
        The repo id with each '/' replaced by '--'.
    """
    var b = model_id.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        if b[i] == 47:  # '/'
            out.append(45)
            out.append(45)
        else:
            out.append(b[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def hub_root() -> String:
    """$HF_HOME/hub, else ~/.cache/huggingface/hub (mirrors huggingface_hub).

    Returns:
        The absolute path to the HuggingFace hub cache root.
    """
    var home = String(getenv("HF_HOME"))
    if home.byte_length() > 0:
        return home + "/hub"
    return String(getenv("HOME")) + "/.cache/huggingface/hub"


def resolve_url(repo: String, rev: String, file: String) -> String:
    """HuggingFace `/resolve/<rev>/<file>` download URL for `repo`.

    Args:
        repo: The HuggingFace repo id.
        rev: The revision (branch, tag, or commit) to resolve.
        file: The file path within the repo.

    Returns:
        The full HuggingFace resolve URL for the file.
    """
    return "https://huggingface.co/" + repo + "/resolve/" + rev + "/" + file


def shard_names(index_text: String) -> List[String]:
    """Distinct '*.safetensors' filenames named anywhere in the index JSON's
    weight_map. Robust substring scan — every shard appears as a quoted value, and
    no JSON structural token ends in '.safetensors'.

    Args:
        index_text: The raw text of the safetensors index JSON.

    Returns:
        The distinct '*.safetensors' shard filenames found in the index.
    """
    var names = List[String]()
    var parts = index_text.split('"')
    for i in range(len(parts)):
        var seg = String(parts[i])
        if seg.endswith(".safetensors"):
            var seen = False
            for j in range(len(names)):
                if names[j] == seg:
                    seen = True
                    break
            if not seen:
                names.append(seg^)
    return names^


def write_bytes(path: String, data: List[UInt8]) raises:
    """Write `data` to `path`, chunking to stay under macOS write(2)'s ~2 GiB cap.

    Args:
        path: The destination file path.
        data: The bytes to write.

    Raises:
        Error: if the file cannot be opened or written.
    """
    # macOS write(2) rejects a single call larger than INT_MAX (~2 GiB) with
    # EINVAL, and the 3B shards exceed that — so write in bounded chunks.
    var n = len(data)
    var sp = Span(data)
    with open(path, "w") as f:
        var off = 0
        comptime CHUNK = 256 * 1024 * 1024
        while off < n:
            var end = off + CHUNK
            if end > n:
                end = n
            f.write_bytes(sp[off:end])
            off = end


def fetch(mut client: HttpClient, url: String) raises -> Response:
    """GET `url` over `client` and return the full `Response`.

    Args:
        client: The HTTP client to use.
        url: The URL to GET.

    Returns:
        The full HTTP response.

    Raises:
        Error: on network or HTTP client failure.
    """
    var resp = client.get(url)
    return resp^


def remote_size(mut client: HttpClient, url: String) -> Int:
    """Content-Length of the resolved file (HEAD, following redirects), or -1 if
    unknown — used to tell a complete download from a truncated/empty one.

    Args:
        client: The HTTP client to use.
        url: The URL to issue a HEAD request against.

    Returns:
        The Content-Length in bytes, or -1 if unknown.
    """
    try:
        var r = client.head(url)
        if r.status != 200:
            return -1
        var cl = r.headers.get("content-length")
        if cl.byte_length() == 0:
            return -1
        return atol(cl)
    except:
        return -1


def download_one(
    mut client: HttpClient,
    repo: String,
    rev: String,
    file: String,
    snap_dir: String,
    optional: Bool,
    expected_sha256: String = "",
) raises -> String:
    """Download <repo>/<rev>/<file> into snap_dir. Skips if already present.
    Returns the X-Repo-Commit header (so the caller can pin the snapshot), or ""
    if an optional file was absent (404).

    When `expected_sha256` is a 64-char hex sha256 (the file's git-LFS `lfs.oid`),
    the downloaded bytes are hashed and compared BEFORE being written — a mismatch
    raises and nothing is persisted, so a tampered/poisoned weight never lands on
    disk. Any other value (empty, or a non-sha256 length) skips hash verification
    and the size check remains the only guard (e.g. non-LFS blobs, whose content
    sha256 HF does not publish).

    Args:
        client: The HTTP client to use.
        repo: The HuggingFace repo id.
        rev: The revision to resolve.
        file: The file path within the repo.
        snap_dir: The snapshot directory to write the file into.
        optional: If True, a 404 is tolerated and skipped instead of raising.
        expected_sha256: The expected sha256 (lowercase hex) of the file's bytes,
            or "" to skip hash verification.

    Returns:
        The X-Repo-Commit header value, or "" if the file was skipped or absent.

    Raises:
        Error: if a required file returns a non-200, non-404 HTTP status, the
            write fails, or the downloaded bytes fail sha256 verification.
    """
    var dest = snap_dir + "/" + file
    var url = resolve_url(repo, rev, file)
    # Resume: skip only if the local file is byte-complete vs the remote. A
    # truncated or 0-byte file (e.g. an earlier interrupted/failed write) must be
    # re-fetched, not skipped. (A pre-existing complete file was already sha256-
    # verified when a prior run wrote it, so we don't re-hash multi-GB shards on
    # every resume.)
    if exists(dest):
        var want = remote_size(client, url)
        if want > 0 and Int(getsize(dest)) == want:
            print("  have   ", file)
            return ""
    var resp = fetch(client, url)
    if resp.status == 404 and optional:
        print("  skip   ", file, "(not in repo)")
        return ""
    if resp.status != 200:
        raise Error(
            "GET "
            + file
            + " -> HTTP "
            + String(resp.status)
            + " ("
            + resp.reason
            + ")"
        )
    var commit = resp.headers.get("x-repo-commit")
    var n = len(resp.body)
    # Integrity: verify the sha256 of the downloaded bytes against HF's published
    # git-LFS oid BEFORE writing, so a tampered/corrupt weight is never persisted.
    if expected_sha256.byte_length() == 64:
        var got = sha256_hex(Span(resp.body))
        if got != expected_sha256:
            raise Error(
                "SECURITY: sha256 mismatch for "
                + file
                + " — expected "
                + expected_sha256
                + " but the downloaded bytes hash to "
                + got
                + ". Refusing to write a tampered or corrupt weight file."
            )
        print("  verify ", file, "sha256 ✓")
    write_bytes(dest, resp.body)
    print("  wrote  ", file, "(", n, "bytes )")
    return commit


def main() raises:
    """Download a Qwen2.5 checkpoint into the HuggingFace cache layout, pinning the
    snapshot to the resolved commit and the `main` ref.

    Raises:
        Error: on an unknown flag, a failed config.json fetch, or any download
            or write failure.
    """
    # Parse argv: [model-id] [--revision REV]
    var model = String(DEFAULT_MODEL)
    var rev = String("main")
    var args = argv()
    var i = 1
    var positional = 0
    while i < len(args):
        var a = String(args[i])
        if a == "--revision" or a == "-r":
            i += 1
            if i < len(args):
                rev = String(args[i])
        elif a.startswith("--"):
            raise Error("unknown flag: " + a)
        else:
            if positional == 0:
                model = a
            positional += 1
        i += 1

    var hub = hub_root()
    var repo_dir = hub + "/models--" + slug(model)
    print("model:   ", model, "@", rev)
    print("hub:     ", hub)

    var client = HttpClient(
        base_url="",
        max_redirects=10,
        timeout_ms=TIMEOUT_MS,
        user_agent="millfolio-downloader/0.1",
    )

    # 1) config.json first — mandatory, and its X-Repo-Commit pins the snapshot
    #    (a moving ref like `main` resolves to the exact revision downloaded).
    print("resolving revision...")
    var cfg = fetch(client, resolve_url(model, rev, "config.json"))
    if cfg.status != 200:
        raise Error(
            "config.json -> HTTP "
            + String(cfg.status)
            + " ("
            + cfg.reason
            + ")"
        )
    var commit = cfg.headers.get("x-repo-commit")
    if commit.byte_length() == 0:
        # Fall back to the literal ref if HF omitted the header.
        commit = rev
    print("commit:  ", commit)

    var snap = repo_dir + "/snapshots/" + commit
    if not exists(snap):
        makedirs(snap)

    # Fetch the model tree at the pinned commit so we know each LFS file's
    # expected sha256 (`lfs.oid`) and can verify weights against tampering. If the
    # tree is unavailable we degrade to the size-only check and say so loudly —
    # rather than failing the whole install — but weights then go unverified.
    var tree_body = List[UInt8]()
    var tree_ok = False
    try:
        var tr = fetch(
            client,
            "https://huggingface.co/api/models/" + model + "/tree/" + commit,
        )
        if tr.status == 200:
            tree_body = tr.body.copy()
            tree_ok = True
        else:
            print(
                "  WARN: tree API HTTP",
                tr.status,
                "— weight sha256 verification UNAVAILABLE (size check only)",
            )
    except:
        print(
            "  WARN: tree API fetch failed — weight sha256 verification"
            " UNAVAILABLE (size check only)"
        )
    if tree_ok:
        print("  manifest: got sha256 hashes for weight verification")

    # Write config.json into the snapshot now (we already have its bytes).
    var cfg_dest = snap + "/config.json"
    if not exists(cfg_dest):
        write_bytes(cfg_dest, cfg.body)
        print("  wrote   config.json (", len(cfg.body), "bytes )")
    else:
        print("  have    config.json")

    # 2) Weights: sharded (index.json present) or a single model.safetensors.
    print("downloading weights...")
    var idx = fetch(
        client, resolve_url(model, rev, "model.safetensors.index.json")
    )
    if idx.status == 200:
        write_bytes(snap + "/model.safetensors.index.json", idx.body)
        print("  wrote   model.safetensors.index.json")
        var idx_text = String(StringSlice(unsafe_from_utf8=Span(idx.body)))
        var shards = shard_names(idx_text)
        print("  ", len(shards), "shard(s)")
        for s in range(len(shards)):
            var sha = hf_lfs_sha256(Span(tree_body), shards[s])
            _ = download_one(
                client, model, rev, shards[s], snap, False, expected_sha256=sha
            )
    else:
        var sha = hf_lfs_sha256(Span(tree_body), "model.safetensors")
        _ = download_one(
            client,
            model,
            rev,
            "model.safetensors",
            snap,
            False,
            expected_sha256=sha,
        )

    # 3) Auxiliary + tokenizer assets (best-effort; absent files are skipped).
    print("downloading aux + tokenizer assets...")
    var aux = [
        String("generation_config.json"),
        String("tokenizer.json"),
        String("tokenizer_config.json"),
        String("vocab.json"),
        String("merges.txt"),
        String("special_tokens_map.json"),
    ]
    for a in range(len(aux)):
        # Aux/tokenizer assets are normally non-LFS (no published sha256 → "" →
        # size check only), but pass the lookup anyway so any that IS an LFS file
        # gets verified too.
        var sha = hf_lfs_sha256(Span(tree_body), aux[a])
        _ = download_one(
            client, model, rev, aux[a], snap, True, expected_sha256=sha
        )

    # 4) Pin the ref so hf_cache_path() resolves <hub>/models--<slug>/refs/main.
    var refs = repo_dir + "/refs"
    if not exists(refs):
        makedirs(refs)
    var cb = List[UInt8]()
    for x in commit.as_bytes():
        cb.append(x)
    write_bytes(refs + "/main", cb)

    print("done. snapshot at:")
    print("  ", snap)
    print("the server resolves it via: serve", model)
