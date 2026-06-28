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


comptime DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Generous per-file read+connect timeout (30 min) — a multi-GB shard on a slow
# link must not trip flare's default 30 s.
comptime TIMEOUT_MS = 1_800_000


def slug(model_id: String) -> String:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'Qwen--Qwen2.5-3B-Instruct' (HF cache dir).
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
    """$HF_HOME/hub, else ~/.cache/huggingface/hub (mirrors huggingface_hub)."""
    var home = String(getenv("HF_HOME"))
    if home.byte_length() > 0:
        return home + "/hub"
    return String(getenv("HOME")) + "/.cache/huggingface/hub"


def resolve_url(repo: String, rev: String, file: String) -> String:
    return "https://huggingface.co/" + repo + "/resolve/" + rev + "/" + file


def shard_names(index_text: String) -> List[String]:
    """Distinct '*.safetensors' filenames named anywhere in the index JSON's
    weight_map. Robust substring scan — every shard appears as a quoted value, and
    no JSON structural token ends in '.safetensors'."""
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
    var resp = client.get(url)
    return resp^


def remote_size(mut client: HttpClient, url: String) -> Int:
    """Content-Length of the resolved file (HEAD, following redirects), or -1 if
    unknown — used to tell a complete download from a truncated/empty one."""
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
) raises -> String:
    """Download <repo>/<rev>/<file> into snap_dir. Skips if already present.
    Returns the X-Repo-Commit header (so the caller can pin the snapshot), or ""
    if an optional file was absent (404)."""
    var dest = snap_dir + "/" + file
    var url = resolve_url(repo, rev, file)
    # Resume: skip only if the local file is byte-complete vs the remote. A
    # truncated or 0-byte file (e.g. an earlier interrupted/failed write) must be
    # re-fetched, not skipped.
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
    write_bytes(dest, resp.body)
    print("  wrote  ", file, "(", n, "bytes )")
    return commit


def main() raises:
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
            _ = download_one(client, model, rev, shards[s], snap, False)
    else:
        _ = download_one(client, model, rev, "model.safetensors", snap, False)

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
        _ = download_one(client, model, rev, aux[a], snap, True)

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
