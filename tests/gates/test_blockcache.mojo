"""Gate: disk block KV-cache roundtrip (blockcache.mojo). `pixi run test-blockcache`.

Stores synthetic per-layer K/V buffers to disk in blocks, corrupts the GPU
buffers, restores from disk, and checks the result is bit-identical — plus
chained-hash determinism and the token-id collision guard. Needs the GPU (uses
map_to_host), so it lives with the engine, not the pure-Python suite.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from model import BlockCache

comptime DevBuf = DeviceBuffer[DType.float32]


def fill(ctx: DeviceContext, mut buf: DevBuf, base: Float32, n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = base + Float32(i) * 0.5
    ctx.synchronize()


def clobber(ctx: DeviceContext, mut buf: DevBuf, n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = -7.0
    ctx.synchronize()


def checksum(ctx: DeviceContext, mut buf: DevBuf, n: Int) raises -> Float64:
    var s = Float64(0.0)
    with buf.map_to_host() as h:
        for i in range(n):
            s += Float64(h[i])
    return s


def main() raises:
    var ctx = DeviceContext()
    var B = 4
    var nkv = 8
    var nlayers = 5
    var npos = 12  # 3 full blocks
    var clen = npos * nkv
    var ok = True

    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    var pre_k = List[Float64]()
    var pre_v = List[Float64]()
    for l in range(nlayers):
        var k = ctx.enqueue_create_buffer[DType.float32](clen)
        var v = ctx.enqueue_create_buffer[DType.float32](clen)
        fill(ctx, k, Float32(100 * l + 1), clen)
        fill(ctx, v, Float32(1000 * l + 3), clen)
        pre_k.append(checksum(ctx, k, clen))
        pre_v.append(checksum(ctx, v, clen))
        kcs.append(k^)
        vcs.append(v^)

    var ids = List[Int]()
    for t in range(npos):
        ids.append(2000 + t * 13)

    var cache = BlockCache(
        ".scratch/kvstore_gate", B, nkv, nlayers, 1 << 30, "test/model"
    )
    if not cache.enabled:
        raise Error("cache failed to init")
    var hashes = cache.chained_hashes(ids)

    if len(hashes) != 3:
        print("  FAIL: expected 3 blocks, got", len(hashes))
        ok = False

    var h2 = cache.chained_hashes(ids)
    for i in range(len(hashes)):
        if hashes[i] != h2[i]:
            print("  FAIL: hash not deterministic at", i)
            ok = False

    cache.store_blocks(kcs, vcs, hashes, ids, 0, 3)
    cache.touch_and_evict(hashes, 3)
    if cache.longest_run(hashes, ids) != 3:
        print("  FAIL: longest_run != 3 after store")
        ok = False

    for l in range(nlayers):
        clobber(ctx, kcs[l], clen)
        clobber(ctx, vcs[l], clen)
    cache.restore_blocks(kcs, vcs, hashes, 0, 3)
    for l in range(nlayers):
        if (
            checksum(ctx, kcs[l], clen) != pre_k[l]
            or checksum(ctx, vcs[l], clen) != pre_v[l]
        ):
            print("  FAIL: restore not bit-identical at layer", l)
            ok = False

    # collision guard: a different token at block 0 must not match
    var other = ids.copy()
    other[1] = 424242
    if cache.longest_run(cache.chained_hashes(other), other) != 0:
        print("  FAIL: collision guard — different ids matched")
        ok = False

    if ok:
        print("blockcache gate: PASS")
    else:
        raise Error("blockcache gate failed")
