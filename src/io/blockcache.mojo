"""Disk-backed block KV cache (server-side prefix cache that survives restarts).

The in-memory Session holds one prefix's K/V in GPU memory for the life of the
process. This persists K/V to disk in fixed `B`-token blocks so a restart — or a
*different* conversation that shares a prefix — reuses already-computed K/V
instead of re-prefilling (minutes on 3B) from scratch.

Design (the vLLM/SGLang radix-cache shape):
  - Split the prompt's token ids into `B`-token blocks; give each a *chained*
    hash `hᵢ = hash(hᵢ₋₁, block_i_tokens)`, so `hᵢ` identifies the whole prefix
    up to block i. Leading-prefix matches (the warm system+tools prefix shared by
    every request; a resumed conversation) therefore hit.
  - Each block is one file `<hex hash>.blk` = [B token ids (int32, for a
    collision-proof check on load)] + per layer [K slice, V slice] (f32). A block
    is valid only at its own absolute positions [i·B,(i+1)·B) — fine, because
    reuse is always a leading prefix and RoPE ties K to absolute position.
  - An `index` file lists block hashes in LRU order (also the inventory). Accessed
    blocks move to the end each request, so the warm prefix never ages out; the
    front is evicted when the store exceeds its block budget. `meta` stamps the
    model dims; a mismatch wipes the store (never load stale K/V into a model).

All host-side: K/V slices are moved with map_to_host (cheap on Apple's unified
memory — a pointer, no copy) + raw file bytes.
"""

from std.os import makedirs, remove
from std.os.path import exists
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy

comptime DevBuf = DeviceBuffer[DType.float32]
comptime FNV_OFFSET = UInt64(14695981039346656037)
comptime FNV_PRIME = UInt64(1099511628211)


def _mix(h0: UInt64, v: UInt64) -> UInt64:
    """FNV-1a over the 8 bytes of `v`, chained from `h0`."""
    var h = h0
    for b in range(8):
        h = (h ^ ((v >> UInt64(8 * b)) & UInt64(0xFF))) * FNV_PRIME
    return h

def _hex16(h: UInt64) -> String:
    var s = String("")
    for i in range(16):
        var nib = Int((h >> UInt64(60 - 4 * i)) & UInt64(0xF))
        s += chr(48 + nib) if nib < 10 else chr(97 + nib - 10)
    return s

def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

def _write_text(path: String, s: String) raises:
    with open(path, "w") as f:
        f.write(s)


struct BlockCache(Movable):
    var dir: String        # store directory (per model)
    var B: Int             # tokens per block
    var nkv: Int           # K/V row width = HKV * HEAD_DIM
    var nlayers: Int
    var max_blocks: Int    # budget / per-block bytes
    var enabled: Bool
    var order: List[String]  # block hex hashes, LRU order (oldest first) = inventory

    def __init__(out self, dir: String, B: Int, nkv: Int, nlayers: Int,
                 budget_bytes: Int, model_id: String):
        self.dir = dir
        self.B = B
        self.nkv = nkv
        self.nlayers = nlayers
        var blk_bytes = B * 4 + nlayers * 2 * B * nkv * 4
        self.max_blocks = budget_bytes // blk_bytes
        self.enabled = True
        self.order = List[String]()
        try:
            makedirs(dir, exist_ok=True)
            self._init_meta(model_id)
        except:
            self.enabled = False  # any fs trouble → cache disabled, server still works

    def _init_meta(mut self, model_id: String) raises:
        # Stamp dims; if an existing store doesn't match, wipe it (stale model).
        var stamp = model_id + "\n" + String(self.B) + "\n" + String(self.nkv) + "\n" + String(self.nlayers) + "\n"
        var meta = self.dir + "/meta"
        if exists(meta) and _read_text(meta) == stamp and exists(self.dir + "/index"):
            for line in _read_text(self.dir + "/index").split("\n"):
                var h = String(String(line).strip())
                if h.byte_length() > 0:
                    self.order.append(h)
        else:
            self._wipe()
            _write_text(meta, stamp)

    def _wipe(mut self):
        for i in range(len(self.order)):
            try:
                remove(self._path(self.order[i]))
            except:
                pass
        self.order = List[String]()
        try:
            if exists(self.dir + "/index"):
                remove(self.dir + "/index")
        except:
            pass

    def _path(self, hex: String) -> String:
        return self.dir + "/" + hex + ".blk"

    def chained_hashes(self, ids: List[Int]) -> List[UInt64]:
        """One chained hash per full B-token block of `ids` (partial tail ignored)."""
        var out = List[UInt64]()
        var nblocks = len(ids) // self.B
        var h = FNV_OFFSET
        for bi in range(nblocks):
            for j in range(self.B):
                h = _mix(h, UInt64(ids[bi * self.B + j]))
            out.append(h)
        return out^

    def _ids_match(self, path: String, ids: List[Int], bi: Int) raises -> Bool:
        """Collision check: the block file's stored token ids == ids[bi*B:(bi+1)*B]."""
        with open(path, "r") as f:
            var raw = f.read_bytes(self.B * 4)
            if len(raw) < self.B * 4:
                return False
            var p = raw.unsafe_ptr().bitcast[Int32]()
            for j in range(self.B):
                if Int(p[j]) != ids[bi * self.B + j]:
                    return False
        return True

    def longest_run(self, hashes: List[UInt64], ids: List[Int]) raises -> Int:
        """Number of leading blocks present on disk (with matching token ids)."""
        if not self.enabled:
            return 0
        var run = 0
        while run < len(hashes):
            var path = self._path(_hex16(hashes[run]))
            if not exists(path):
                break
            if not self._ids_match(path, ids, run):
                break
            run += 1
        return run

    def _write_ids(self, mut f: FileHandle, ids: List[Int], start: Int) raises:
        var buf = List[UInt8]()
        for j in range(self.B):
            var v = UInt32(ids[start + j])
            buf.append(UInt8(v & 0xFF))
            buf.append(UInt8((v >> 8) & 0xFF))
            buf.append(UInt8((v >> 16) & 0xFF))
            buf.append(UInt8((v >> 24) & 0xFF))
        f.write_bytes(Span(buf))

    def store_blocks(self, mut kcs: List[DevBuf], mut vcs: List[DevBuf],
                     hashes: List[UInt64], ids: List[Int], a: Int, b: Int) raises:
        """Write blocks [a, b) from the session's K/V buffers to disk."""
        if not self.enabled:
            return
        var slice_f = self.B * self.nkv               # floats per (block, layer) slice
        for bi in range(a, b):
            with open(self._path(_hex16(hashes[bi])), "w") as f:
                self._write_ids(f, ids, bi * self.B)
                for l in range(self.nlayers):
                    with kcs[l].map_to_host() as h:
                        var p = h.unsafe_ptr().bitcast[UInt8]() + bi * slice_f * 4
                        f.write_bytes(Span[UInt8, MutExternalOrigin](ptr=p, length=slice_f * 4))
                    with vcs[l].map_to_host() as h:
                        var p = h.unsafe_ptr().bitcast[UInt8]() + bi * slice_f * 4
                        f.write_bytes(Span[UInt8, MutExternalOrigin](ptr=p, length=slice_f * 4))

    def restore_blocks(self, mut kcs: List[DevBuf], mut vcs: List[DevBuf],
                       hashes: List[UInt64], a: Int, b: Int) raises:
        """Load blocks [a, b) from disk into the session's K/V buffers."""
        if not self.enabled:
            return
        var slice_f = self.B * self.nkv
        for bi in range(a, b):
            with open(self._path(_hex16(hashes[bi])), "r") as f:
                _ = f.read_bytes(self.B * 4)   # skip the token-id header
                for l in range(self.nlayers):
                    with kcs[l].map_to_host() as h:
                        var raw = f.read_bytes(slice_f * 4)
                        memcpy(dest=h.unsafe_ptr().bitcast[UInt8]() + bi * slice_f * 4,
                               src=raw.unsafe_ptr(), count=slice_f * 4)
                    with vcs[l].map_to_host() as h:
                        var raw = f.read_bytes(slice_f * 4)
                        memcpy(dest=h.unsafe_ptr().bitcast[UInt8]() + bi * slice_f * 4,
                               src=raw.unsafe_ptr(), count=slice_f * 4)

    def touch_and_evict(mut self, hashes: List[UInt64], nblocks: Int) raises:
        """Move this request's blocks to the LRU tail, then evict from the front
        until within the block budget. Persist the index."""
        if not self.enabled:
            return
        var accessed = List[String]()
        for i in range(nblocks):
            accessed.append(_hex16(hashes[i]))
        # keep prior order minus accessed, then accessed at the end (most recent)
        var kept = List[String]()
        for i in range(len(self.order)):
            var h = self.order[i]
            var dup = False
            for j in range(len(accessed)):
                if accessed[j] == h:
                    dup = True
                    break
            if not dup:
                kept.append(h)
        for j in range(len(accessed)):
            kept.append(accessed[j])
        # evict oldest (front) over budget
        var start = 0
        if self.max_blocks > 0 and len(kept) > self.max_blocks:
            start = len(kept) - self.max_blocks
            for i in range(start):
                try:
                    remove(self._path(kept[i]))
                except:
                    pass
        self.order = List[String]()
        var idx = String("")
        for i in range(start, len(kept)):
            self.order.append(kept[i])
            idx += kept[i] + "\n"
        _write_text(self.dir + "/index", idx)
