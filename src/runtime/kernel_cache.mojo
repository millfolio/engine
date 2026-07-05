"""Process-lifetime store of compiled GPU kernels, so each Metal compute
pipeline is built ONCE and reused for every later dispatch.

The stdlib convenience path `ctx.enqueue_function[k](...)` compiles the kernel
fresh on *every* call (it calls `compile_function` then discards the resulting
`DeviceFunction`), so with a cold Metal shader cache each user query re-runs the
Metal shader compiler on the whole generative hot path — pinning the GPU on
`newComputePipelineState…`/`MTLCompiler` while the compute workers idle.

`cached_enqueue` instead holds the compiled `DeviceFunction` in a global slot
keyed by the kernel's mangled linkage name (unique per monomorphization,
shape-agnostic since the op layouts carry their dims as runtime values). The
first dispatch of a kernel compiles it; every later dispatch — including the
warm-up at load — reuses the cached pipeline. Once `_warmup()` has driven every
hot-path kernel once, real queries dispatch compile-free.

The model forward runs single-threaded (one KV session, serialized requests) and
warm-up populates every slot before serving, so the check-then-compile below is
not a concurrency hazard in practice.
"""

from std.ffi import _Global
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim
from std.builtin.device_passable import DevicePassable
from std.reflection import reflect_fn


def _empty_slot[T: Movable]() -> Optional[T]:
    """Thin initializer for a global cache slot: an empty (uncompiled) Optional.
    The kernel is compiled lazily on the slot's first `cached_enqueue`."""
    return None


@always_inline
def cached_enqueue[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *actual_arg_types: DevicePassable,
](
    ctx: DeviceContext,
    *args: *actual_arg_types,
    grid_dim: Dim,
    block_dim: Dim,
) raises:
    """Enqueue `func` reusing its process-lifetime compiled `DeviceFunction`.

    Drop-in for `ctx.enqueue_function[func](args..., grid_dim=, block_dim=)`,
    but compiles the Metal pipeline only on the first dispatch of this exact
    monomorphization and reuses it thereafter. The cache key is the kernel's
    mangled linkage name — identical monomorphizations (any runtime shape) share
    one compiled pipeline; distinct comptime specializations get their own.

    Args:
        ctx: The GPU device context.
        args: The kernel arguments (same as `enqueue_function`).
        grid_dim: The grid dimensions.
        block_dim: The block dimensions.

    Raises:
        If compilation or the launch fails.
    """
    comptime FnT = type_of(ctx.compile_function[func]())
    comptime key = reflect_fn[func].linkage_name()
    var slot = _Global[key, _empty_slot[FnT]].get_or_create_ptr()
    if not slot[]:
        slot[] = ctx.compile_function[func]()
    ctx.enqueue_function(
        slot[].value(), *args, grid_dim=grid_dim, block_dim=block_dim
    )
