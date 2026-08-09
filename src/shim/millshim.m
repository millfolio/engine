// millshim: batch many Mojo-compiled Metal kernel dispatches into ONE command
// buffer per call — the stopgap for Mojo's DeviceGraph being unimplemented on
// Metal (delete this when it lands upstream; see ModCon Aug 18 recheck).
//
// ABI (empirically established, .scratch/shim/probe2.m + tt_test.m):
//   * every top-level kernel arg = its own binding index, set via setBytes;
//   * pointer args are raw 8-byte gpuAddress values (NOT setBuffer);
//   * TileTensor args = leading 8-byte gpuAddress (+ ignored tail);
//   * every referenced MTLBuffer needs useResource for residency — we track
//     all aliased/created buffers and mark them once per encoder.
//
// Buffers: millshim_alias() wraps an EXISTING page-aligned host allocation
// (AsyncRT device buffers on unified memory are shared-storage) via
// newBufferWithBytesNoCopy on an aligned superset of the range, returning the
// gpuAddress corresponding to the caller's pointer. The shim never frees them.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static NSMutableArray<id<MTLComputePipelineState>> *g_psos;
static NSMutableArray<id<MTLBuffer>> *g_bufs;
static id<MTLCommandBuffer> g_cb;
static id<MTLComputeCommandEncoder> g_enc;
static mach_timebase_info_data_t g_tb;

int64_t millshim_init(void) {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return -1;
    g_q = [g_dev newCommandQueue];
    g_psos = [NSMutableArray new];
    g_bufs = [NSMutableArray new];
    mach_timebase_info(&g_tb);
    return 0;
}

int32_t millshim_load_kernel(const void *bytes, int64_t len, const char *name) {
    NSError *err = nil;
    dispatch_data_t dd = dispatch_data_create(bytes, (size_t)len, NULL,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [g_dev newLibraryWithData:dd error:&err];
    if (!lib) { NSLog(@"millshim: lib load failed: %@", err); return -1; }
    id<MTLFunction> f = [lib newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!f) { NSLog(@"millshim: fn %s missing (have %@)", name, lib.functionNames); return -1; }
    // Reflect the kernel's ACTUAL argument binding model (buffer vs bytes, index)
    // — the ground truth for how the current Mojo toolchain lowers kernel args.
    MTLComputePipelineReflection *refl = nil;
    id<MTLComputePipelineState> pso =
        [g_dev newComputePipelineStateWithFunction:f
                                           options:MTLPipelineOptionBindingInfo
                                        reflection:&refl
                                             error:&err];
    if (!pso) { NSLog(@"millshim: pso failed: %@", err); return -1; }
    if (getenv("MILLSHIM_REFLECT")) {
        NSLog(@"millshim: kernel '%s' — %lu args:", name,
              (unsigned long)refl.bindings.count);
        for (id<MTLBinding> b in refl.bindings) {
            NSLog(@"  arg[%lu] '%@' type=%ld (0=buffer,2=threadgroup,...) %@",
                  (unsigned long)b.index, b.name, (long)b.type,
                  b.argument ? @"" : @"");
        }
    }
    [g_psos addObject:pso];
    return (int32_t)g_psos.count - 1;
}

uint64_t millshim_alias(void *host_ptr, int64_t len) {
    uintptr_t p = (uintptr_t)host_ptr;
    uintptr_t base = p & ~((uintptr_t)vm_page_size - 1);
    size_t span = ((p + (size_t)len + vm_page_size - 1) & ~((uintptr_t)vm_page_size - 1)) - base;
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                               length:span
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!b) { NSLog(@"millshim: alias failed for %p len %lld", host_ptr, len); return 0; }
    [g_bufs addObject:b];
    return b.gpuAddress + (p - base);
}

// Allocate a NATIVE g_dev shared buffer (zeroed), tracked in g_bufs so push's
// range-match + begin's useResource cover it. Returns its gpuAddress. Diagnostic:
// bind this as a kernel output to test whether the kernel writes at all, isolating
// kernel execution from AsyncRT-buffer aliasing coherence.
uint64_t millshim_native_alloc(int64_t nbytes) {
    id<MTLBuffer> b = [g_dev newBufferWithLength:(NSUInteger)nbytes
                                         options:MTLResourceStorageModeShared];
    if (!b) return 0;
    memset(b.contents, 0, (size_t)nbytes);
    [g_bufs addObject:b];
    return b.gpuAddress;
}

// The host contents pointer for a native/aliased buffer (by gpuAddress), so the
// caller can fill inputs directly (fast, no per-element FFI).
uint64_t millshim_native_contents(uint64_t gpuAddr) {
    for (id<MTLBuffer> b in g_bufs) {
        uint64_t base = b.gpuAddress;
        if (gpuAddr >= base && gpuAddr < base + (uint64_t)b.length)
            return (uint64_t)((uintptr_t)b.contents + (gpuAddr - base));
    }
    return 0;
}

// Read a float from a native/aliased buffer by gpuAddress + element index.
float millshim_native_read_f32(uint64_t gpuAddr, int64_t idx) {
    for (id<MTLBuffer> b in g_bufs) {
        uint64_t base = b.gpuAddress;
        if (gpuAddr >= base && gpuAddr < base + (uint64_t)b.length)
            return ((const float *)b.contents)[(gpuAddr - base) / 4 + (uint64_t)idx];
    }
    return -999.0f;
}

void millshim_begin(void) {
    g_cb = [g_q commandBuffer];
    g_enc = [g_cb computeCommandEncoder];
    for (id<MTLBuffer> b in g_bufs)
        [g_enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
}

void millshim_push(int32_t pso, int64_t gx, int64_t gy, int64_t gz,
                   int64_t bx, int64_t by, int64_t bz,
                   const uint8_t *blob, const int32_t *lens, int32_t nargs) {
    [g_enc setComputePipelineState:g_psos[(NSUInteger)pso]];
    const uint8_t *cur = blob;
    for (int32_t i = 0; i < nargs; i++) {
        // Mojo lowers EVERY kernel arg to a buffer binding (verified via pipeline
        // reflection). A TENSOR arg's leading 8 bytes are the buffer's gpuAddress
        // — bind the actual MTLBuffer at that offset so the kernel's `device T*`
        // points at the real data (NOT the {ptr,dim} struct bytes). A SCALAR arg
        // (K/N/…) is a small value that matches no buffer → setBytes inline.
        BOOL bound = NO;
        if (lens[i] >= 8) {
            uint64_t v;
            memcpy(&v, cur, sizeof(v));
            for (id<MTLBuffer> b in g_bufs) {
                uint64_t base = b.gpuAddress;
                if (v >= base && v < base + (uint64_t)b.length) {
                    [g_enc setBuffer:b offset:(NSUInteger)(v - base) atIndex:(NSUInteger)i];
                    bound = YES;
                    if (getenv("MILLSHIM_REFLECT"))
                        NSLog(@"  push arg[%d] len=%d → setBuffer(off=%llu)", i, lens[i], v - base);
                    break;
                }
            }
        }
        if (!bound) {
            [g_enc setBytes:cur length:(NSUInteger)lens[i] atIndex:(NSUInteger)i];
            if (getenv("MILLSHIM_REFLECT")) {
                uint64_t vv = 0; if (lens[i] >= 8) memcpy(&vv, cur, 8);
                NSLog(@"  push arg[%d] len=%d → setBytes (v=%llu)", i, lens[i], vv);
            }
        }
        cur += lens[i];
    }
    [g_enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)bx, (NSUInteger)by, (NSUInteger)bz)];
}

double millshim_commit(void) {
    [g_enc endEncoding];
    uint64_t t0 = mach_absolute_time();
    [g_cb commit];
    [g_cb waitUntilCompleted];
    uint64_t t1 = mach_absolute_time();
    if (g_cb.status == MTLCommandBufferStatusError)
        NSLog(@"millshim: cmdbuf ERROR status=%ld: %@", (long)g_cb.status, g_cb.error);
    g_enc = nil; g_cb = nil;
    return (double)(t1 - t0) * g_tb.numer / g_tb.denom / 1.0e6;
}
