# Multiplatform Phase 2: Memory and Error Contract

## Scope
Phase 2 standardizes:
- deterministic runtime status behavior (`BackendError`)
- explicit buffer memory/access/interop intent descriptors
- explicit copy-region descriptors for all runtime copy operations

This phase does not require HIP/METAL bring-up.

## BackendError Rules
All `BackendRuntime` operations must:
1. Return a `BackendError` with non-empty `operation`.
2. Map failures to stable `BackendErrorCode` categories.
3. Fail fast on invalid argument and capability mismatch paths.

## Memory Descriptor Contract
`BufferDesc` now encodes:
- `memory_class`: `Device`, `Unified`, `HostPinned`
- `host_access`: `None`, `Read`, `Write`, `ReadWrite`
- `interop_mode`: `None`, `External`
- `size_bytes`, `alignment`, `debug_name`

Validation behavior for CUDA runtime:
1. `size_bytes` must be `> 0`.
2. `alignment` must be power-of-two.
3. `interop_mode != None` returns `Unsupported`.
4. `HostPinned` memory class returns `Unsupported` (planned, not implemented).
5. `Device + host_access != None` returns `Unsupported`.
6. `Unified + host_access == None` is normalized to `ReadWrite`.

## Copy Region Contract
Runtime copy APIs use explicit region descriptors:
- `CopyRegion`: `size_bytes`, `dst_offset`, `src_offset`
- `BufferTransferRegion`: `size_bytes`, `buffer_offset`

Validation rules:
1. Null queue/buffer pointers return `InvalidArgument`.
2. `size_bytes == 0` is success no-op.
3. Offset arithmetic must be overflow-safe.
4. Region bounds must stay within buffer sizes.

## Runtime Bootstrap
`runtime_factory` is the single backend runtime bootstrap surface:
- `create_backend_runtime(const BackendConfig&)`

Legacy `BackendContext` / `create_backend_context` path is retired.
