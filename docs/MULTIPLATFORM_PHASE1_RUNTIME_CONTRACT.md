# Multiplatform Phase 1 Runtime Contract

## Scope
Phase 1 defines backend runtime contracts for:
- Device/runtime bootstrap
- Queue lifecycle and synchronization
- Event lifecycle and queue ordering
- Buffer lifecycle and copy operations
- Capability profile reporting

Phase 1 does not require full public API decoupling from CUDA-native containers.

## Runtime Entities

### `BackendRuntime`
Authoritative operation surface for backend runtime operations.

Required responsibilities:
1. Create queues, events, and buffers.
2. Synchronize queue/event/device.
3. Record/wait event on queues.
4. Perform async copy operations.
5. Report capability profile.

### `BackendQueue`
Opaque execution queue handle with backend/device identity and native handle exposure.

### `BackendEvent`
Opaque synchronization event handle with backend/device identity and native handle exposure.

### `BackendBuffer`
Opaque buffer handle with backend/device identity, descriptor metadata, and native pointer handle.

### `CapabilityProfile`
Backend feature declaration:
- `supports_queues`
- `supports_events`
- `supports_device_buffers`
- `supports_host_visible_buffers`
- `supports_unified_memory`
- `supports_interop_buffers`
- `supports_graph_capture`
- `compute_capability`
- `total_global_memory_bytes`

## Error/Status Contract
All runtime operations return `BackendError`:
- `code` in `BackendErrorCode`
- `backend` identity
- `operation` name
- contextual `message`

`BackendErrorCode` categories:
- Success
- InvalidArgument
- Unsupported
- OutOfMemory
- Timeout
- DeviceLost
- SynchronizationError
- RuntimeFailure
- UnknownFailure

## Lifecycle Rules
1. A queue/event/buffer belongs to exactly one runtime backend + device.
2. Mixed-runtime operations must return `InvalidArgument`.
3. Runtime object destruction must release owned backend resources.
4. Queue and event ordering semantics are explicit (`record_event`, `wait_event`).
5. Synchronization semantics are explicit (`synchronize_queue`, `synchronize_event`, `synchronize_device`).

## CUDA Phase 1 Coverage
CUDA implementation in Phase 1 supports:
1. Queue creation (non-blocking/default flags).
2. Event creation (timing-disable/default flags).
3. Device and unified buffers.
4. Host-to-device, device-to-host, and device-to-device async copies.
5. Queue/event/device synchronization and event ordering.

Not implemented in Phase 1 CUDA runtime:
1. HIP/METAL runtime implementations.
2. Interop-capable buffers.
3. Host-pinned buffer class via runtime contract.
