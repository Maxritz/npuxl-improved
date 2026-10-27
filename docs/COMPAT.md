# Legacy compatibility

## What changed

The original repository had:

- `virtual_npu.c/h` – flat-memory batched matmul experiments;
- `pcem/src/video/vgpu_npu.c` – a PCem Voodoo-backed compute unit;
- PCem build files that referenced Voodoo renderer headers;
- Sherlock tracing with hard-coded `elapsed = 0.0005`;
- a `test_npu_matmul.c` in the PCem tree that did not actually compute matmul.

The new runtime:

- has **zero PCem headers or sources** in the build;
- implements both legacy APIs in `src/compat/compat.cpp`;
- maps legacy matmul opcodes to the optimized command stream;
- implements real Sherlock timers, bytes and ops counters;
- fixes the 32-bit address truncation in `virtual_npu_compute()`;
- preserves culling semantics for CPU/Voodoo (per element) and Wildcat/Merged
  (per tile);
- provides `vgpu_npu_matmul_npu()` as a real matmul extension.

## Building old tests

```bash
make -j$(nproc)
cc -O2 -Iinclude -Ilegacy legacy/test_voodoo_2x2.c \
  -Llib -lvnpu -Wl,-rpath,'$ORIGIN/../lib' -o /tmp/test_voodoo_2x2
/tmp/test_voodoo_2x2
```

Include paths:

- `include/virtual_npu.h` -> `include/vnpu_compat.h`
- `include/vgpu_npu.h` -> `include/vnpu_compat.h`
- `include/includes/private/video/vgpu_npu.h` -> `include/vnpu_compat.h`

The last path exists so the original PCem-tree tests compile unchanged when
`-Iinclude` is used.

## `virtual_npu_*` mapping

| Legacy call | New behavior |
|---|---|
| `virtual_npu_create` | reserves flat memory, creates device + stream |
| `virtual_npu_memory` | returns host pointer to flat memory |
| `virtual_npu_compute` | validates 64-bit bounds, queues a POD legacy command |
| `virtual_npu_process` | runs `vnpu_matmul_f32`, applies culling, updates counters |
| `virtual_npu_read_status` | returns accumulated ops/processed/culled/dispatches |
| `virtual_npu_reset/invalidate` | clears queue and counters |
| `virtual_npu_create_team/thread` | lightweight scheduler records |
| `virtual_npu_create_area` | 4 KiB-aligned bump allocator inside flat memory |
| bitmap calls | real per-page bit set/clear |

Opcode mapping:

- `NPU_OP_CPU`, `NPU_OP_VNPU_AUTO` -> `VNPU_UNIT_AUTO`
- `NPU_OP_VOODOO`, `NPU_OP_MERGED` -> `VNPU_UNIT_VOODOO`
- `NPU_OP_WILDCAT`, `NPU_OP_SHADER_*` -> `VNPU_UNIT_WILDCAT`
- `NPU_OP_FUSED_GEMM`, `NPU_OP_TENSOR_CORE` -> `VNPU_UNIT_CELL`

## `vgpu_npu_*` mapping

| Legacy call | New behavior |
|---|---|
| `vgpu_npu_create` | reserves 2 GiB virtually with `mmap(MAP_NORESERVE)` |
| `vgpu_npu_create_ex` | same with explicit size (new extension) |
| `readl/writel` | bounds-checked 32-bit flat memory access |
| clear/fill/blit | real memory commands |
| texture upload | copies payload into flat memory |
| bind TMU / draw triangle / swapbuffers | deterministic no-ops with counters |
| unit1 compute | real ALU ops with correct block culling |
| `vgpu_npu_matmul_npu` | real GEMM through the new runtime |
| status/results | accumulated from the compatibility layer |

Graphics calls are intentionally retained as no-ops: a renderer would drag PCem
back into the dependency graph and has no role in LLM inference.

## Sherlock compatibility

`sherlock_trace_start()` and `sherlock_trace_end()` now use thread-local
monotonic timestamps. `sherlock_get_trace()` returns real dispatch, memory,
compute, sync, bytes and ops values. `sherlock_dump_trace()` prints percentages
based on measured total time. The old 0.5 ms constant is gone.

## Known differences

- Legacy cull thresholds compare the raw 32-bit control word, matching the old
  tests. New code should use `tile_mask` with `WILDCAT` instead.
- `NPU_OP_CPU` is no longer slower on purpose; all legacy opcodes use the
  optimized runtime. Use `VNPU_FORCE_ISA=scalar` for a scalar baseline.
- Flat memory is reserved virtually; touching all 2 GiB will commit pages.
- Legacy calls are synchronous. New code should use streams/events.
