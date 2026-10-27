Virtual NPU — batched/tensor compute NPU (CPU / Voodoo / Wildcat)
==================================================================

This is a self-contained virtual NPU used for accelerator-style benchmarking.
It is *not* hardware and not the PCem video-core NPU. It models three compute
styles against the same math so you can compare dispatch/culling cost and
correctness:

- CPU direct: reference tiled matmul kernel.
- Voodoo: batched command-stream style — one batch dispatch, per-tile base
  reuse, explicit tile walk with triangle clamping at tile edges.
- Wildcat: programmable compute unit style, with an optional hierarchical-Z-style
  tile culling filter that can skip whole tiles when their control value is low.

The NPU exposes:
- a flat memory space for A, B, C, and optional cull control
- a command queue with `virtual_npu_compute()` + `virtual_npu_process()`
- status/accumulators: ops, processed, culled, dispatches, batches

The Unit 1 compute instruction set is currently mapped to:
- 0x01 ADD
- 0x02 MUL
- 0x03 XOR
- 0x04 BLEND0
- 0x05 BLEND1
- 0x06 BLEND2
- 0x07 BLEND3
- bit 8 accumulation toggle

The compute loop is batched by count and operates on packed 32-bit words.

Build
-----
```
cd npu
g++ -std=c++17 -O2 -Wall -Wextra -o test_matmul_batched test_matmul_batched.c virtual_npu.c
```

Run matmul comparison
---------------------
```
cd npu
./test_matmul_batched
```

It runs C = A * B for a few M×K×N shapes, compares NPU Voodoo vs CPU and NPU
Wildcat vs CPU, and prints time, TFLOPS, dispatches/batches/ops/processed/culled.

What the numbers mean
---------------------
Because this NPU still runs in ordinary C on the same CPU, the "NPU" paths are
software models of different compute styles, not a separate physical device. The
comparison is therefore meaningful for:
- correctness of the batched/tensor-style kernels
- dispatch/culling behavior and overhead
- tile math and edge clamping

For a more realistic accelerator comparison, the next step is to give the NPU
real batched/tensor math primitives (for example dot-product/FMA-style ops and a
sensible memory layout), rather than per-element matmul inside each tile.

Current status
--------------
- Matmul correctness checked against a CPU reference for the shapes in
  `test_matmul_batched.c`.
- The batched/tensor interfaces now take explicit (M, K, N) tile parameters so
  tile math is no longer derived incorrectly from a flattened count.
