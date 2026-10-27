# Architecture

## 1. Design goals

The runtime is a **software device**. An LLM engine should be able to treat it
like a small accelerator: allocate buffers, push commands into a stream, wait
on events, read status registers and collect telemetry. The implementation is
free to use CPU threads, SIMD, local stores, culling and network workers.

Non-goals for v1: pretending to be a physical NPU, emulating a PCI device,
implementing a full model/tokenizer, or depending on a vendor SDK.

## 2. Layer diagram

```text
  LLM engine / Python / CLI / legacy test
        | C ABI (vnpu.h) or backend vtable
        v
  Command processor (vnpu_stream_submit)
        | fixed POD command packets
        v
  Scheduler (choose_unit / build_tiles_generic)
        +----------------+----------------+----------------+
        | Voodoo FIFO    | Wildcat cull   | Cell DMA       |
        +----------------+----------------+----------------+
        v
  Kernel dispatcher (runtime CPUID)
        +---------+---------+---------+---------+
        | scalar  | AVX2    | AVX512  | VNNI    |
        +---------+---------+---------+---------+
        v
  Persistent thread pool + per-worker local store
        |
        +--> Sherlock trace / status registers / events
        |
        +--> cluster split ------------------> vnpu-node TCP workers
```

## 3. Device object

`vnpu_device_t` owns:

- a `CpuFeatures` snapshot (ISA, SIMD width, cache sizes, memory);
- a persistent `ThreadPool` for compute;
- a vector of 256 KiB local stores, one per worker;
- a `TraceState` with per-opcode aggregates and Chrome trace events;
- 16 virtual 32-bit MMIO registers;
- atomic status counters: commands, kernel/dispatch/memory/sync time, bytes,
  FLOPs, tiles, culls, DMA, interrupts, errors.

`vnpu_mmio_read32()` and `vnpu_mmio_write32()` expose ID, version, control,
status, error, command head/tail, doorbell, IRQ status/mask, active CUs,
feature bits. Writing the doorbell raises IRQ 0; completing a command batch
raises IRQ 1 and invokes the optional interrupt callback.

## 4. Streams and events

Each `vnpu_stream_t` owns one worker thread and a FIFO of `CommandBatch`
objects. Submission is non-blocking; `vnpu_stream_sync()` waits until the queue
is empty and all pending commands have retired. `vnpu_event_t` is a timed
completion object. Commands inside one stream are ordered; different streams
execute concurrently on the shared compute pool.

Convenience functions such as `vnpu_matmul_f32()` submit one command and then
sync. High-performance engines should batch commands and sync once per layer or
per token step.

## 5. Command packet

`vnpu_command_t` is a fixed-size POD structure:

```text
uint32 opcode, flags
uint64 id
union {
  matmul_f32, matmul_i8, rmsnorm, softmax, rope, binary, unary,
  swiglu, attention, quantize_i8, pack_weights_i8, mtp_head,
  copy, memset, raw[192]
}
```

The union is intentionally flat. No command owns memory, so engines can build
batches on the stack or in pinned host memory. Future operators are added as
new opcodes and new union fields while `VNPU_ABI_VERSION` stays 1.

## 6. Compute units

### Voodoo

- static tile order: M outer, N inner;
- persistent row/col bases inside the microkernel;
- tile geometry from the device or autotuner;
- best for large dense prefill GEMMs.

### Wildcat

- interleaved N-major tile order for work stealing;
- optional `tile_mask` culls whole zero tiles;
- smaller M tile (default ≤32) and K window (≤128);
- good for sparse/conditional execution and MoE-style masks.

### Cell

- each worker gets an aligned 256 KiB local store;
- A/B panels are explicitly copied (DMA) into that store in K windows;
- register-blocked accumulation stays inside the local store;
- ideal for remote workers, tiny independent tasks and cache-sensitive shapes.

`VNPU_UNIT_AUTO` chooses Cell for low-latency/decode shapes, Wildcat when a
cull mask is present, otherwise Voodoo.

## 7. GEMM kernels

### F32 NN

`A[M,K] * B[K,N] -> C[M,N]`.

- AVX-512: 4 rows × 16 columns in registers, one B vector per K, four FMA
  accumulators.
- AVX2: 4 rows × 8 columns.
- NEON: 4 rows × 4 columns.
- Scalar: correctness oracle and fallback.
- Tails and transposed A fall back to scalar tiles.

### F32 NT (transposed B)

For decode, weights are often `[N,K]`. A vectorized dot-product kernel streams
both A and B rows contiguously and horizontally reduces one accumulator.

### INT8 VNNI

- symmetric per-row activation scales, per-column weight scales;
- weights packed as `[N/16][K/4][16][4]` signed bytes;
- activations converted to `u8 = i8 + 128`;
- `_mm512_dpbusd_epi32` accumulates 4 products per 32-bit lane;
- precomputed `sum(B)` removes the `128 * sum(B)` bias;
- 4 M rows and up to 4 N blocks (64 columns) are processed together;
- for K ≥ 512 and wide N, broadcast-A vectors are precomputed in local store.

## 8. Transformer operators

- RMSNorm: vectorized sum of squares, one reciprocal sqrt, vectorized scale.
- Softmax: vectorized max, fast `exp2` polynomial, normalize.
- RoPE: half-rotation or interleaved GPT-J rotation, optional positions.
- SwiGLU/GELU: fast sigmoid/tanh approximations with vectorized exp.
- Attention: streaming online softmax over K blocks with causal/additive mask;
  vectorized dot products and V accumulation.

## 9. Tracing

`ScopedTrace` wraps each command. It records start, duration, opcode, unit,
worker, bytes read/written and FLOPs. `TraceState` aggregates the four buckets
(compute/memory/dispatch/sync), per-opcode totals and the slowest events.
`vnpu_trace_dump()` prints a roofline-style report and
`vnpu_trace_write_chrome()` emits Chrome Trace JSON.

Tracing adds roughly 25–45 ns/event on the reference VM. Disable it for final
timing; autotune does this automatically.

## 10. Cluster

`vnpu_cluster_t` holds weighted local workers and remote workers. Local workers
own a stream on an existing device; remote workers own a persistent TCP
connection to a `vnpu-node`. Weights are cached by a stable FNV-1a key; matmul
splits M rows by worker weight and merges results into the caller’s C buffer.

Speculative decoding uses generic logits callbacks, so draft and verifier can
be local, remote, or an MTP head. The cluster object only records job stats.

## 11. GPU extension point

The public ABI already has `VNPU_UNIT_GPU` and
`VNPU_FEATURE_GPU_FUTURE`. A future backend can:

- keep the same `vnpu_command_t`;
- map `vnpu_buffer_t` to device allocations;
- implement `MATMUL_F32/I8` with cuBLAS/MIOpen/oneDNN;
- keep Sherlock events, cluster splitting and speculative orchestration
  unchanged.

No ABI break is required.
