# Roadmap

## v1.1 — engine plugins

- A compilable `ggml_backend_vnpu` plugin that maps `ggml_tensor` ops to
  `vnpu_command_t` batches.
- An ONNX Runtime custom execution provider shim for MatMul, LayerNormalization,
  RotaryEmbedding and MultiHeadAttention.
- A stable `vnpu_model_weight_t` descriptor so engines can register externally
  allocated host/GPU buffers without copying.
- More fused epilogues: bias+SiLU+mul, bias+GELU+add, residual add.

## v1.2 — quantization

- INT4 / grouped INT4 weight packing with VNNI-style decode.
- BF16 and FP16 activation GEMM on AVX-512-BF16 and ARM NEON.
- Per-group scales and zero points for asymmetric activations.
- Accuracy harness that reports perplexity deltas against F32.
- Pre-quantized activation path to remove `QUANTIZE_I8` from decode.

## v1.3 — GPU mapping

`VNPU_UNIT_GPU` is already reserved in the public ABI. Planned work:

- `vnpu_buffer_t` gains a backend device pointer while keeping
  `vnpu_buffer_host_ptr()` valid.
- `MATMUL_F32/I8` dispatch to CUDA/ROCm/oneAPI kernels when a GPU build is
  enabled.
- Heterogeneous scheduler sends large prefill GEMMs to GPU, decode GEMMs to
  Cell/VNNI CPU workers, and attention to whichever backend has the KV cache.
- Cluster weights can be sharded by layer instead of duplicated.

## v1.4 — network

- Explicit little-endian wire encoding and protocol negotiation.
- TLS or QUIC transport option.
- RDMA/libfabric backend for large activations.
- KV-cache-aware routing and layer/pipeline parallelism.
- Draft-model and verifier placement optimizer based on measured RTT/FLOPs.

## v2.0 — speculative decoding

- Logits callbacks that return both probabilities and top-k tokens.
- Exact stochastic acceptance with `min(1, p/q)`.
- Medusa/Eagle-style tree attention verification.
- Dynamic `gamma` tuning from acceptance rate.
- Integration with real MTP draft heads rather than toy logits.

## Continuous work

- Autotune per model shape at load time and cache the result on disk.
- NUMA-aware thread pools and cgroup quota detection.
- More ARM NEON kernels and SVE detection.
- Windows DLL and ARM64 CI.
- Fuzzing against scalar references for every new kernel.
