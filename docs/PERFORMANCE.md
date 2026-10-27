# Performance and measurement methodology

## Reference environment

- 2 logical Intel Xeon vCPUs (KVM guest)
- AVX2, FMA, AVX-512F/DQ/BW/VL, AVX-512 VNNI
- 48 KiB L1D, 1.25 MiB L2, 54 MiB L3
- GCC 14.2, C++17, `-O3 -fno-math-errno -ffp-contract=fast -funroll-loops`
- `make benchmark`, 5 iterations for GEMM, 3 for autotune/LLM block

The baseline is a naive scalar triple-loop `C = A*B` compiled in the same
binary. It is representative of an unoptimized engine path, **not** of MKL,
OpenBLAS, BLIS, oneDNN or a hand-tuned AVX-512 BLAS. The VM is noisy; exact
numbers vary between runs. `benchmarks/*.json` is authoritative for each run.

## Dense GEMM

### 512×512×512 (final benchmark run)

| Path | Time | GFLOP/s | Speedup vs scalar |
|---|---:|---:|---:|
| Scalar baseline | ~168 ms | ~1.6 | 1.0× |
| F32 AUTO | 3.76 ms | 71.3 | 42.2× |
| F32 Voodoo | 3.16 ms | 85.0 | 50.3× |
| F32 Wildcat | 2.69 ms | 99.8 | 59.1× |
| INT8 AUTO | 3.15 ms | 85.2 | 50.5× |
| INT8 Voodoo | 3.72 ms | 72.2 | 42.7× |
| INT8 Wildcat | 4.35 ms | 61.7 | 36.6× |
| Autotuned F32 | 4.38 ms | 61.3 | 38.3× |
| Autotuned INT8 VNNI | 2.02 ms | 132.8 | 85.5× |

The autotune and `bench` runs are separate processes with different cache state.
The best F32 path in the matrix was Wildcat; the best standalone autotuned INT8
path reached 132.8 GFLOP/s.

### 256×256×256

| Path | Time | GFLOP/s | Speedup |
|---|---:|---:|---:|
| F32 Voodoo | 0.620 ms | 54.1 | 29.5× |
| F32 Wildcat | 0.528 ms | 63.5 | 34.6× |
| INT8 Voodoo | 0.465 ms | 72.2 | 39.4× |
| INT8 Wildcat | 0.443 ms | 75.8 | 41.3× |

### 128×128×128

| Path | Time | GFLOP/s | Speedup |
|---|---:|---:|---:|
| F32 Voodoo | 0.206 ms | 20.4 | 12.4× |
| F32 Wildcat | 0.199 ms | 21.0 | 12.8× |
| INT8 Voodoo | 0.182 ms | 23.1 | 14.1× |
| INT8 Wildcat | 0.187 ms | 22.4 | 13.7× |

### 64×64×64

Tiny GEMMs are dispatch/latency-bound. Speedups range from ~0.7× to ~2.1×.
Use larger batches, fuse commands, or use the Cell unit with persistent streams.

## Observations

1. **The 12–20× target is exceeded on medium/large GEMMs.** On this VM,
   35–85× vs a naive scalar baseline is typical. Against optimized BLAS the
   gain will be smaller.
2. **INT8 pays off when K and N are large enough.** For 256³ and the autotuned
   512³ case VNNI is faster than F32; for 128³ dispatch and quantization
   overhead dominate.
3. **Cell is intentionally slower for large dense GEMM.** Its local-store DMA
   path is for small independent chunks and distributed workers, not peak
   FLOPs.
4. **Autotune matters.** It sweeps tile geometries and units; the winning
   INT8 configuration reached 132.8 GFLOP/s.
5. **Sherlock identified compute-bound behavior.** Arithmetic intensity was
   ~96 FLOP/byte on the GEMM benchmark. The next wins came from VNNI, larger N
   blocks, broadcast-A reuse and fewer dispatch calls.

## Synthetic transformer block

Command:

```bash
./bin/vnpu-cli llm-bench --hidden 512 --intermediate 1376 \
  --heads 8 --layers 2 --prefill 32 --decode 1 --dtype both \
  --iterations 3 --json benchmarks/llm-block.json \
  --trace benchmarks/llm-trace.json
```

Final results:

| Phase | F32 | INT8 |
|---|---:|---:|
| prefill 32 tokens | 16.33 ms, 1,959 tok/s, 25.0 GFLOP/s | 11.68 ms, 2,741 tok/s, 35.0 GFLOP/s |
| decode 1 token | 11.31 ms, 88.4 tok/s, 1.12 GFLOP/s | 10.42 ms, 96.0 tok/s, 1.21 GFLOP/s |

Sherlock trace for that run:

- 420 events, ~190 ms total, 100% compute bucket;
- MATMUL_F32 and MATMUL_I8 dominate;
- QUANTIZE_I8 remains a visible cost in the INT8 path;
- RMSNorm/RoPE/SwiGLU/ADD/ATTENTION together are smaller but non-negligible.

The biggest remaining bottleneck is activation quantization in the INT8 path —
tens of milliseconds across the benchmark depending on run. Pre-quantizing
hidden states or keeping residual streams quantized would remove most of that
cost.

## Cluster and speculation

- Two loopback nodes, 256³ GEMM: correct result, 1.35× effective worker
  parallelism, 0.86× wall-clock vs local. Network overhead dominated.
- Distributed speculative demo: 20 tokens in 13.8 ms, 16/20 accepted,
  theoretical 1.08–1.47× with a toy model. Real gains require a cheap draft
  head and a single batched verifier round trip.

## Reproducing

```bash
make clean
make -j$(nproc)
make benchmark
cat benchmarks/gemm.json
cat benchmarks/autotune-f32.json
cat benchmarks/autotune-i8.json
cat benchmarks/llm-block.json
cat benchmarks/cluster-demo.txt
cat benchmarks/specdemo.txt
```

Chrome traces:

- `benchmarks/sherlock-trace.json` – GEMM units/dtypes;
- `benchmarks/llm-trace.json` – transformer block.

Summarize a trace:

```bash
python3 scripts/trace_report.py benchmarks/sherlock-trace.json 20
```

## Caveats

- These are synthetic random matrices, not real model weights. Quantization
  error and cache behavior will differ.
- No tokenizer, sampling, KV cache or model loader is included.
- AVX-512 can downclock on some CPUs; measure on your target instance.
- Network v1 is plaintext TCP and native-endian.
- Scalar baseline speedup is not a fair comparison against optimized BLAS.
