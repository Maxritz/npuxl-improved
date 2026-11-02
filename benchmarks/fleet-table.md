# Fleet Benchmark Table — RX 6700 XT (gfx1031, 12GB VRAM, 6C/12T Zen3)

Hardware: RX 6700 XT (gfx1031, wave32, 20 MPs, 12 GiB VRAM), Ryzen 5 5600X, DDR4-3000 (48 GB/s ceiling), ROCm 7.1 HIP 7.16.26332, Windows 11.
Engines: `build-hip-gfx1031/bin/strata-poly.exe` (15:41) + `strata.exe` (13:50).
vNPU bridge: MXFP4 table-driven INT8 (ggml kvalues_fp4), pack-once-per-blob, default OFF, opt-in via STRATA_VNPU_EXPERIMENTAL=1.

## Results

| # | Model | Command / Switches | Prefill (tok/s) | Decode (tok/s) | Coherence |
|---|-------|-------------------|-----------------|----------------|-----------|
| 1 | Qwen3.5-35B-A3B-UD-Q4_K_XL (18.3 GB, qwen35moe) | `strata-poly --model C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --max-context 512 --tokens 1..10 --max-new 10` (bridge OFF, default) | **16.6** (10 tok) | **30.4** | ✅ CORRECT: `4838,1665,15,16,17,18,19,20,21,22` |
| 2 | Qwen3.5-35B-A3B-UD-Q4_K_XL (long prompt) | `strata-poly --model ... --max-context 4096 --tokens <311 tok> --max-new 10` (bridge OFF) | **50.1** (311 tok) | **13.0** | ✅ CORRECT: `220,220,16,15,15,15,15,15,15,15` |
| 3 | Qwen3.5-35B-A3B-UD-Q4_K_XL (bridge ON, experimental) | Same as #1 + `STRATA_VNPU_EXPERIMENTAL=1` | 13.4 (10 tok) | 12.5 | ❌ GARBAGE: `4838,1665,4,5,6,363,198,2,198,2` — INT8 double-quant too lossy on MXFP4 |
| 4 | Qwen3.8-Flash-Next 78GB (qwen4exp) | `strata --pack C:\pack-qwen38 --native E:\x\qwen3.8-flash-next-reap-288-Q4_K_M.gguf --ple-gguf <same> --expert-profile C:\pack-qwen38\expert-profile-288.bin --max-context 128 --expert-cache 0 --no-pool --tokens 1..5 --max-new 2 --spec 4 --prefill 16 --vram-reserve-mib 512` | BLOCKED | BLOCKED | ❌ VRAM: "GEMM scratch does not fit" even at 128 ctx. 12GB VRAM insufficient (needs 24GB+). Engine loads (48 layers, 288 experts/layer, profile 13824 pairs) but prefill buffers don't fit. Pack: C:\pack-qwen38 (1.48GB dense, 288-expert native_experts.txt, tokenizer). 288-expert profile built: C:\pack-qwen38\expert-profile-288.bin (110KB, 13824 pairs). |
| 5 | ornith9b (qwen35 dense, 4.6 GB) | `strata-poly --model C:\x\ornith9b.gguf` | BLOCKED | BLOCKED | ❌ `qwen35` dense arch — strata-poly rejects: "which this engine does not run" |
| 6 | Qwen3.8-4B-Q8_0 (qwen35 dense, 4.3 GB) | `strata-poly --model C:\x\Qwen3.8-4B-Q8_0.gguf` | BLOCKED | BLOCKED | ❌ `qwen35` dense arch — strata-poly rejects |
| 7 | flint-q4_1 (flint, 23 GB) | `strata-poly --model C:\x\flint-q4_1.gguf` | BLOCKED | BLOCKED | ❌ `flint` arch — no engine target. Flint added to llama.cpp PR #30928 (Oct 2026). Submodule at a6de9af (pre-flint). Needs full engine implementation (49 layers, 256 experts, sliding-window + full attention, sigmoid top-6). |

## Summary

- **Only Qwen3.5-35B runs** on this HIP build (strata-poly, qwen35moe arch).
- Bridge OFF (exact path): 16.6 prefill / 30.4 decode short, 50.1 prefill / 13.0 decode long. Correct tokens.
- Bridge ON (experimental): 2-3x slower AND produces garbage. INT8 double-quant on MXFP4 too lossy. **PARKED** — default OFF.
- Flash: engine loads but 12GB VRAM insufficient for prefill. Needs 24GB+ card. 288-expert profile built (C:\pack-qwen38\expert-profile-288.bin).
- ornith/Qwen3.8-4B: dense `qwen35` not supported by strata-poly.
- flint: needs new engine implementation + updated llama.cpp submodule.

## vNPU Bridge Status
- Code: `src/poly/vnpu_poly.cpp` — table-driven MXFP4 expanders (ggml kvalues_fp4), pack-once-per-blob, zero-init combine fix in `src/poly/cpu_experts.cpp`.
- Parity: **FAILS** on real weights (garbage tokens). Bridge stays default OFF until parity harness (vnpu_poly_parity vs cpu_experts on real blobs) passes.
- Direct vs cmd path: bitwise identical on synthetic data. Real-weight parity unproven.

## 288-Expert Profile
- Built: `C:\pack-qwen38\expert-profile-288.bin` (110KB, 48 layers × 288 experts = 13824 pairs)
- Method: `python tools/make_profile.py --n-expert 288 --no-base` (filled ranking, no real routing traces)
- A real profile needs `--dump-routing` traces from running the model (requires 24GB+ VRAM)

## NPU Push (npuxl-improved, commit 17e8d8d)
- Direct fast path: `vnpu_gemm_f32_direct` (calling-thread, no command/stream/tiles/trace overhead)
- Shard-by-expert modes 7/8 in pool.cpp with auto-select via STRATA_SHARD_EXPERTS
- Table-driven MXFP4/NVFP4 expanders using ggml kvalues_fp4
- Pack-once-per-blob INT8 cache (pointer-keyed, no per-call dequant)
- Cold DRAM: INT8 38.8 GB/s 80.8%, F32 36.8 GB/s 76.7% of DDR4-3000 48.0 GB/s ceiling
- Prefill INT8: 388.7 GFLOP/s
- Shard wins: 1.97x T=1, 1.55x T=8, 1.13-1.18x T=32