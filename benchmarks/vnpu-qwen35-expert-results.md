# vNPU Qwen3.5 expert validation — 2026-11-02 (Windows, Ryzen 5 5600X 6C/12T, RX 6700 XT gfx1031)

Box: AMD Ryzen 5 5600X (6C/12T), ROCm at C:\Rocm10 (not used by these CPU/vnpu runs).
Model: C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (embd 2048, ff 512, top_k 8, 40 layers).
Repo B: C:\npuxl-improved, binaries in C:\npuxl-improved\build-win\.
Stray engine procs killed before every run (`unbiased-strata,strata-poly,strata,strata-device`, none running).

New test source: C:\npuxl-improved\tools\vnpu_qwen35_expert.cpp, binary build-win/vnpu_qwen35_expert.exe.
No Strata sources touched. No server started. No commits made.

## 1. strata_schedule_compare.exe — [R row-split 6wk/1dev] vs [E expert-shard 6dev] (8 experts, whole layer)

Same work both ways: 8 experts x [gate Mx2048x512 + up Mx2048x512 + SiLU + down Mx512x2048],
H=2048 FF=512 top_k=8, F32 voodoo, 6 workers. Source: tools/strata_schedule_compare.cpp.
R = each worker does rows r0..r1 of ALL 8 experts (Strata pool modes 3/4 today).
E = worker w owns experts w, w+6, ... whole (1 vnpu device per core).

| prefill T | iters | FLOP | R ms | R GFLOP/s | E ms | E GFLOP/s | speedup E/R |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.05 GFLOP | 5.707 | 8.8 | 2.812 | 17.9 | **2.03x** |
| 8 | 5 | 0.40 GFLOP | 7.379 | 54.6 | 4.712 | 85.4 | **1.57x** |
| 32 | 5 | 1.61 GFLOP | 11.009 | 146.3 | 10.142 | 158.8 | **1.09x** |

Raw:
```
T=1 iters=5 flop=0.05 GFLOP
[R row-split 6wk/1dev] 5.707 ms  8.8 GFLOP/s
[E expert-shard 6dev]  2.812 ms  17.9 GFLOP/s  speedup 2.03x
T=8 iters=5 flop=0.40 GFLOP
[R row-split 6wk/1dev] 7.379 ms  54.6 GFLOP/s
[E expert-shard 6dev]  4.712 ms  85.4 GFLOP/s  speedup 1.57x
T=32 iters=5 flop=1.61 GFLOP
[R row-split 6wk/1dev] 11.009 ms  146.3 GFLOP/s
[E expert-shard 6dev]  10.142 ms  158.8 GFLOP/s  speedup 1.09x
```
Consistent with prior results file (1.97x / 1.52–1.57x / 1.13–1.18x): E wins everywhere,
advantage shrinks as T grows (arithmetic intensity rises, weight reuse across tokens).

## 2. strata_offload_6npu.exe --prefill 32 --iters 5 (8 experts x 1 layer, decode + prefill)

backend=AVX2+FMA threads=6. Source: tools/strata_offload_6npu.cpp.
A = single NPU with 6 threads; B = 6x NPU with 1 thread each, 8 experts sharded round-robin.
```
[A single-NPU-6thr] decode 1tok x8exp x1layer: 4.217 ms  11.9 GFLOP/s
[A single-NPU-6thr] prefill 32tok x8exp x1layer: 16.698 ms  1916.4 tok/s  96.5 GFLOP/s
[A single-NPU-6thr] full-model est (40 layers): decode 5.9 tok/s  prefill-32 47.9 tok/s
[B 6xNPU-1thr] decode 1tok x8exp x1layer: 2.867 ms  17.6 GFLOP/s  speedup-vs-A 1.47x
[B 6xNPU-1thr] prefill 32tok x8exp x1layer: 10.880 ms  2941.1 tok/s  148.0 GFLOP/s  speedup 1.53x
[B 6xNPU-1thr] full-model est (40 layers): decode 8.7 tok/s  prefill-32 73.5 tok/s
```
6x1-thread sharded beats 1x6-thread by ~1.5x on both decode and prefill-32.
Prior file quoted 1.45x/1.46x — reproduced (this run 1.47x/1.53x, within noise).

## 3. vnpu-cli bench (decode + prefill GEMM shapes, both dtypes, --iterations 5)

`vnpu-cli bench --shape 1x2048x512 --shape 1x512x2048 --shape 32x2048x512 --shape 32x512x2048 --dtype both --iterations 5`.
speedup = vs faster of naive scalar triple loop / compiler-vectorisable plain-C loop.
```
shape          dtype unit              ms    GFLOP/s    speedup    iters check
1x2048x512     f32   auto          0.0994     21.107      0.75x        5 OK
1x2048x512     f32   voodoo        0.0828     25.322      0.90x        5 OK
1x2048x512     f32   wildcat       0.0698     30.028      1.07x        5 OK
1x2048x512     f32   cell          0.2715      7.724      0.28x        5 OK
1x2048x512     i8    auto          0.0718     29.208      1.04x        5 OK
1x2048x512     i8    voodoo        0.0793     26.452      0.94x        5 OK
1x2048x512     i8    wildcat       0.0630     33.267      1.19x        5 OK
1x2048x512     i8    cell          0.2256      9.294      0.33x        5 OK
1x512x2048     f32   auto          0.1219     17.198      0.59x        5 OK
1x512x2048     f32   voodoo        0.0837     25.056      0.86x        5 OK
1x512x2048     f32   wildcat       0.0741     28.309      0.97x        5 OK
1x512x2048     f32   cell          0.2266      9.255      0.32x        5 OK
1x512x2048     i8    auto          0.0754     27.828      0.95x        5 OK
1x512x2048     i8    voodoo        0.0611     34.312      1.18x        5 OK
1x512x2048     i8    wildcat       0.0735     28.540      0.98x        5 OK
1x512x2048     i8    cell          0.1829     11.467      0.39x        5 OK
32x2048x512    f32   auto          0.3696    181.552      6.19x        5 OK
32x2048x512    f32   voodoo        0.3414    196.581      6.70x        5 OK
32x2048x512    f32   wildcat       0.3065    218.952      7.47x        5 OK
32x2048x512    f32   cell          2.7817     24.125      0.82x        5 OK
32x2048x512    i8    auto          0.2812    238.652      8.14x        5 OK
32x2048x512    i8    voodoo        0.2527    265.546      9.06x        5 OK
32x2048x512    i8    wildcat       0.2519    266.368      9.09x        5 OK
32x2048x512    i8    cell          1.6533     40.591      1.38x        5 OK
32x512x2048    f32   auto          0.3031    221.423      7.55x        5 OK
32x512x2048    f32   voodoo        0.2674    250.931      8.55x        5 OK
32x512x2048    f32   wildcat       0.2612    256.945      8.76x        5 OK
32x512x2048    f32   cell          2.7988     23.978      0.82x        5 OK
32x512x2048    i8    auto          0.2560    262.124      8.94x        5 OK
32x512x2048    i8    voodoo        0.2462    272.579      9.29x        5 OK
32x512x2048    i8    wildcat       0.2179    308.037     10.50x        5 OK
32x512x2048    i8    cell          1.6893     39.725      1.35x        5 OK
```
Takeaway: prefill GEMMs run 180–310 GFLOP/s (compute-efficient); decode GEMMs run ~20–34
GFLOP/s (memory-bound — exactly where expert-sharding wins). voodoo/wildcat are the fast
units; `cell` is the slow reference and stays slow. i8 beats f32 ~1.2–1.4x on prefill.
Consistent with prior file's numbers (this run slightly faster on prefill, within noise).

## 4. NEW: vnpu_qwen35_expert — ONE Qwen3.5 expert, whole vs FF-row-sharded

Source: tools/vnpu_qwen35_expert.cpp. H=2048 FF=512, gate/up/down F32 random.
Per M: gate (Mx2048x512) + up (Mx2048x512) + SiLU + down (Mx512x2048), FLOP = 6*M*H*FF.
(A) single device 6 threads, whole expert. (B) 6 devices x 1 thread, FF rows sharded
(device d owns gate/up/down rows [d*per,(d+1)*per), down partials sum-reduced on host).
Weight panels pre-packed once at setup (steady-state, untimed); timed = GEMMs + SiLU + reduce.
Build: `g++ -std=c++17 -O3 -Iinclude -Isrc tools/vnpu_qwen35_expert.cpp -o build-win/vnpu_qwen35_expert.exe -Lbuild-win -lvnpu -lpthread -lws2_32`.
maxdiff = max |outA - outB| (bitwise-parity check of the two schedules).

Run 1 (--iters 5):
```
M    A_ms(6thr)   A_GFLOP/s  B_ms(6x1thr) B_GFLOP/s  speedup   maxdiff
1    0.350        18.0       0.519        12.1       0.68     x 1.31e-06
8    0.607        83.0       0.890        56.5       0.68     x 1.43e-06
32   2.914        69.1       2.427        82.9       1.20     x 2.15e-06
```
Run 2 (--iters 5, repeat):
```
M    A_ms(6thr)   A_GFLOP/s  B_ms(6x1thr) B_GFLOP/s  speedup   maxdiff
1    0.323        19.5       0.526        12.0       0.61     x 1.31e-06
8    0.631        79.8       0.869        57.9       0.73     x 1.43e-06
32   2.049        98.3       2.050        98.2       1.00     x 2.15e-06
```
maxdiff ~1e-6 in all cases: the two schedules agree to fp32-noise — correctness holds either way.

Reading: splitting ONE expert's tiny GEMMs across 6 devices LOSES on decode (0.6–0.7x at
M=1/M=8) — thread-launch + host-side down-partial reduce dominate a 1x2048x512 GEMM that
takes ~0.07–0.12 ms on its own. At M=32 the split breaks even to 1.2x (run-to-run noise:
2.43 vs 2.05 ms on B — the workload is just big enough that extra cores start to pay).
Contrast with sections 1–2: sharding whole experts across workers (8 experts over 6 cores)
WINS 1.5–2.0x on the same decode shapes. The granularity is the whole story.

## 5. Conclusion: which schedule wins, and what it means for pool.cpp modes 7/8 + the vnpu bridge

1. **Shard ACROSS experts (whole expert per worker) wins everywhere: 2.03x (T=1) → 1.57x (T=8) → 1.09x (T=32).** Reproduced vs prior file. The 6x1-thread topology beats 1x6-thread by ~1.5x on decode and prefill-32 (§2).
2. **Split WITHIN one expert (FF-row shard) loses on decode: 0.61–0.73x at M=1/M=8, ~1.0–1.2x at M=32** (§4). Same GEMM shapes, opposite result — because per-GEMM overhead (thread wake, extra dispatches, partial-sum reduce) exceeds the compute of a single decode GEMM.
3. Rule for pool.cpp modes 7/8: **shard at expert granularity, never below it.** Task = one whole expert (gate/up rows 0..FF, SiLU, quant, down rows 0..H for that expert only). Each blob read once, stays in its core's cache, no cross-worker reduce. With top_k=8 and 6 workers the balance is 1–2 experts/worker — fine. Within-expert row-splitting (today's modes 3/4) should be the fallback only when n_active_experts < n_workers or M is large (prefill), where §4 shows it stops losing.
4. For the vnpu bridge (sibling's src/kernels/cpu/vnpu_expert.{hpp,cpp}): one vnpu device with num_threads=1 per pool worker; dispatch whole experts to worker devices; keep Strata's `s2_expert_*`/`native_*` row kernels as fallback and parity reference (maxdiff ~1e-6 here says parity is achievable). Gate the swap on `vnpu-cli bench --shape 1x2048x512 --shape 1x512x2048` (decode) exactly as re-run in §3, then before/after with `strata_schedule_compare --prefill {1,8,32}` (§1) on identical workloads with the correctness check re-run — per the standing rule.
