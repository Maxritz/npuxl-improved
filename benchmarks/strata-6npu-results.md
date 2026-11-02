# Strata 6-NPU CPU-expert offload test — 2026-11-02 (Windows, Ryzen 5 5600X 6C/12T, RX 6700 XT gfx1031)

Box: AMD Ryzen 5 5600X (6C/12T), ROCm at C:\Rocm10 (not used by these CPU/vnpu runs).
Repo A: C:\Users\rr\Desktop\UnbiasedStrata-main (Strata HIP port, Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf)
Repo B: C:\npuxl-improved (virtual NPU), MinGW binaries in C:\npuxl-improved\build-win\

Stray engine procs killed before runs: `Stop-Process unbiased-strata,strata-poly,strata,strata-device -Force` (none running).

## 1. strata_schedule_compare.exe — [R row-split 6wk/1dev] vs [E expert-shard 6dev]

Same work both ways: 8 experts x [gate Mx2048x512 + up Mx2048x512 + SiLU + down Mx512x2048],
H=2048 FF=512 top_k=8, F32 voodoo, 6 workers. Source: tools/strata_schedule_compare.cpp.
R = each worker does rows r0..r1 of ALL 8 experts (Strata pool mode 3/4 today).
E = worker w owns experts w, w+6, ... whole (1 vnpu device per core).

| prefill T | iters | FLOP | R row-split ms | R GFLOP/s | E expert-shard ms | E GFLOP/s | speedup E/R |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.05 GFLOP | 5.709 | 8.8 | 2.894 | 17.4 | **1.97x** |
| 8 | 5 | 0.40 GFLOP | 7.904 (also 7.403 on repeat) | 50.9 (54.4) | 5.208 (also 4.721) | 77.3 (85.3) | **1.52–1.57x** |
| 32 | 5 | 1.61 GFLOP | 10.952 (also 11.239) | 147.1 (143.3) | 9.691 (also 9.504) | 166.2 (169.5) | **1.13–1.18x** |

Raw:
```
T=1 iters=5 flop=0.05 GFLOP
[R row-split 6wk/1dev] 5.709 ms  8.8 GFLOP/s
[E expert-shard 6dev]  2.894 ms  17.4 GFLOP/s  speedup 1.97x
T=8 iters=5 flop=0.40 GFLOP
[R row-split 6wk/1dev] 7.904 ms  50.9 GFLOP/s
[E expert-shard 6dev]  5.208 ms  77.3 GFLOP/s  speedup 1.52x
T=32 iters=5 flop=1.61 GFLOP
[R row-split 6wk/1dev] 10.952 ms  147.1 GFLOP/s
[E expert-shard 6dev]  9.691 ms  166.2 GFLOP/s  speedup 1.13x
```
(repeat of the --prefill 8/32 pair gave R=7.403/E=4.721 speedup 1.57x at T=8, R=11.239/E=9.504 speedup 1.18x at T=32 — same conclusion.)

## 2. strata_offload_6npu.exe --prefill 32 --iters 5

backend=AVX2+FMA threads=6. Source: tools/strata_offload_6npu.cpp.
A = single NPU with 6 threads; B = 6x NPU with 1 thread each. F32 replay of 8 experts x 1 layer.
```
[A single-NPU-6thr] decode 1tok x8exp x1layer: 4.460 ms  11.3 GFLOP/s
[A single-NPU-6thr] prefill 32tok x8exp x1layer: 15.294 ms  2092.4 tok/s  105.3 GFLOP/s
[A single-NPU-6thr] full-model est (40 layers): decode 5.6 tok/s  prefill-32 52.3 tok/s
[B 6xNPU-1thr] decode 1tok x8exp x1layer: 3.081 ms  16.3 GFLOP/s  speedup-vs-A 1.45x
[B 6xNPU-1thr] prefill 32tok x8exp x1layer: 10.469 ms  3056.6 tok/s  153.8 GFLOP/s  speedup 1.46x
[B 6xNPU-1thr] full-model est (40 layers): decode 8.1 tok/s  prefill-32 76.4 tok/s
```
So sharding 8 routed experts over 6x1-thread NPUs beats 1x6-thread NPU by ~1.45x on both decode and prefill-32,
consistent with the schedule-compare result at T=8 (1.5x). Earlier session's F32 replay quoted decode
4.2ms->2.8ms / prefill-32 15ms->10ms per layer — reproduced within run-to-run noise.

## 3. vnpu-cli bench (decode + prefill shapes, both dtypes)

`vnpu-cli bench --shape MxKxN --iterations 5`. Best unit shown; `cell` is the slow reference.
speedup = vs naive scalar triple loop.

Decode shapes:
```
shape          dtype unit              ms    GFLOP/s    speedup    iters check
1x2048x512     f32   auto          0.1106     18.955      1.08x        5 OK
1x2048x512     f32   voodoo        0.0884     23.718      1.36x        5 OK
1x2048x512     f32   wildcat       0.0869     24.122      1.38x        5 OK
1x2048x512     f32   cell          0.2349      8.927      0.51x        5 OK
1x2048x512     i8    auto          0.0687     30.508      1.75x        5 OK
1x2048x512     i8    voodoo        0.0644     32.544      1.86x        5 OK
1x2048x512     i8    wildcat       0.0739     28.378      1.62x        5 OK
1x2048x512     i8    cell          0.2283      9.184      0.53x        5 OK
1x512x2048     f32   auto          0.1127     18.612      0.99x        5 OK
1x512x2048     f32   voodoo        0.1000     20.963      1.12x        5 OK
1x512x2048     f32   wildcat       0.0886     23.665      1.26x        5 OK
1x512x2048     f32   cell          0.2361      8.884      0.47x        5 OK
1x512x2048     i8    auto          0.0629-0.0715 29.3-33.3 1.57-1.78x  5 OK (voodoo best 0.0629/33.32/1.78x)
1x512x2048     i8    cell          0.1872     11.203      0.60x        5 OK
```
Prefill shapes:
```
32x2048x512    f32   wildcat(best) 0.3102    216.327      7.41x        5 OK (voodoo 0.3514/191.0/6.54x)
32x2048x512    i8    wildcat(best) 0.2563    261.837      8.97x        5 OK (voodoo 0.3040/220.8/7.57x)
32x2048x512    f32/i8 cell         2.6703/1.6436 25.1/40.8 0.86x/1.40x 5 OK
32x512x2048    f32   voodoo(best)  0.2845    235.867      7.92x        5 OK (wildcat 0.3204/209.5/7.03x)
32x512x2048    i8    wildcat(best) 0.2163    310.201     10.41x        5 OK (voodoo 0.2403/279.3/9.37x)
32x512x2048    f32/i8 cell         2.7731/1.7369 24.2/38.6 0.81x/1.30x 5 OK
```
Takeaway: i8 beats f32 by ~1.3-1.5x on decode shapes and ~1.2-1.4x on prefill shapes on the
fast units (voodoo/wildcat). Prefill GEMMs run 200-310 GFLOP/s effective — compute-efficient;
decode GEMMs run ~20-33 GFLOP/s — memory-bound, which is exactly where expert-sharding wins.

## 4. Conclusion: which schedule wins, and what it means for pool.cpp

**Expert-shard wins everywhere: 1.97x (T=1) → ~1.55x (T=8) → ~1.15x (T=32).**
The advantage shrinks as T grows because arithmetic intensity rises (prefill reuses weights
across 32 tokens), but it never inverts — even at T=32 the sharded schedule is still ~15% faster
on this 6-core box.

Why: Strata's `run_split_multi` modes 3/4 stripe an equal range of the phase's rows across ALL
experts (`mrows_ = n*FF` then `g0..g1` may span two experts; every worker's range touches every
blob). All 6 cores reread all 8 blobs → ~8x DRAM reread on a DRAM-bound kernel. Expert-shard
gives worker d whole experts (d, d+6, …): each 1.38 MB Q2_0 blob (or native blob) is read once,
stays in that core's L2/L3 slice, and the per-core vnpu device (1 thread) never contends on a
shared device/stream. The 6npu offload replay confirms the same 1.45x on real thread topology.

Implication for `src/kernels/cpu/pool.cpp` (`run_split_multi`, modes 3/4):
- Today (row-split): `mtasks_ = phase_tasks(mrows_)` with `mrows_ = n*FF` (mode 3) / `n*H` (mode 4);
  `drain()` maps task i → global row range g0..g1, which can span experts; `split_multi_[e]`
  buffers are touched by many workers. Good load balance (3 tasks/thread), bad locality.
- Proposed (expert-shard): 1 task (or 1 task-group) per expert; worker claims whole expert e
  (gate/up rows 0..FF then quant then down rows 0..H for that e only). Each blob read once.
  Trade-off: with top_k=8-10 and 5-6 workers, balance is 1-2 experts/worker — fine; tail only
  matters when n_experts < n_workers (single-expert fallback already exists in `run()`).
- Recommended: add a mode (e.g. 7/8 or a `run_shard_multi`) that shards by expert, keep 3/4 for
  the many-expert or single-expert-with-many-tokens case; pick by `n <= 2*n_workers ? shard : split`.
  The vnpu numbers say: shard by default on decode (T<=8), split only helps nothing measured here.

## 5. Strata signatures checked (read-only, no edits made)

pool.hpp:
- `struct ExpertJob { const uint8_t* blob; const ActQ* act; float* out; float weight; int slot; };`
- `struct ExpertJobMulti { const uint8_t* blob; int nt; const ActQ* act[MAXT]; float* out[MAXT]; const void* nact[MAXT]; };`
  with `MAXT = 8` (expert.hpp:183).
- `ExpertPool::run(ExpertJob*,int)` mode 0; `run_split(ExpertJob*,int)` modes 1/2;
  `run_split_multi(ExpertJobMulti*,int)` modes 3/4; `run_split_multi_native(const NativeFmt&,ExpertJobMulti*,int)` modes 5/6.
- Geometry (expert.hpp): `H=2560, FF=640, NE=512, BLOB=1,382,400`; `ActQ{q[H],scale[MAXC],sum[MAXC],hx[MAXC],...}`.
- `act_quant_q8_1(const float*,int,ActQ&)`; `s2_expert_gu_rows(blob,a1,ff,r0,r1)` /
  `s2_expert_down_rows(blob,a2,out,r0,r1)`; multi variants
  `s2_expert_gu_rows_multi(blob,a1,nt,ff,r0,r1)` / `s2_expert_down_rows_multi(blob,a2,nt,out,r0,r1)`.
native_expert.hpp / native_expert.cpp:144,250:
- `struct NativeFmt { gu_type,d_type,gu_act,d_act; n_embd,n_ff; gu_row,d_row; up_off,down_off; bytes; act_bytes,h_bytes; tail_off; }`
- `native_quant_act(f,x,dst)` / `native_quant_h(f,h,dst)`
- `native_gu_rows(f,blob,act,nt,ff,r0,r1)` — ff[t][r]=silu(gate_r.a[t])*(up_r.a[t]), rows [r0,r1)
- `native_down_rows(f,blob,hq,nt,out,r0,r1)` — out[t][r]=down_r.hq[t]
- `native_fmt(gu_type,d_type,n_embd,n_ff,f,err)`, `q2_rows_any(...)` for Q2_0 down.

## 6. Minimal vnpu adapter sketch (NOT implemented — files/functions to touch only)

Goal: 1 vnpu device per pool worker core; per expert: dequant blob rows → quantize_i8 → matmul_i8
gate/up → swiglu → quantize → matmul_i8 down. Keeps Strata's bitwise contract by validating
against `s2_expert_*` / `native_*` row kernels in parity first.

New file (suggested): `src/kernels/cpu/vnpu_expert.cpp` (+ header):
- `vnpu_pool_init(n_workers)` — one `vnpu_device_t*` + stream per worker (num_threads=1),
  mirroring `strata_schedule_compare.cpp:72-74` devs/dst creation; pin to `physical_cores(true)`.
- `vnpu_expert_shard(const NativeFmt or Q2_0 blob, act, nt, out)` — whole-expert path:
  dequant gate/up rows → `vnpu_matmul_i8` (act 1x2048/32x2048 x W 2048x512) → swiglu in fp32 →
  quant → `vnpu_matmul_i8` (ff x down 512x2048). Bench shapes above are exactly these.
- Fallback: keep calling `native_gu_rows`/`native_down_rows` (or `s2_expert_*_multi`) when
  vnpu unavailable or parity fails.

Touch (read existing, then extend — no edits made in this session):
- `src/kernels/cpu/pool.cpp` `run_split_multi` (~line 802) + `drain` modes 3/4 (~line 746):
  add shard-by-expert tasking (task e = expert e whole) alongside row-range tasking.
- `include/strata/kernels/cpu/pool.hpp:187` `run_split_multi` decl + `ExpertJobMulti` (:57):
  add `run_shard_multi` decl; possibly add per-worker device handle (opaque ptr to avoid
  vnpu dep in header).
- `src/kernels/cpu/expert.cpp:472,486` `s2_expert_gu/down_rows_multi` and
  `src/kernels/cpu/native_expert.cpp:144,250` `native_gu/down_rows`: reuse as reference +
  fallback; parity harness like `expert_multi_test.cpp` / `iq_avx2_parity.cpp`.
- Build: Strata CMake — add optional `STRATA_VNPU` link against `C:\npuxl-improved\build-win\libvnpu.dll`
  (or static `libvnpu.a` under MinGW; MSVC import lib needed for the HIP build).
- Validation: `vnpu-cli bench --shape 1x2048x512 --shape 1x512x2048 --dtype both` (decode) and
  `--shape 32x...` (prefill) as the gate before swapping any pool path; then
  `strata_schedule_compare --prefill {1,8,32}` and `strata_offload_6npu --prefill 32` as
  before/after on identical workloads, per the standing rule (baseline preserved above).
