# vNPU 6-device CPU-expert bridge - build log

Date: 2026-11-02. Repo: UnbiasedStrata (not a git repo; no commits).
Build dir: build-hip-gfx1031 (Ninja, Release, ROCm HIP, gfx1031).

## What this session did

A parallel subagent had already landed part of this bridge before this
session started (found in tree, NOT written here):
- `include/strata/kernels/cpu/vnpu_expert.hpp` - VnpuShard class +
  `vnpu_bridge_enabled()`.
- `src/kernels/cpu/vnpu_smoke_test.cpp` - smoke harness (Q4_0 synthetic
  expert, compares bridge vs native path within 1e-3).
- `src/kernels/cpu/pool.cpp` drain mode 8 hook: tries
  `VnpuShard::instance().run_native(...)` when `mj.f32[0] != nullptr`,
  else falls through to the exact kernels (rounding unchanged).
- F32 activation plumbing: `ExpertJobMulti.f32[MAXT]`
  (pool.hpp), filled in expert_source.cpp, prefill.cpp, q36_engine.cu
  (null-safe; exact path ignores it).
- `CMakeLists.txt` already lists `src/kernels/cpu/vnpu_expert.cpp`
  in `strata_kernels_cpu` sources (no hard link to libvnpu).

This session wrote:
- `src/kernels/cpu/vnpu_expert.hpp` - free-function API header
  (`vnpu_available`, `vnpu_devices`, `vnpu_run_expert_native`) next to
  the VnpuShard header. Small overlap: one extra header file kept so
  the task-spec API exists as written.
- `src/kernels/cpu/vnpu_expert.cpp` - the implementation (this file's
  earlier draft was overwritten). Implements BOTH APIs over one
  loader + device set:
  - LoadLibrary on `C:\npuxl-improved\build-win\libvnpu.dll`
    (`STRATA_VNPU_DLL` overrides, `STRATA_VNPU=0` disables).
  - want_devices: `STRATA_VNPU_DEVICES` or 6 (1 per physical core).
    One `vnpu_device_t` + stream per device, `cfg.num_threads = 1`.
    One mutex per device (a stream is ordered, not concurrent).
  - Whole-expert path: dequant gate/up/down rows via ggml `to_float`
    (+ NVFP4 tail scales, same values native_expert.cpp uses);
    activations from caller F32 (`f32`, the VnpuShard path) or
    dequanted from `nact` (free API); `vnpu_matmul_f32` gate/up
    (trans_b=1), SiLU in F32 on CPU, `vnpu_matmul_f32` down, scatter
    per-token rows. False on: no DLL, missing export,
    `q2_native_kernels` Q2 pack (type 42 has no ggml to_float),
    bad geometry, nulls. Caller falls back to the exact kernels.

## API decisions (reconciled with the parallel session)

- Two APIs coexist: the task-spec free functions
  (`vnpu_available/devices/run_expert_native`, in
  `src/kernels/cpu/vnpu_expert.hpp`) and the landed `VnpuShard`
  class (in `include/strata/kernels/cpu/vnpu_expert.hpp`) that
  pool.cpp mode 8 actually calls. Both share one loader/devices.
- Worker->device mapping: `worker % ndev`, host (-1) -> device 0
  (class path); first-call sticky thread-local slot (free path).
  Do NOT mix both APIs from the same threads in one run.
- Bitwise contract: NOT claimed for the F32 replay (different
  evaluation order vs vec_dot). Pool keeps exact kernels as
  fallback; parity harness compares (smoke test allows 1e-3 abs).
  Task text asked to "validate vs native in parity first"; the
  comparison lives in `vnpu_smoke_test.cpp`, not wired to CTest.
- `ne < nt || nff < nt` guard: temp buffers borrow `sc.down`
  (`ne*nff >= nt*nff` and `>= nt*ne` need `ne,nff >= nt`;
  always true in practice: nt<=8).

## Build results

1. Standalone clang++ syntax check (exact task-spec command):
   `clang++.exe -std=c++20 -O2 -DNDEBUG -fms-runtime-lib=dll
   -DSTRATA_NATIVE_EXPERTS=1 -I include -I src/kernels/cpu
   -I third_party/llama.cpp/ggml/include
   -I C:/npuxl-improved/include -c src/kernels/cpu/vnpu_expert.cpp`
   -> exit 0. Only warning: `getenv` deprecation (3x, codebase-wide
   pattern; same in native_expert.cpp/pool.cpp). No commit made.
2. Full lib rebuild: `ninja strata_kernels_cpu` in build-hip-gfx1031
   -> exit 0, `vnpu_expert.cpp.obj` compiled, lib relinked.
   (First run said "no work to do": build.ninja already listed the
   file from the parallel session but the obj was newer than my
   just-written cpp; bumped mtime to force the real compile.)
3. DLL export check (`llvm-readobj --coff-exports libvnpu.dll`):
   all 6 loaded symbols present: vnpu_config_init,
   vnpu_device_create/destroy, vnpu_stream_create/destroy,
   vnpu_matmul_f32.
4. NOT run: vnpu_smoke_test link/run, shard_parity, full engine
   build, DLL-at-runtime init. Next step: link the smoke test
   (it is not yet in CMakeLists `add_executable`) and run it with
   and without the DLL present (fallback path), then a real-model
   parity run before enabling by default.

## Files created/modified this session

- created: `src/kernels/cpu/vnpu_expert.hpp` (free API)
- created(overwrote stub): `src/kernels/cpu/vnpu_expert.cpp`
- created: this results file.
- modified: none (pool.cpp hook, CMake entry, f32 plumbing
  pre-existed from the parallel session; left untouched).
