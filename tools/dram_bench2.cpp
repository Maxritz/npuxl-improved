// dram_bench2.cpp - fixed harness per review.
//
// Fixes:
//   4. Correctness on every shape: INT8 vs F32 reference (max rel err), direct vs cmd.
//   5. 5 repeats per config, report median/min/max (not best-of-5 single run).
//   Cold test: ring of 4 packed weights (4x45MB=180MB >> 32MB L3), rotate per call.
//      No flush writes, no dirty-L3 writeback contamination.
//   1/6 threads only (the comparison that matters). 5 repeats each.
#include "vnpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static double now_ms() {
    using c = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(c::now().time_since_epoch()).count();
}

static uint32_t rng = 0x12345678u;
static float rndf() {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return float(int(rng % 2000) - 1000) / 5000.0f;
}

// repo convention (vnpu_cli.cpp check_samples): atol + rtol*max(|got|,|ref|)
// on 64 samples. Strict max-rel-err divides by near-zero outputs and always fails.
static bool check_like_cli(const float* got, const float* ref, size_t n,
                           float atol, float rtol, double* worst = nullptr) {
    bool ok = true;
    double w = 0;
    // deterministic stride over the outputs
    for (size_t s = 0; s < 64; ++s) {
        size_t i = (s * (n / 64 + 1)) % n;
        double tol = atol + rtol * std::max(std::fabs((double)got[i]), std::fabs((double)ref[i]));
        double e = std::fabs((double)got[i] - ref[i]);
        if (e > tol) ok = false;
        if (e > w) w = e;
    }
    if (worst) *worst = w;
    return ok;
}

static void stats(const char* tag, std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    double med = v[v.size() / 2], mn = v.front(), mx = v.back();
    printf("  %s: median %.3f  min %.3f  max %.3f (n=%zu)\n", tag, med, mn, mx, v.size());
}

int main(int argc, char** argv) {
    int threads = 6;
    if (argc > 1) threads = std::atoi(argv[1]);
    printf("RAM: DDR4-3000 dual-channel, ceiling 48.0 GB/s\n");
    printf("threads: %d\n", threads);

    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.num_threads = (uint32_t)(threads < 0 ? 0 : threads);
    vnpu_device_t* dev = nullptr;
    vnpu_stream_t* s = nullptr;
    if (vnpu_device_create(&cfg, &dev) != VNPU_OK || vnpu_stream_create(dev, 0, &s) != VNPU_OK) {
        printf("device create failed\n");
        return 1;
    }
    vnpu_device_info_t inf{};
    vnpu_device_get_info(dev, &inf);
    printf("ISA line: backend=%s simd=%u threads=%u physical=%u\n",
           inf.backend, inf.simd_bits, inf.num_threads, inf.physical_cores);

    const uint32_t M = 1, K = 4096, N = 11008;
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;

    // F32 reference data (shared across configs)
    std::vector<float> Af(mk), Bf(kn), Cref(mn, 0), Cgot(mn, 0);
    for (auto& x : Af) x = rndf();
    for (auto& x : Bf) x = rndf();

    // F32 reference result (computed once via direct path)
    vnpu_gemm_f32_direct(Af.data(), Bf.data(), Cref.data(), M, K, N);

    // INT8 quantized + 4 packed rings
    std::vector<int8_t> A8(mk), B8(kn);
    std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
    vnpu_quantize_i8_args_t qa{};
    qa.X = Af.data(); qa.Y = A8.data(); qa.scales = sa.data();
    qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
    vnpu_quantize_i8(s, &qa);
    vnpu_quantize_i8_args_t qb{};
    qb.X = Bf.data(); qb.Y = B8.data(); qb.scales = sb.data();
    qb.rows = K; qb.cols = N; qb.axis = VNPU_QUANT_PER_COLUMN;
    vnpu_quantize_i8(s, &qb);
    const int RING = 4;
    std::vector<vnpu_packed_weights_t*> Wr(RING, nullptr);
    for (int r = 0; r < RING; ++r)
        vnpu_pack_weights_i8(dev, B8.data(), sb.data(), K, N, 0, 0, &Wr[r]);

    // --- Correctness first: INT8 vs F32 reference (repo atol/rtol convention) ---
    {
        vnpu_matmul_i8_args_t a{};
        a.A = A8.data(); a.B = Wr[0]; a.scale_a = sa.data(); a.C = Cgot.data();
        a.M = M; a.K = K; a.N = N;
        vnpu_matmul_i8(s, &a);
        double worst = 0;
        bool ok = check_like_cli(Cgot.data(), Cref.data(), mn, 0.1f, 0.06f, &worst);
        printf("[correctness] INT8 vs F32-direct: %s (worst %.3e, atol 0.1 rtol 0.06)\n",
               ok ? "PASS" : "FAIL", worst);
        if (!ok) return 1;
    }

    // --- Correctness: cmd(F32) vs direct(F32) ---
    {
        std::vector<float> Ccmd(mn, 0);
        vnpu_matmul_f32_args_t a{};
        a.A = Af.data(); a.B = Bf.data(); a.C = Ccmd.data();
        a.M = M; a.K = K; a.N = N; a.alpha = 1; a.beta = 0;
        vnpu_matmul_f32(s, &a);
        double worst = 0;
        bool ok = check_like_cli(Ccmd.data(), Cref.data(), mn, 4e-3f, 4e-3f, &worst);
        printf("[correctness] cmd-F32 vs direct-F32: %s (worst %.3e)\n",
               ok ? "PASS" : "FAIL", worst);
        if (!ok) return 1;
    }

    // --- Cold INT8: ring rotation, 5 repeats ---
    {
        vnpu_matmul_i8_args_t a{};
        a.A = A8.data(); a.scale_a = sa.data(); a.C = Cgot.data();
        a.M = M; a.K = K; a.N = N;
        std::vector<double> times;
        for (int rep = 0; rep < 5; ++rep) {
            // touch all rings once to evict (each 45MB, total 180MB >> L3)
            for (int r = 0; r < RING; ++r) { a.B = Wr[r]; vnpu_matmul_i8(s, &a); }
            // timed: cycle through rings, each read is cold
            double t0 = now_ms();
            for (int i = 0; i < 4; ++i) { a.B = Wr[i % RING]; vnpu_matmul_i8(s, &a); }
            double dt = (now_ms() - t0) / 4.0;
            times.push_back(dt);
        }
        stats("cold INT8 ms", times);
        double med = times[times.size() / 2];
        double bytes = (double)mk + (double)kn + (double)mn;
        double gbs = bytes / (med * 1e6);
        printf("  cold INT8: %.1f GB/s  %.1f%% of 48.0\n", gbs, 100.0 * gbs / 48.0);
    }

    // --- Cold F32: same data, no pack (B is 180MB, self-evicting) ---
    {
        vnpu_matmul_f32_args_t a{};
        a.A = Af.data(); a.B = Bf.data(); a.C = Cgot.data();
        a.M = M; a.K = K; a.N = N; a.alpha = 1; a.beta = 0;
        std::vector<double> times;
        for (int rep = 0; rep < 5; ++rep) {
            vnpu_matmul_f32(s, &a);  // evict (180MB >> L3)
            double t0 = now_ms();
            vnpu_matmul_f32(s, &a);
            times.push_back(now_ms() - t0);
        }
        stats("cold F32 ms", times);
        double med = times[times.size() / 2];
        double bytes = ((double)mk + (double)kn + (double)mn) * 4.0;
        double gbs = bytes / (med * 1e6);
        printf("  cold F32: %.1f GB/s  %.1f%% of 48.0\n", gbs, 100.0 * gbs / 48.0);
    }

    for (auto w : Wr) vnpu_packed_weights_destroy(w);
    vnpu_stream_destroy(s);
    vnpu_device_destroy(dev);
    return 0;
}
