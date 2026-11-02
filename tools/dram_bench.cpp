// dram_bench.cpp - cold-DRAM GEMV vs warm-GEMM matrix per spec.
//
// Usage: dram_bench.exe [threads]
//   threads: 0=auto(6), 1, 3, 6, 12
//
// Matrix:
//   1. Decode GEMV M=1 4096x11008 INT8, COLD (45MB > 32MB L3). Report GB/s, % of 48.0.
//   2. Same shape F32 (180MB). Baseline memory-bound comparison.
//   3. Prefill GEMM M=32 4096x4096 INT8, warm. Report GFLOP/s.
//   4. Per-op overhead M=1 2048x512: stream(cmd) vs direct. Tells if dispatch matters.
#include "vnpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// 256MB eviction buffer: memset before each cold iteration to flush L3 (32MB).
static std::vector<uint8_t> evict_buf;
static void flush_caches() {
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < evict_buf.size(); i += 64) {
        evict_buf[i] = (uint8_t)(i & 0xFF);
        sink ^= evict_buf[i];
    }
    (void)sink;
}

int main(int argc, char** argv) {
    int threads = 0;
    if (argc > 1) threads = std::atoi(argv[1]);
    printf("RAM: DDR4-3000 dual-channel, ceiling 48.0 GB/s\n");
    printf("threads: %d (0=auto)\n", threads);

    evict_buf.resize(256ull * 1024 * 1024, 0);

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

    // --- 1. Decode GEMV M=1 4096x11008 INT8, COLD ---
    {
        const uint32_t M = 1, K = 4096, N = 11008;
        std::vector<float> Af((size_t)M * K), Bf((size_t)K * N), C((size_t)M * N, 0);
        for (auto& x : Af) x = rndf();
        for (auto& x : Bf) x = rndf();
        std::vector<int8_t> A8(Af.size()), B8(Bf.size());
        std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
        vnpu_quantize_i8_args_t qa{};
        qa.X = Af.data(); qa.Y = A8.data(); qa.scales = sa.data();
        qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
        vnpu_quantize_i8(s, &qa);
        vnpu_quantize_i8_args_t qb{};
        qb.X = Bf.data(); qb.Y = B8.data(); qb.scales = sb.data();
        qb.rows = K; qb.cols = N; qb.axis = VNPU_QUANT_PER_COLUMN;
        vnpu_quantize_i8(s, &qb);
        vnpu_packed_weights_t* W = nullptr;
        vnpu_pack_weights_i8(dev, B8.data(), sb.data(), K, N, 0, 0, &W);
        vnpu_matmul_i8_args_t a{};
        a.A = A8.data(); a.B = W; a.scale_a = sa.data(); a.C = C.data();
        a.M = M; a.K = K; a.N = N;
        for (int i = 0; i < 2; ++i) vnpu_matmul_i8(s, &a);  // warmup (loads code, not data)
        double best = 1e30;
        for (int i = 0; i < 5; ++i) {
            flush_caches();
            double t0 = now_ms();
            vnpu_matmul_i8(s, &a);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        double bytes = (double)M * K + (double)K * N + (double)M * N;  // i8 traffic
        double gbs = bytes / (best * 1e6);
        printf("[1] cold decode 1x4096x11008 i8: %.3f ms  %.1f GB/s  %.1f%% of 48.0  %s\n",
               best, gbs, 100.0 * gbs / 48.0, "COLD");
        vnpu_packed_weights_destroy(W);
    }

    // --- 2. Same shape F32, COLD ---
    {
        const uint32_t M = 1, K = 4096, N = 11008;
        std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N, 0);
        for (auto& x : A) x = rndf();
        for (auto& x : B) x = rndf();
        vnpu_matmul_f32_args_t a{};
        a.A = A.data(); a.B = B.data(); a.C = C.data();
        a.M = M; a.K = K; a.N = N; a.alpha = 1; a.beta = 0;
        for (int i = 0; i < 2; ++i) vnpu_matmul_f32(s, &a);
        double best = 1e30;
        for (int i = 0; i < 5; ++i) {
            flush_caches();
            double t0 = now_ms();
            vnpu_matmul_f32(s, &a);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        double bytes = ((double)M * K + (double)K * N + (double)M * N) * 4.0;
        double gbs = bytes / (best * 1e6);
        printf("[2] cold decode 1x4096x11008 f32: %.3f ms  %.1f GB/s  %.1f%% of 48.0  %s\n",
               best, gbs, 100.0 * gbs / 48.0, "COLD");
    }

    // --- 3. Prefill GEMM M=32 4096x4096 INT8, WARM ---
    {
        const uint32_t M = 32, K = 4096, N = 4096;
        std::vector<float> Af((size_t)M * K), Bf((size_t)K * N), C((size_t)M * N, 0);
        for (auto& x : Af) x = rndf();
        for (auto& x : Bf) x = rndf();
        std::vector<int8_t> A8(Af.size()), B8(Bf.size());
        std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
        vnpu_quantize_i8_args_t qa{};
        qa.X = Af.data(); qa.Y = A8.data(); qa.scales = sa.data();
        qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
        vnpu_quantize_i8(s, &qa);
        vnpu_quantize_i8_args_t qb{};
        qb.X = Bf.data(); qb.Y = B8.data(); qb.scales = sb.data();
        qb.rows = K; qb.cols = N; qb.axis = VNPU_QUANT_PER_COLUMN;
        vnpu_quantize_i8(s, &qb);
        vnpu_packed_weights_t* W = nullptr;
        vnpu_pack_weights_i8(dev, B8.data(), sb.data(), K, N, 0, 0, &W);
        vnpu_matmul_i8_args_t a{};
        a.A = A8.data(); a.B = W; a.scale_a = sa.data(); a.C = C.data();
        a.M = M; a.K = K; a.N = N;
        for (int i = 0; i < 3; ++i) vnpu_matmul_i8(s, &a);
        double best = 1e30;
        for (int i = 0; i < 5; ++i) {
            double t0 = now_ms();
            vnpu_matmul_i8(s, &a);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        double flop = 2.0 * M * K * N;
        printf("[3] warm prefill 32x4096x4096 i8: %.3f ms  %.1f GFLOP/s  %s\n",
               best, flop / (best * 1e6), "WARM");
        vnpu_packed_weights_destroy(W);
    }

    // --- 4. Per-op overhead M=1 2048x512: cmd(stream) vs direct ---
    {
        const uint32_t M = 1, K = 2048, N = 512;
        std::vector<float> A((size_t)M * K, 0.5f), B((size_t)K * N, 0.25f), C((size_t)M * N, 0);
        vnpu_matmul_f32_args_t a{};
        a.A = A.data(); a.B = B.data(); a.C = C.data();
        a.M = M; a.K = K; a.N = N; a.alpha = 1; a.beta = 0;
        for (int i = 0; i < 3; ++i) vnpu_matmul_f32(s, &a);
        double t0 = now_ms();
        for (int i = 0; i < 20; ++i) vnpu_matmul_f32(s, &a);
        double cmd = (now_ms() - t0) / 20.0;
        for (int i = 0; i < 3; ++i) vnpu_gemm_f32_direct(A.data(), B.data(), C.data(), M, K, N);
        t0 = now_ms();
        for (int i = 0; i < 20; ++i) vnpu_gemm_f32_direct(A.data(), B.data(), C.data(), M, K, N);
        double direct = (now_ms() - t0) / 20.0;
        printf("[4] overhead 1x2048x512 f32: cmd %.4f ms  direct %.4f ms  ratio %.2fx  %s\n",
               cmd, direct, cmd / direct, "WARM");
    }

    vnpu_stream_destroy(s);
    vnpu_device_destroy(dev);
    return 0;
}
