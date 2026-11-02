// bench direct fast path vs command-layer path, same shapes, same threads.
#include "vnpu.h"
#include <chrono>
#include <cstdio>
#include <vector>

static double now_ms() {
    using c = std::chrono::steady_clock;
    return std::chrono::duration<double,std::milli>(c::now().time_since_epoch()).count();
}

int main() {
    vnpu_config_t cfg; vnpu_config_init(&cfg); cfg.num_threads = 6;
    vnpu_device_t* dev = nullptr; vnpu_stream_t* s = nullptr;
    vnpu_device_create(&cfg, &dev); vnpu_stream_create(dev, 0, &s);
    struct Shape { uint32_t M,K,N; const char* n; };
    Shape shapes[] = {{1,2048,512,"1x2048x512"},{1,512,2048,"1x512x2048"},
                      {32,2048,512,"32x2048x512"},{32,512,2048,"32x512x2048"}};
    for (auto& sh : shapes) {
        std::vector<float> A((size_t)sh.M*sh.K, 0.5f), B((size_t)sh.K*sh.N, 0.25f), C((size_t)sh.M*sh.N, 0.f);
        // warmup
        for (int i = 0; i < 3; ++i) vnpu_gemm_f32_direct(A.data(), B.data(), C.data(), sh.M, sh.K, sh.N);
        double t0 = now_ms();
        for (int i = 0; i < 20; ++i) vnpu_gemm_f32_direct(A.data(), B.data(), C.data(), sh.M, sh.K, sh.N);
        double direct = (now_ms() - t0) / 20.0;
        vnpu_matmul_f32_args_t a{};
        a.A = A.data(); a.B = B.data(); a.C = C.data();
        a.M = sh.M; a.K = sh.K; a.N = sh.N; a.alpha = 1; a.beta = 0;
        for (int i = 0; i < 3; ++i) vnpu_matmul_f32(s, &a);
        t0 = now_ms();
        for (int i = 0; i < 20; ++i) vnpu_matmul_f32(s, &a);
        double cmd = (now_ms() - t0) / 20.0;
        double flop = 2.0 * sh.M * sh.K * sh.N;
        printf("%-14s direct %8.4f ms %7.1f GFLOP/s | cmd %8.4f ms %7.1f GFLOP/s | speedup %.2fx\n",
               sh.n, direct, flop/(direct*1e6), cmd, flop/(cmd*1e6), cmd/direct);
    }
    vnpu_stream_destroy(s); vnpu_device_destroy(dev);
    return 0;
}
