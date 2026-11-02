/* strata_offload_6npu.cpp - replay Strata Qwen3.5 CPU-expert work across 6 NPUs.
 *
 * Model: Qwen3.5-35B-A3B (qwen35moe): n_embd=2048, n_ff_expert=512,
 * top_k=8, n_layers=40, n_experts=256.
 *
 * Per token per layer, the CPU pool runs the MISSED experts (those not in
 * VRAM cache). Each expert = gate (H->FF) + up (H->FF) + SiLU + down (FF->H).
 * Shapes (decode M=1, prefill M=T):
 *   gate/up: Mx2048x512
 *   down:    Mx512x2048
 * FLOPs per expert per row: 2*H*FF*2 (gate+up) + 2*FF*H (down) = 6*H*FF
 *   = 6*2048*512 = 6.29 MFLOP per token.
 * Per token per layer (8 experts): ~50 MFLOP. x40 layers = ~2.0 GFLOP/token.
 *
 * Test A (single NPU, 6 threads) vs Test B (6 NPUs x 1 thread, experts sharded):
 *   decode: 8 experts x [1x2048x512 x2 + 1x512x2048]
 *   prefill: 32 tokens, 8 experts each active -> 8 x [32x2048x512 x2 + 32x512x2048]
 */
#include "vnpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now_ms() {
    using c = std::chrono::steady_clock;
    return std::chrono::duration<double,std::milli>(c::now().time_since_epoch()).count();
}
static uint32_t rng = 0x12345678u;
static float rndf() {
    rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
    return float(int(rng%2000)-1000)/5000.0f;
}

// one expert's 3 GEMMs on a stream; returns ms
static double run_expert(vnpu_stream_t* s, const float* act, const float* gateW,
                         const float* upW, const float* downW,
                         float* gate, float* up, float* ff, float* out,
                         uint32_t M) {
    // gate: Mx2048x512, up: Mx2048x512, down: Mx512x2048
    vnpu_matmul_f32_args_t a{};
    a.A=act; a.B=gateW; a.C=gate; a.M=M; a.K=2048; a.N=512;
    a.alpha=1; a.beta=0; a.unit=VNPU_UNIT_VOODOO;
    double t0=now_ms();
    vnpu_matmul_f32(s,&a);
    a.B=upW; a.C=up; vnpu_matmul_f32(s,&a);
    // SiLU on gate (scalar fallback on host - not timed as GEMM, fused approx)
    for (uint32_t i=0;i<M*512;i++){ float g=gate[i]; ff[i]=(g/(1.0f+expf(-g)))*up[i]; }
    a.A=ff; a.B=downW; a.C=out; a.K=512; a.N=2048; vnpu_matmul_f32(s,&a);
    vnpu_stream_sync(s);
    return now_ms()-t0;
}

int main(int argc, char** argv) {
    uint32_t T = 32;
    uint32_t iters = 5;
    for(int i=0;i<argc;i++){ if(!strcmp(argv[i],"--prefill")&&i+1<argc) T=atoi(argv[++i]);
        if(!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]); }
    const uint32_t H=2048, FF=512, KTOP=8, LAYERS=40;

    // weights: 8 distinct experts (the routed set for one token batch)
    std::vector<float> gateW(KTOP*H*FF), upW(KTOP*H*FF), downW(KTOP*FF*H);
    for(auto&x:gateW)x=rndf(); for(auto&x:upW)x=rndf(); for(auto&x:downW)x=rndf();

    // ---- Test A: single NPU, 6 threads (what Strata pool ~= today) ----
    vnpu_config_t cfg; vnpu_config_init(&cfg); cfg.num_threads=6;
    vnpu_device_t* devA=nullptr; vnpu_stream_t* sA=nullptr;
    vnpu_device_create(&cfg,&devA); vnpu_stream_create(devA,0,&sA);
    vnpu_device_info_t inf{}; vnpu_device_get_info(devA,&inf);
    printf("backend=%s threads=%u\n", inf.backend, inf.num_threads);

    std::vector<float> act(T*H), gate(T*FF), up(T*FF), ff(T*FF), out(T*H);
    for(auto&x:act)x=rndf();

    // decode M=1
    double decA=0; for(uint32_t it=0;it<iters;it++){ double t0=now_ms();
        for(uint32_t e=0;e<KTOP;e++) run_expert(sA,act.data(),gateW.data()+e*H*FF,upW.data()+e*H*FF,downW.data()+e*FF*H,gate.data(),up.data(),ff.data(),out.data(),1);
        decA+=(now_ms()-t0); } decA/=iters;
    // prefill M=T
    double preA=0; for(uint32_t it=0;it<iters;it++){ double t0=now_ms();
        for(uint32_t e=0;e<KTOP;e++) run_expert(sA,act.data(),gateW.data()+e*H*FF,upW.data()+e*H*FF,downW.data()+e*FF*H,gate.data(),up.data(),ff.data(),out.data(),T);
        preA+=(now_ms()-t0); } preA/=iters;
    double flop_dec = double(KTOP)*6*H*FF; // per 1 token
    double flop_pre = double(KTOP)*6*H*FF*T;
    printf("[A single-NPU-6thr] decode 1tok x8exp x1layer: %.3f ms  %.1f GFLOP/s\n", decA, flop_dec/(decA*1e6));
    printf("[A single-NPU-6thr] prefill %utok x8exp x1layer: %.3f ms  %.1f tok/s  %.1f GFLOP/s\n", T, preA, T*1000.0/preA, flop_pre/(preA*1e6));
    printf("[A single-NPU-6thr] full-model est (40 layers): decode %.1f tok/s  prefill-%u %.1f tok/s\n",
        1000.0/(decA*LAYERS), T, T*1000.0/(preA*LAYERS));

    // ---- Test B: 6 NPUs x 1 thread, 8 experts sharded round-robin ----
    const int NDEV=6;
    vnpu_device_t* devs[NDEV]; vnpu_stream_t* streams[NDEV];
    for(int d=0;d<NDEV;d++){ vnpu_config_t c; vnpu_config_init(&c); c.num_threads=1;
        vnpu_device_create(&c,&devs[d]); vnpu_stream_create(devs[d],0,&streams[d]); }
    // per-device scratch
    std::vector<std::vector<float>> g2(NDEV,std::vector<float>(T*FF)),
        u2(NDEV,std::vector<float>(T*FF)), f2(NDEV,std::vector<float>(T*FF)), o2(NDEV,std::vector<float>(T*H));
    auto run_sharded = [&](uint32_t M)->double {
        double t0=now_ms();
        std::vector<std::thread> th;
        std::vector<double> ms(NDEV,0);
        // expert e -> device e%6
        // group experts per device
        for(int d=0;d<NDEV;d++){ th.emplace_back([&,d](){
            for(uint32_t e=d;e<KTOP;e+=NDEV)
                run_expert(streams[d],act.data(),gateW.data()+e*H*FF,upW.data()+e*H*FF,downW.data()+e*FF*H,
                           g2[d].data(),u2[d].data(),f2[d].data(),o2[d].data(),M);
        });}
        for(auto&t:th) t.join();
        // NOTE: wall-clock of parallel shard; sum outputs (moe combine) ignored for timing
        return now_ms()-t0;
    };
    // warmup
    run_sharded(1); run_sharded(T);
    double decB=0; for(uint32_t it=0;it<iters;it++) decB+=run_sharded(1); decB/=iters;
    double preB=0; for(uint32_t it=0;it<iters;it++) preB+=run_sharded(T); preB/=iters;
    printf("[B 6xNPU-1thr] decode 1tok x8exp x1layer: %.3f ms  %.1f GFLOP/s  speedup-vs-A %.2fx\n", decB, flop_dec/(decB*1e6), decA/decB);
    printf("[B 6xNPU-1thr] prefill %utok x8exp x1layer: %.3f ms  %.1f tok/s  %.1f GFLOP/s  speedup %.2fx\n", T, preB, T*1000.0/preB, flop_pre/(preB*1e6), preA/preB);
    printf("[B 6xNPU-1thr] full-model est (40 layers): decode %.1f tok/s  prefill-%u %.1f tok/s\n",
        1000.0/(decB*LAYERS), T, T*1000.0/(preB*LAYERS));

    for(int d=0;d<NDEV;d++){ vnpu_stream_destroy(streams[d]); vnpu_device_destroy(devs[d]); }
    vnpu_stream_destroy(sA); vnpu_device_destroy(devA);
    return 0;
}
