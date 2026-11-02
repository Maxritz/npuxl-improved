/* strata_schedule_compare.cpp - row-split (Strata pool today) vs expert-shard (your idea).
 *
 * Same work both ways: 8 experts x [gate Mx2048x512 + up Mx2048x512 + SiLU + down Mx512x2048].
 *   Row-split (Strata mode 3/4): every worker touches every expert's rows -> all cores reread all blobs.
 *   Expert-shard (your idea): worker d owns experts d, d+6, ... -> each blob read once.
 * 6 workers, F32 voodoo, Qwen3.5 shapes (H=2048 FF=512 top_k=8).
 */
#include "vnpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
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

static const uint32_t H=2048, FF=512, KTOP=8;
static std::vector<float> gateW, upW, downW, act;
static std::vector<std::vector<float>> gateBuf, upBuf, ffBuf, outBuf;

static void gemm(vnpu_stream_t* s, const float* A, const float* B, float* C,
                 uint32_t M, uint32_t K, uint32_t N) {
    vnpu_matmul_f32_args_t a{};
    a.A=A; a.B=B; a.C=C; a.M=M; a.K=K; a.N=N;
    a.alpha=1; a.beta=0; a.unit=VNPU_UNIT_VOODOO;
    vnpu_matmul_f32(s,&a);
}

// one row-range of gate/up for ALL experts (Strata mode 3: rows striped across the phase)
static void rows_gu(vnpu_stream_t* s, int wid, uint32_t M, int r0, int r1) {
    for (uint32_t e=0;e<KTOP;e++)
        for (int r=r0;r<r1;r++) {
            // row r of gate: act[M,H] x gateW[e][:,r] -> gateBuf[wid][:,r]
            // implemented as 1xH x Hx1 per row per token -> use GEMV-shaped GEMM M x H x 1
            const float* Gw = gateW.data()+e*H*FF + r;      // column r, stride FF
            const float* Uw = upW.data()+e*H*FF + r;
            // gather column into contiguous (panel) then GEMM
            thread_local std::vector<float> colG, colU;
            if (colG.size()<(size_t)H) { colG.resize(H); colU.resize(H); }
            for (uint32_t h=0;h<H;h++){ colG[h]=gateW[e*H*FF+h*FF+r]; colU[h]=upW[e*H*FF+h*FF+r]; }
            gemm(s, act.data(), colG.data(), gateBuf[wid].data()+r, M, H, 1);
            gemm(s, act.data(), colU.data(), upBuf[wid].data()+r, M, H, 1);
        }
    vnpu_stream_sync(s);
}

int main(int argc, char** argv) {
    uint32_t T = 8;
    uint32_t iters = 3;
    for(int i=0;i<argc;i++){ if(!strcmp(argv[i],"--prefill")&&i+1<argc) T=atoi(argv[++i]);
        if(!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]); }
    gateW.resize((size_t)KTOP*H*FF); upW.resize((size_t)KTOP*H*FF); downW.resize((size_t)KTOP*FF*H);
    for(auto&x:gateW)x=rndf(); for(auto&x:upW)x=rndf(); for(auto&x:downW)x=rndf();
    act.resize((size_t)T*H); for(auto&x:act)x=rndf();

    const int NW=6;
    vnpu_config_t cfg; vnpu_config_init(&cfg); cfg.num_threads=6;
    vnpu_device_t* dev=nullptr; vnpu_stream_t* shared=nullptr;
    vnpu_device_create(&cfg,&dev); vnpu_stream_create(dev,0,&shared);
    // per-worker streams on the SAME device (Strata pool today: one pool, shared work)
    vnpu_stream_t* wst[NW]; for(int i=0;i<NW;i++) vnpu_stream_create(dev,0,&wst[i]);
    // per-NPU devices (your idea: one device per core)
    vnpu_device_t* devs[NW]; vnpu_stream_t* dst[NW];
    for(int i=0;i<NW;i++){ vnpu_config_t c; vnpu_config_init(&c); c.num_threads=1;
        vnpu_device_create(&c,&devs[i]); vnpu_stream_create(devs[i],0,&dst[i]); }
    gateBuf.assign(NW,std::vector<float>((size_t)T*FF,0));
    upBuf.assign(NW,std::vector<float>((size_t)T*FF,0));
    ffBuf.assign(NW,std::vector<float>((size_t)T*FF,0));
    outBuf.assign(NW,std::vector<float>((size_t)T*H,0));

    double flop = double(KTOP)*6*H*FF*T;

    // ---- Test R: row-split across 6 workers on ONE device (Strata pool schedule) ----
    auto run_rowsplit = [&]()->double {
        double t0=now_ms();
        std::vector<std::thread> th;
        int per = FF/NW;
        for(int w=0;w<NW;w++){ th.emplace_back([&,w](){
            int r0=w*per, r1=(w==NW-1)?FF:r0+per;
            // gate/up rows for ALL 8 experts striped; use full-GEMM per expert then slice?
            // faithful striping: each worker runs its row-range via strided GEMMs
            for(uint32_t e=0;e<KTOP;e++){
                // rows r0..r1 of expert e: MxH x Hx(r1-r0) using row-major sub-block
                // B is H rows x FF cols; sub-block cols r0..r1: need packed panel
                thread_local std::vector<float> panel;
                panel.resize((size_t)H*(r1-r0));
                for(uint32_t h=0;h<H;h++)
                    for(int r=r0;r<r1;r++) panel[h*(r1-r0)+(r-r0)] = gateW[e*H*FF+h*FF+r];
                gemm(wst[w], act.data(), panel.data(), gateBuf[w].data()+r0, T, H, r1-r0);
                for(uint32_t h=0;h<H;h++)
                    for(int r=r0;r<r1;r++) panel[h*(r1-r0)+(r-r0)] = upW[e*H*FF+h*FF+r];
                gemm(wst[w], act.data(), panel.data(), upBuf[w].data()+r0, T, H, r1-r0);
            }
        });}
        for(auto&t:th) t.join();
        vnpu_stream_sync(shared);
        return now_ms()-t0;
    };
    // ---- Test E: expert-shard across 6 devices (your idea) ----
    auto run_expertshard = [&]()->double {
        double t0=now_ms();
        std::vector<std::thread> th;
        for(int w=0;w<NW;w++){ th.emplace_back([&,w](){
            for(uint32_t e=w;e<KTOP;e+=NW){
                gemm(dst[w], act.data(), gateW.data()+e*H*FF, gateBuf[w].data(), T, H, FF);
                gemm(dst[w], act.data(), upW.data()+e*H*FF, upBuf[w].data(), T, H, FF);
                for(uint32_t i=0;i<T*FF;i++){ float g=gateBuf[w][i]; ffBuf[w][i]=(g/(1.0f+expf(-g)))*upBuf[w][i]; }
                gemm(dst[w], ffBuf[w].data(), downW.data()+e*FF*H, outBuf[w].data(), T, FF, H);
            }
        });}
        for(auto&t:th) t.join();
        return now_ms()-t0;
    };
    run_rowsplit(); run_expertshard();
    double rR=0,rE=0;
    for(uint32_t i=0;i<iters;i++) rR+=run_rowsplit(); rR/=iters;
    for(uint32_t i=0;i<iters;i++) rE+=run_expertshard(); rE/=iters;
    printf("T=%u iters=%u flop=%.2f GFLOP\n", T, iters, flop/1e9);
    printf("[R row-split 6wk/1dev] %.3f ms  %.1f GFLOP/s\n", rR, flop/(rR*1e6));
    printf("[E expert-shard 6dev]  %.3f ms  %.1f GFLOP/s  speedup %.2fx\n", rE, flop/(rE*1e6), rR/rE);

    for(int i=0;i<NW;i++){ vnpu_stream_destroy(wst[i]); vnpu_stream_destroy(dst[i]); vnpu_device_destroy(devs[i]); }
    vnpu_stream_destroy(shared); vnpu_device_destroy(dev);
    return 0;
}
