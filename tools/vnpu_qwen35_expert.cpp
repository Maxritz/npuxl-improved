/* vnpu_qwen35_expert.cpp - replay ONE Qwen3.5 CPU expert exactly.
 *
 * Geometry: H=2048 FF=512, gate/up/down as F32 (random).
 * Per M: gate (Mx2048x512) + up (Mx2048x512) + SiLU + down (Mx512x2048).
 * FLOP per M: 2*M*H*FF (gate) + 2*M*H*FF (up) + 2*M*FF*H (down) = 6*M*H*FF.
 *
 * (A) single device, 6 threads: whole expert on one 6-thread vnpu device.
 * (B) 6 devices, 1 thread each, FF-row sharded: device d owns rows
 *     [d*per,(d+1)*per) of gate/up/down; down partials (MxH each) are
 *     sum-reduced on the host. Weight panels are pre-packed once at setup
 *     (steady-state cost, like a real bridge would); the timed region is
 *     GEMMs + SiLU + reduce only.
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

static const uint32_t H=2048, FF=512;

static void gemm(vnpu_stream_t* s, const float* A, const float* B, float* C,
                 uint32_t M, uint32_t K, uint32_t N) {
    vnpu_matmul_f32_args_t a{};
    a.A=A; a.B=B; a.C=C; a.M=M; a.K=K; a.N=N;
    a.alpha=1; a.beta=0; a.unit=VNPU_UNIT_VOODOO;
    vnpu_matmul_f32(s,&a);
}

int main(int argc, char** argv) {
    uint32_t iters = 5;
    for(int i=0;i<argc;i++)
        if(!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);

    // weights: one expert, F32 random
    std::vector<float> gateW((size_t)H*FF), upW((size_t)H*FF), downW((size_t)FF*H);
    for(auto&x:gateW)x=rndf(); for(auto&x:upW)x=rndf(); for(auto&x:downW)x=rndf();

    // ---- (A) single device, 6 threads ----
    vnpu_config_t cfgA; vnpu_config_init(&cfgA); cfgA.num_threads=6;
    vnpu_device_t* devA=nullptr; vnpu_stream_t* sA=nullptr;
    vnpu_device_create(&cfgA,&devA); vnpu_stream_create(devA,0,&sA);

    // ---- (B) 6 devices x 1 thread ----
    const int ND=6;
    vnpu_device_t* devs[ND]; vnpu_stream_t* sts[ND];
    for(int d=0;d<ND;d++){ vnpu_config_t c; vnpu_config_init(&c); c.num_threads=1;
        vnpu_device_create(&c,&devs[d]); vnpu_stream_create(devs[d],0,&sts[d]); }

    // pre-pack FF-row panels per device (setup, not timed)
    int per = FF/ND;
    std::vector<std::vector<float>> gP(ND), uP(ND), dP(ND);
    std::vector<int> r0v(ND), r1v(ND);
    for(int d=0;d<ND;d++){
        int r0=d*per, r1=(d==ND-1)?(int)FF:r0+per;
        r0v[d]=r0; r1v[d]=r1;
        gP[d].resize((size_t)H*(r1-r0)); uP[d].resize((size_t)H*(r1-r0));
        dP[d].resize((size_t)(r1-r0)*H);
        for(uint32_t h=0;h<H;h++)
            for(int r=r0;r<r1;r++){
                gP[d][h*(r1-r0)+(r-r0)] = gateW[h*FF+r];
                uP[d][h*(r1-r0)+(r-r0)] = upW[h*FF+r];
            }
        for(int r=r0;r<r1;r++)
            for(uint32_t h=0;h<H;h++) dP[d][(r-r0)*H+h] = downW[r*H+h];
    }

    printf("one Qwen3.5 expert H=%u FF=%u F32, iters=%u\n", H, FF, iters);
    printf("%-4s %-12s %-10s %-12s %-10s %-9s %-10s\n",
           "M","A_ms(6thr)","A_GFLOP/s","B_ms(6x1thr)","B_GFLOP/s","speedup","maxdiff");

    for(uint32_t M : {1u, 8u, 32u}) {
        std::vector<float> act((size_t)M*H);
        for(auto&x:act)x=rndf();
        double flop = 6.0*M*H*FF;

        std::vector<float> gate((size_t)M*FF), up((size_t)M*FF),
                           ff((size_t)M*FF), outA((size_t)M*H,0);
        // ---- A: whole expert, 6 threads ----
        auto runA = [&](){
            gemm(sA, act.data(), gateW.data(), gate.data(), M, H, FF);
            gemm(sA, act.data(), upW.data(), up.data(), M, H, FF);
            for(size_t i=0;i<(size_t)M*FF;i++){ float g=gate[i]; ff[i]=(g/(1.0f+expf(-g)))*up[i]; }
            gemm(sA, ff.data(), downW.data(), outA.data(), M, FF, H);
            vnpu_stream_sync(sA);
        };
        runA();
        double tA=0; for(uint32_t i=0;i<iters;i++){ double t0=now_ms(); runA(); tA+=now_ms()-t0; } tA/=iters;

        // ---- B: 6x1-thread, FF-row sharded ----
        std::vector<std::vector<float>> gS(ND), uS(ND), fS(ND), pS(ND);
        for(int d=0;d<ND;d++){ size_t n=(size_t)M*(r1v[d]-r0v[d]);
            gS[d].assign(n,0); uS[d].assign(n,0); fS[d].assign(n,0);
            pS[d].assign((size_t)M*H,0); }
        std::vector<float> outB((size_t)M*H,0);
        auto runB = [&](){
            std::vector<std::thread> th;
            for(int d=0;d<ND;d++){ th.emplace_back([&,d](){
                int nr = r1v[d]-r0v[d];
                gemm(sts[d], act.data(), gP[d].data(), gS[d].data(), M, H, nr);
                gemm(sts[d], act.data(), uP[d].data(), uS[d].data(), M, H, nr);
                for(size_t i=0;i<(size_t)M*nr;i++){ float g=gS[d][i]; fS[d][i]=(g/(1.0f+expf(-g)))*uS[d][i]; }
                gemm(sts[d], fS[d].data(), dP[d].data(), pS[d].data(), M, nr, H);
                vnpu_stream_sync(sts[d]);
            });}
            for(auto&t:th) t.join();
            std::fill(outB.begin(), outB.end(), 0.0f);
            for(int d=0;d<ND;d++)
                for(size_t i=0;i<(size_t)M*H;i++) outB[i]+=pS[d][i];
        };
        runB();
        double tB=0; for(uint32_t i=0;i<iters;i++){ double t0=now_ms(); runB(); tB+=now_ms()-t0; } tB/=iters;

        double maxdiff=0;
        for(size_t i=0;i<(size_t)M*H;i++)
            maxdiff = fmax(maxdiff, (double)fabsf(outA[i]-outB[i]));
        printf("%-4u %-12.3f %-10.1f %-12.3f %-10.1f %-9.2fx %-10.2e\n",
               M, tA, flop/(tA*1e6), tB, flop/(tB*1e6), tA/tB, maxdiff);
    }

    for(int d=0;d<ND;d++){ vnpu_stream_destroy(sts[d]); vnpu_device_destroy(devs[d]); }
    vnpu_stream_destroy(sA); vnpu_device_destroy(devA);
    return 0;
}
